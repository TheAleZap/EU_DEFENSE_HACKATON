"""Siesta cascade demo: listen with the Ch 9 CNN, look with the Ch 6 detector only when a drone is heard.

For one clip folder (audio.wav + video.mp4, e.g. an audio_export clip):
  1. split the audio into 5 s windows and run the Ch 9 CNN (distill_cnn.py ONNX, its own spectrograms)
  2. camera is ON for windows whose class is drone_buzz, OFF otherwise
  3. during ON windows, run the Ch 6 RF-DETR ONNX on every --every-th frame and draw its boxes
Writes an annotated mp4 (with the original sound if ffmpeg is installed) and a JSON timeline.
This is a demo on outside footage, not a score.

  python ch9_acoustic_wake/cascade_demo.py --clip ~/edth/extra/audio_export/787248 \
      --c9-onnx ~/edth/runs/c9_distill_e20/audio_wake_cnn.onnx \
      --c6-kit $C6 --c6-onnx $C6/starter-kit/exports/c6_base20/full/inference_model.onnx --out demo_787248.mp4
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ch6_detection_tracking"))
from distill_cnn import CLASSES, SR, WIN, load_wav, spectrogram  # noqa: E402
from tiling import Detector, classwise_nms, load_kit  # noqa: E402

C6_NAMES = {0: "drone", 1: "airplane", 2: "helicopter"}


def listen(onnx_path: Path, audio: np.ndarray):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = []
    for i in range(max(1, int(np.ceil(len(audio) / WIN)))):
        seg = audio[i * WIN:(i + 1) * WIN]
        if i > 0 and len(seg) < SR:
            break
        _, cl = sess.run(None, {"spectrogram": spectrogram(seg)[None].numpy()})
        p = np.exp(cl[0] - cl[0].max()); p /= p.sum()
        k = int(p.argmax())
        out.append({"start_s": i * WIN / SR, "end_s": min(len(audio), (i + 1) * WIN) / SR,
                    "sound_class": CLASSES[k], "p": round(float(p[k]), 3), "camera_on": k == 0})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", type=Path, required=True, help="folder with audio.wav and video.mp4")
    ap.add_argument("--c9-onnx", type=Path, required=True)
    ap.add_argument("--c6-kit", type=Path, required=True)
    ap.add_argument("--c6-onnx", type=Path, required=True)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--score-thr", type=float, default=0.5)
    ap.add_argument("--every", type=int, default=2, help="run the detector on every n-th frame while ON")
    ap.add_argument("--always-on", action="store_true", help="ignore the audio gate (comparison)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    windows = listen(args.c9_onnx, load_wav(args.clip / "audio.wav"))
    det = Detector(args.c6_onnx, load_kit(args.c6_kit / "starter-kit"), img_size=args.img_size)

    cap = cv2.VideoCapture(str(args.clip / "video.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    silent = args.out.with_suffix(".silent.mp4")
    vw = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    t, frames, on_frames, boxes, last = 0, 0, 0, 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        sec = t / fps
        w = next((x for x in windows if x["start_s"] <= sec < x["end_s"]), windows[-1])
        on = args.always_on or w["camera_on"]
        if on:
            on_frames += 1
            if t % args.every == 0:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                last = [d for d in classwise_nms(det.run(pil, top_k=30)) if d[1] >= args.score_thr]
                boxes += len(last)
            for c, s, (x, y, bw, bh) in last:
                cv2.rectangle(frame, (int(x), int(y)), (int(x + bw), int(y + bh)), (0, 0, 255), 2)
                cv2.putText(frame, f"{C6_NAMES.get(c, c)} {s:.2f}", (int(x), max(12, int(y) - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            last = []
            frame = (frame * 0.45).astype(np.uint8)
        banner = f"{'CAMERA ON ' if on else 'CAMERA OFF'} | heard: {w['sound_class']} {w['p']:.2f} | t={sec:4.1f}s"
        cv2.rectangle(frame, (0, 0), (W, 28), (0, 0, 0), -1)
        cv2.putText(frame, banner, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if on else (160, 160, 160), 2)
        vw.write(frame)
        t += 1; frames += 1
    vw.release(); cap.release()

    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(silent), "-i", str(args.clip / "audio.wav"),
                        "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                        "-shortest", str(args.out)], check=True)
        silent.unlink()
    else:
        silent.rename(args.out)
    summary = {"clip": args.clip.name, "frames": frames, "camera_on_frac": round(on_frames / max(1, frames), 3),
               "detector_calls": det.calls, "boxes_drawn": boxes, "windows": windows}
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "windows"}))
    print(" ".join(f"{w['start_s']:.0f}s:{'ON' if w['camera_on'] else 'off'}({w['sound_class'][:5]})" for w in windows))


if __name__ == "__main__":
    main()
