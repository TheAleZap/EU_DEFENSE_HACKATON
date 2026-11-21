"""Siesta Ch 9 predictor: fused, always-consistent wake + class (pairs with train_cv.py).

Rule: p = mean(sigmoid(wake_logit), softmax(sound_class_logits)[drone_buzz])
      wake_triggered = p >= wake_threshold (read from the ONNX metadata, or --thr)
      sound_class = "drone_buzz" if wake else the most likely non-drone class

Output: kit JSONL {"id", "wake_triggered", "sound_class"} -> score with
  python <kit>/starter-kit/evaluate_submissions.py acoustic --gt <kit>/dataset/valid/labels.jsonl --preds preds.jsonl
(Valid clips were used in CV training, so a valid score from the final model is optimistic;
report the out-of-fold numbers from cv_report.json instead.)

Usage:
  python predict_consistent.py --challenge-dir <challenge-5-acoustic-visual-cuas> --onnx runs/c9_cv/audio_wake_cnn.onnx --split test --out c9_test.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

SOUND_CLASSES = ("drone_buzz", "vehicle_mechanical", "explosion", "ambient_outdoor")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--challenge-dir", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--thr", type=float, default=None, help="override the fused threshold")
    args = ap.parse_args()

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    meta = sess.get_modelmeta().custom_metadata_map
    thr = args.thr if args.thr is not None else float(meta["wake_threshold"])
    if meta.get("siesta_rule") != "fused_mean" and args.thr is None:
        print("WARNING: model was not trained by siesta train_cv.py; its threshold is for wake prob alone.")
    in_name = sess.get_inputs()[0].name
    names = [o.name for o in sess.get_outputs()]
    di = SOUND_CLASSES.index("drone_buzz")
    rows = [json.loads(l) for l in (args.challenge_dir / "dataset" / args.split / "labels.jsonl").read_text().splitlines() if l.strip()]
    lines, debug = [], []
    for r in rows:
        with Image.open(args.challenge_dir / r["spectrogram"]) as im:
            x = np.asarray(im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR), np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))[None]
        outs = dict(zip(names, sess.run(names, {in_name: x})))
        pw = float(1 / (1 + np.exp(-np.clip(outs["wake_logit"][0], -40, 40))))
        lg = outs["sound_class_logits"][0]
        pc = np.exp(lg - lg.max()); pc /= pc.sum()
        fused = 0.5 * (pw + float(pc[di]))
        wake = fused >= thr
        if wake:
            cls = "drone_buzz"
        else:
            q = pc.copy(); q[di] = -1
            cls = SOUND_CLASSES[int(np.argmax(q))]
        lines.append(json.dumps({"id": r["id"], "wake_triggered": bool(wake), "sound_class": cls}))
        debug.append({"id": r["id"], "wake_prob": round(pw, 4), "p_drone_class": round(float(pc[di]), 4),
                      "fused": round(fused, 4)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"clips": len(lines), "fused_thr": thr, "wakes": sum('"wake_triggered": true' in l for l in lines)}))
    for d in debug:
        print(d)


if __name__ == "__main__":
    main()
