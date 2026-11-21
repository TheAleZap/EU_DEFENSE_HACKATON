"""Siesta Ch 6: per-frame detections on the tracking sequences (input for track.py).

Modes
  whole  : one whole-frame pass per frame, the same detection the kit's MOT baseline uses (1 call/frame).
           With the 20-epoch untiled model this scores higher on valid than tiled detection.
  full   : every native tile + whole-frame pass on every frame (expensive)
  motion : "peek before you stare" (idea B). For the EFFICIENCY story; use FULL mode for the scored submission
           unless motion mode scores as well on valid (tested coverage: seq 2: 97%, seq 3: 95%, seq 4: only 19% with 12 windows). Detector runs only on native windows around
           (a) the strongest camera-motion-compensated motion candidates,
           (b) the previous frame's confident detections (keeps slow/hovering targets), and
           (c) a full tile sweep every --sweep-every frames.
           Prints model calls per frame, our compute-saving number for the pitch.

Output JSON: {"seq_0001": {"1": [[cls, score, x, y, w, h], ...], ...}, ..., "_stats": {...}}
Frames are 1-based to match MOT.

Usage:
  python detect_video.py --kit <challenge-1-cuas-tracking> --onnx inference_model.onnx --mode motion --out dets_motion.json
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # find sibling modules

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from tiling import TILE, Detector, classwise_nms, detect_tiled, load_kit, window_around


def read_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def _align(src_g, dst_g, scale=0.5):
    s = cv2.resize(src_g, None, fx=scale, fy=scale)
    d = cv2.resize(dst_g, None, fx=scale, fy=scale)
    p0 = cv2.goodFeaturesToTrack(s, 400, 0.01, 8)
    if p0 is None or len(p0) < 8:
        return np.eye(2, 3, dtype=np.float32)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(s, d, p0, None)
    good = st.ravel() == 1
    if good.sum() < 8:
        return np.eye(2, 3, dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(p0[good], p1[good], method=cv2.RANSAC)
    if M is None:
        return np.eye(2, 3, dtype=np.float32)
    M = M.astype(np.float32)
    M[:, 2] /= scale
    return M


def motion_candidates(prev_g, cur_g, next_g, k_sigma=4.0, max_area=4000):
    """Camera-motion-compensated 3-frame differencing -> [(cx, cy, strength)]."""
    h, w = cur_g.shape
    a = cv2.warpAffine(prev_g, _align(prev_g, cur_g), (w, h), borderMode=cv2.BORDER_REPLICATE)
    b = cv2.warpAffine(next_g, _align(next_g, cur_g), (w, h), borderMode=cv2.BORDER_REPLICATE)
    c = cur_g.astype(np.int16)
    d = np.minimum(np.abs(c - a.astype(np.int16)), np.abs(c - b.astype(np.int16))).astype(np.uint8)
    d = cv2.GaussianBlur(d, (3, 3), 0)
    thr = max(float(d.mean() + k_sigma * d.std()), 12.0)
    m = cv2.morphologyEx((d > thr).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if 2 <= area <= max_area:
            out.append((float(cents[i][0]), float(cents[i][1]), float(d[y:y + bh, x:x + bw].max()) * area ** 0.5))
    return out


def pick_windows(points, W, H, max_windows, tile=TILE):
    """Greedy: strongest points first; skip points already inside a chosen window (margin 32 px)."""
    wins = []
    for cx, cy, _ in sorted(points, key=lambda p: -p[2]):
        if len(wins) >= max_windows:
            break
        if any(x0 + 32 <= cx <= x0 + tw - 32 and y0 + 32 <= cy <= y0 + th - 32 for x0, y0, tw, th in wins):
            continue
        wins.append(window_around(cx, cy, W, H, tile))
    return wins


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", type=Path, required=True)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--mode", choices=["whole", "full", "motion"], default="motion")
    ap.add_argument("--seqs", nargs="*", default=None, help="default: all seq_* in tracking_eval")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-windows", type=int, default=12, help="motion mode: motion windows per frame (tested: 6 is too few for 7 px drones)")
    ap.add_argument("--sweep-every", type=int, default=15, help="motion mode: full sweep period (frames)")
    ap.add_argument("--keep-thr", type=float, default=0.05, help="drop detections below this score")
    ap.add_argument("--carry-thr", type=float, default=0.3, help="re-look around detections above this")
    args = ap.parse_args()

    kit = load_kit(args.kit / "starter-kit")
    det = Detector(args.onnx, kit)
    root = args.kit / "tracking_eval"
    seqs = args.seqs or sorted(p.name for p in root.glob("seq_*") if (p / "video.mp4").exists())
    result, stats = {}, {}
    for seq in seqs:
        frames = read_video(root / seq / "video.mp4")
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        H, W = grays[0].shape
        calls0, t0 = det.calls, time.perf_counter()
        per_frame, prev_conf = {}, []
        for t, f in enumerate(frames):
            pil = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            if args.mode == "whole":
                dets = classwise_nms(det.run(pil, top_k=30))
            elif args.mode == "full" or t % args.sweep_every == 0:
                dets = detect_tiled(det, pil)
            else:
                pts = []
                if 0 < t < len(frames) - 1:
                    pts = motion_candidates(grays[t - 1], grays[t], grays[t + 1])
                wins = pick_windows(pts, W, H, args.max_windows)
                for c, s, b in prev_conf:                      # keep looking where something was
                    wins.append(window_around(b[0] + b[2] / 2, b[1] + b[3] / 2, W, H))
                dets = []
                for x0, y0, tw, th in dict.fromkeys(wins):
                    dets += det.run(pil.crop((x0, y0, x0 + tw, y0 + th)), (x0, y0), top_k=30)
                dets = classwise_nms(dets)
            dets = [d for d in dets if d[1] >= args.keep_thr]
            prev_conf = [d for d in dets if d[1] >= args.carry_thr][:3]
            per_frame[str(t + 1)] = [[int(c), round(float(s), 5)] + [round(float(v), 2) for v in b] for c, s, b in dets]
        result[seq] = per_frame
        stats[seq] = {"frames": len(frames), "size": [W, H],
                      "model_calls_per_frame": round((det.calls - calls0) / max(1, len(frames)), 2),
                      "seconds": round(time.perf_counter() - t0, 1)}
        print(seq, stats[seq])
    result["_stats"] = {"mode": args.mode, **stats}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
