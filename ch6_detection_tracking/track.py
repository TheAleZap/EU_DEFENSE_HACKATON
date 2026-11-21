"""Siesta Ch 6 tracker: detections JSON (from detect_video.py) -> MOT files for the kit scorer.

Design (ideas B + C from the team brief):
  * Two-stage association (ByteTrack-style): high-score detections first, then low-score
    detections may only extend existing tracks.
  * Gating by centre distance scaled to box size (IoU is useless for 7 px boxes) plus a
    constant-velocity prediction, optionally compensated for camera motion (--kit).
  * Scorer-aware output policy: the kit's MOT scorer charges FP + FN for a misplaced box
    but only FN for silence, so we only write boxes of CONFIRMED tracks (>= --min-hits
    detections, mean score >= --confirm-score), drop short tracks (< --min-len), and fill
    only short gaps (<= --max-gap) by linear interpolation between real detections.
  * Offline mode (default) back-fills a confirmed track's early frames. Use --online to
    emit only frames after confirmation (real-time behaviour, for honest demo numbers).

Output rows: frame(1-based),track_id,x,y,w,h,conf,class_id(1-based),1.0  -> pred_mot/seq_XXXX.txt
Score:  python <kit>/starter-kit/evaluate_submissions.py mot --gt-dir <kit>/tracking_eval --pred-dir pred_mot --split valid
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # find sibling modules

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


class Track:
    _next = 1

    def __init__(self, frame, det):
        self.id = Track._next
        Track._next += 1
        self.obs = {frame: det}                  # frame -> (cls, score, [x,y,w,h])
        cx, cy = det[2][0] + det[2][2] / 2, det[2][1] + det[2][3] / 2
        self.pos = np.array([cx, cy], float)
        self.vel = np.zeros(2)
        self.size = np.array(det[2][2:4], float)
        self.last = frame
        self.misses = 0

    def predict(self, frame, cam=None):
        dt = frame - self.last
        p = self.pos + self.vel * dt
        if cam is not None:                       # camera moved between last and now
            p = cam[:, :2] @ p + cam[:, 2]
        return p

    def update(self, frame, det, cam=None):
        cx, cy = det[2][0] + det[2][2] / 2, det[2][1] + det[2][3] / 2
        new = np.array([cx, cy], float)
        dt = max(1, frame - self.last)
        base = self.pos if cam is None else cam[:, :2] @ self.pos + cam[:, 2]
        v = (new - base) / dt
        self.vel = 0.6 * self.vel + 0.4 * v if len(self.obs) > 1 else v
        self.pos = new
        self.size = 0.7 * self.size + 0.3 * np.array(det[2][2:4], float)
        self.obs[frame] = det
        self.last = frame
        self.misses = 0

    def confirmed(self, min_hits, confirm_score):
        if len(self.obs) < min_hits:
            return False
        return float(np.mean([d[1] for d in self.obs.values()])) >= confirm_score


def associate(tracks, dets, frame, gate, cams):
    """Greedy nearest-neighbour on gated, size-normalised centre distance."""
    pairs = []
    for ti, tr in enumerate(tracks):
        cam = cams.get(frame) if cams else None
        p = tr.predict(frame, cam)
        scale = max(8.0, float(np.max(tr.size)))
        for di, d in enumerate(dets):
            c = np.array([d[2][0] + d[2][2] / 2, d[2][1] + d[2][3] / 2])
            dist = float(np.linalg.norm(c - p)) / scale
            if dist <= gate * (1 + 0.5 * min(tr.misses, 4)):
                pairs.append((dist, ti, di))
    pairs.sort()
    used_t, used_d, out = set(), set(), []
    for dist, ti, di in pairs:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        out.append((ti, di))
    return out, [i for i in range(len(dets)) if i not in used_d]


def camera_motion(kit_dir: Path, seq: str):
    """Per-frame affine mapping previous-frame coords -> current-frame coords (frame keys 1-based)."""
    import cv2
    from detect_video import _align, read_video
    frames = read_video(kit_dir / "tracking_eval" / seq / "video.mp4")
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    return {t + 1: _align(grays[t - 1], grays[t]) for t in range(1, len(grays))}


def merge_classes(dets, mode, iou_thr=0.5):
    """Class-agnostic NMS. RF-DETR often returns one box several times with different classes;
    the MOT scorer ignores class, so each copy would be a false positive.
    mode 'max' keeps the best copy's score, 'sum' adds the copies' scores (capped at 1)."""
    if mode == "none":
        return dets
    from tiling import iou_xywh
    kept = []
    for d in sorted(dets, key=lambda d: -d[1]):
        for i, k in enumerate(kept):
            if iou_xywh(d[2], k[2]) >= iou_thr:
                if mode == "sum":
                    kept[i] = (k[0], min(1.0, k[1] + d[1]), k[2])
                break
        else:
            kept.append(d)
    return kept


def run_sequence(per_frame, args, cams=None):
    Track._next = 1
    active, finished = [], []
    frames = sorted(int(f) for f in per_frame)
    merge = getattr(args, "merge", "none")
    for f in frames:
        dets = merge_classes([(d[0], d[1], d[2:6]) for d in per_frame[str(f)]], merge)
        dets = [d for d in dets if d[1] >= args.low_thr]
        high = [d for d in dets if d[1] >= args.high_thr]
        low = [d for d in dets if d[1] < args.high_thr]
        matches, unmatched_high = associate(active, high, f, args.gate, cams)
        matched_t = set()
        for ti, di in matches:
            active[ti].update(f, high[di], cams.get(f) if cams else None)
            matched_t.add(ti)
        rest = [tr for i, tr in enumerate(active) if i not in matched_t and len(tr.obs) >= 2]
        m2, _ = associate(rest, low, f, args.gate, cams)
        for ti, di in m2:
            rest[ti].update(f, low[di], cams.get(f) if cams else None)
            matched_t.add(active.index(rest[ti]))
        for i, tr in enumerate(active):
            if i not in matched_t:
                tr.misses += 1
                if cams and f in cams:               # keep position in current-frame coordinates
                    tr.pos = cams[f][:, :2] @ tr.pos + cams[f][:, 2]
        for di in unmatched_high:
            active.append(Track(f, high[di]))
        still = []
        for tr in active:
            (finished if tr.misses > args.max_miss else still).append(tr)
        active = still
    return finished + active


def emit(tracks, args):
    rows = []
    for tr in tracks:
        if not tr.confirmed(args.min_hits, args.confirm_score) or len(tr.obs) < args.min_len:
            continue
        fs = sorted(tr.obs)
        if args.online:                                 # only after confirmation (real-time view)
            fs = fs[args.min_hits - 1:]
        cls = Counter(d[0] for d in tr.obs.values()).most_common(1)[0][0]
        out = {}
        for f in fs:
            c, s, b = tr.obs[f]
            if s >= args.emit_thr:
                out[f] = (b, s)
        keys = sorted(out)
        for a, b2 in zip(keys, keys[1:]):               # fill short gaps only
            gap = b2 - a
            if 1 < gap <= args.max_gap + 1:
                ba, bb = np.array(out[a][0]), np.array(out[b2][0])
                for k in range(1, gap):
                    w = k / gap
                    out[a + k] = (list((1 - w) * ba + w * bb), min(out[a][1], out[b2][1]))
        for f in sorted(out):
            b, s = out[f]
            rows.append((f, tr.id, b, s, cls))
    rows.sort()
    return [f"{f},{tid},{b[0]:.2f},{b[1]:.2f},{max(b[2], 1):.2f},{max(b[3], 1):.2f},{s:.4f},{cls + 1},1.0"
            for f, tid, b, s, cls in rows]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", type=Path, required=True, help="JSON from detect_video.py")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--kit", type=Path, default=None, help="challenge-1 folder: enables camera-motion compensation")
    ap.add_argument("--high-thr", type=float, default=0.5)
    ap.add_argument("--low-thr", type=float, default=0.1)
    ap.add_argument("--emit-thr", type=float, default=0.1)
    ap.add_argument("--confirm-score", type=float, default=0.45)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--min-len", type=int, default=5)
    ap.add_argument("--max-miss", type=int, default=10)
    ap.add_argument("--max-gap", type=int, default=8)
    ap.add_argument("--gate", type=float, default=3.0, help="max centre distance in box-size units")
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--merge", choices=["none", "max", "sum"], default="none",
                    help="class-agnostic merge of same-box detections before tracking")
    args = ap.parse_args()

    data = json.loads(args.dets.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for seq, per_frame in data.items():
        if seq.startswith("_"):
            continue
        cams = camera_motion(args.kit, seq) if args.kit else None
        lines = emit(run_sequence(per_frame, args, cams), args)
        (args.out_dir / f"{seq}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(seq, "rows", len(lines), "tracks", len({l.split(',')[1] for l in lines}))


if __name__ == "__main__":
    main()
