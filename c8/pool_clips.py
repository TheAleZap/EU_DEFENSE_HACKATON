"""Siesta Ch 8: rank clips, not frames (idea E).

Verified: teacher labels are identical for every frame of a clip, and test rows carry
`video_id`. So we replace each frame's priority with its clip's mean (or max) priority,
plus a tiny frame-level tie-breaker, and set triage_keep by clip majority.

Input : kit triage JSONL ({"id","triage_keep","priority_score"}) from predict_onnx_baseline.py triage
Output: same format, pooled -> score with the kit:
  python <kit>/starter-kit/evaluate_submissions.py triage --gt <kit>/dataset/valid/labels.jsonl --preds pooled.jsonl

Usage:
  python pool_clips.py --kit <challenge-4-edge-triage> --split valid --preds triage_preds.jsonl --out pooled.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", type=Path, required=True)
    ap.add_argument("--split", default="valid", choices=["valid", "test"])
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", default="mean", choices=["mean", "max"])
    ap.add_argument("--eps", type=float, default=1e-3, help="frame-level tie-breaker weight")
    args = ap.parse_args()

    rows = {json.loads(l)["id"]: json.loads(l) for l in
            (args.kit / "dataset" / args.split / "labels.jsonl").read_text().splitlines() if l.strip()}
    preds = [json.loads(l) for l in args.preds.read_text().splitlines() if l.strip()]
    missing = set(rows) - {p["id"] for p in preds}
    if missing:
        raise SystemExit(f"predictions missing {len(missing)} ids, e.g. {sorted(missing)[:3]}")
    by_clip = defaultdict(list)
    for p in preds:
        by_clip[rows[p["id"]]["video_id"]].append(p)
    out = []
    for clip, ps in by_clip.items():
        vals = [float(p["priority_score"]) for p in ps]
        agg = max(vals) if args.mode == "max" else sum(vals) / len(vals)
        keep = sum(bool(p["triage_keep"]) for p in ps) * 2 >= len(ps)
        for p in ps:
            s = min(1.0, max(0.0, agg + args.eps * float(p["priority_score"])))
            out.append({"id": p["id"], "triage_keep": bool(keep), "priority_score": round(s, 6)})
    order = {p["id"]: i for i, p in enumerate(preds)}
    out.sort(key=lambda r: order[r["id"]])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r) for r in out) + "\n", encoding="utf-8")
    print(json.dumps({"frames": len(out), "clips": len(by_clip), "mode": args.mode}))


if __name__ == "__main__":
    main()
