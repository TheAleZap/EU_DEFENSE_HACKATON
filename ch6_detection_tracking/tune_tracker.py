"""Siesta Ch 6: automatic tracker tuning with leave-one-sequence-out (LOO), scored by the KIT's MOT scorer.

Put this file in ch6_detection_tracking/ (next to track.py). It needs:
  * a detections JSON from detect_video.py (use --mode full for scored work)
  * the kit folder (challenge-1-cuas-tracking)

What it does
  1. Tries a grid of tracker settings on the 4 valid sequences (seq_0001-0004).
  2. LOO: for each sequence, picks the best settings on the OTHER 3 and scores them on the held-out one.
     The mean held-out score is an honest estimate (no tuning on the sequence being scored).
  3. Prints the settings that are best on all 4 valid sequences, with the exact track.py command to use.
Objective = MOTA + IDF1 (both from the kit scorer).

Usage:
  python ch6_detection_tracking/tune_tracker.py --dets dets_full.json --kit <challenge-1-cuas-tracking> [--quick]
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import track  # noqa: E402

VALID = ["seq_0001", "seq_0002", "seq_0003", "seq_0004"]


def load_scorer(kit: Path):
    p = kit / "starter-kit" / "evaluate_submissions.py"
    spec = importlib.util.spec_from_file_location("kit_eval", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["kit_eval"] = m
    spec.loader.exec_module(m)
    return m


def make_single_seq_gt(kit: Path, tmp: Path):
    """One GT folder per sequence (manifest filtered) so the kit scorer can score sequences separately."""
    man = json.loads((kit / "tracking_eval" / "sequences_manifest.json").read_text())
    out = {}
    for seq in VALID:
        d = tmp / f"gt_{seq}"
        (d / seq / "gt").mkdir(parents=True)
        shutil.copy(kit / "tracking_eval" / seq / "gt" / "gt.txt", d / seq / "gt" / "gt.txt")
        (d / "sequences_manifest.json").write_text(json.dumps(
            {"sequences": [s for s in man["sequences"] if s["sequence"] == seq]}))
        out[seq] = d
    return out


def score_seq(scorer, gt_dir: Path, pred_dir: Path) -> dict:
    return scorer.evaluate_mot_tracking(gt_dir, pred_dir, split="valid")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dets", type=Path, required=True)
    ap.add_argument("--kit", type=Path, required=True)
    ap.add_argument("--no-cam", action="store_true", help="skip camera-motion compensation (faster)")
    ap.add_argument("--quick", action="store_true", help="smaller grid")
    args = ap.parse_args()

    scorer = load_scorer(args.kit)
    data = json.loads(args.dets.read_text())
    cams = {} if args.no_cam else {s: track.camera_motion(args.kit, s) for s in VALID}
    grid = {
        "high_thr": [0.3, 0.4, 0.5, 0.6],
        "confirm_score": [0.3, 0.4, 0.5],
        "min_hits": [2, 3, 5],
        "min_len": [3, 8],
        "max_gap": [4, 8, 15],
        "gate": [2.0, 3.0, 5.0],
    }
    if args.quick:
        grid = {"high_thr": [0.4, 0.6], "confirm_score": [0.3, 0.5], "min_hits": [2, 4], "min_len": [5],
                "max_gap": [8], "gate": [3.0]}
    fixed = {"low_thr": 0.1, "emit_thr": 0.1, "max_miss": 10, "online": False}
    keys = list(grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
    print(f"{len(combos)} settings x {len(VALID)} sequences")

    results = []          # (params, {seq: metrics})
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        gts = make_single_seq_gt(args.kit, tmp)
        for i, p in enumerate(combos):
            a = SimpleNamespace(**p, **fixed)
            per = {}
            for seq in VALID:
                lines = track.emit(track.run_sequence(data[seq], a, cams.get(seq) if cams else None), a)
                pd = tmp / "pred"
                if pd.exists():
                    shutil.rmtree(pd)
                pd.mkdir()
                (pd / f"{seq}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                per[seq] = score_seq(scorer, gts[seq], pd)
            results.append((p, per))
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(combos)}")

    def obj(per, seqs):
        return sum(per[s]["MOTA"] + per[s]["IDF1"] for s in seqs) / len(seqs)

    print("\nLEAVE-ONE-SEQUENCE-OUT (honest estimate):")
    held = []
    for s in VALID:
        others = [o for o in VALID if o != s]
        best_p, best_per = max(results, key=lambda r: obj(r[1], others))
        m = best_per[s]
        held.append(m)
        print(f"  held-out {s}: MOTA {m['MOTA']:+.3f} IDF1 {m['IDF1']:.3f} continuity {m['track_continuity']:.3f}  (tuned on {others})")
    print(f"  mean held-out MOTA {sum(m['MOTA'] for m in held) / 4:+.3f}, IDF1 {sum(m['IDF1'] for m in held) / 4:.3f}")

    best_p, best_per = max(results, key=lambda r: obj(r[1], VALID))
    print("\nBEST ON ALL 4 VALID SEQUENCES (use for test seq_0005-0008):", json.dumps(best_p))
    for s in VALID:
        m = best_per[s]
        print(f"  {s}: MOTA {m['MOTA']:+.3f} IDF1 {m['IDF1']:.3f} continuity {m['track_continuity']:.3f}")
    flags = " ".join(f"--{k.replace('_', '-')} {v}" for k, v in best_p.items())
    cam = "" if args.no_cam else f" --kit {args.kit}"
    print(f"\nCommand:\n  python ch6_detection_tracking/track.py --dets {args.dets} --out-dir pred_mot{cam} {flags}")
    print("Then score all 4 together with the kit scorer and log it in EXPERIMENTS_LOG.md.")


if __name__ == "__main__":
    main()
