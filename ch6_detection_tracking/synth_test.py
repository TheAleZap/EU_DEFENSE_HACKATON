"""Synthetic sanity test for track.py (NOT a real score: no trained detector involved).

Fakes detector output on the 4 valid sequences from GT: jittered true boxes with misses,
random false positives, and one persistent clutter false positive per sequence (a "tree").
Compares the tracker with the output policy OFF vs ON using the kit's own MOT scorer.

Usage: python synth_test.py --kit <challenge-1-cuas-tracking> --work <tmp dir>
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path


def fake_dets(kit: Path, seed=0, miss=0.25, n_fp=3):
    rng = random.Random(seed)
    man = json.loads((kit / "tracking_eval" / "sequences_manifest.json").read_text())
    out = {}
    for s in man["sequences"]:
        if s["split"] != "valid":
            continue
        W, H = s["width"], s["height"]
        gt = {}
        for line in (kit / "tracking_eval" / s["sequence"] / "gt" / "gt.txt").read_text().splitlines():
            r = line.split(",")
            gt[int(r[0])] = [float(v) for v in r[2:6]]
        size = sum(max(b[2], b[3]) for b in gt.values()) / len(gt)
        tree = [rng.uniform(0, W - size), rng.uniform(0, H - size), size, size]
        frames = {}
        for f in range(1, s["frame_count"] + 1):
            dets = []
            b = gt.get(f)
            if b and rng.random() > miss:
                j = 0.08 * max(b[2], b[3])
                dets.append([0, rng.uniform(0.35, 0.9), b[0] + rng.gauss(0, j), b[1] + rng.gauss(0, j),
                             b[2] * rng.uniform(0.9, 1.1), b[3] * rng.uniform(0.9, 1.1)])
            for _ in range(rng.randint(0, n_fp)):
                dets.append([rng.randint(0, 2), rng.uniform(0.05, 0.55), rng.uniform(0, W - size),
                             rng.uniform(0, H - size), size, size])
            if rng.random() < 0.6:
                dets.append([0, rng.uniform(0.25, 0.45), tree[0] + rng.gauss(0, 1), tree[1] + rng.gauss(0, 1), size, size])
            frames[str(f)] = dets
        out[s["sequence"]] = frames
    return out


def score(kit: Path, pred_dir: Path) -> dict:
    r = subprocess.run([sys.executable, str(kit / "starter-kit" / "evaluate_submissions.py"), "mot",
                        "--gt-dir", str(kit / "tracking_eval"), "--pred-dir", str(pred_dir), "--split", "valid"],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout[r.stdout.index("{"):])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    here = Path(__file__).parent
    for seed in (0, 1, 2):
        dets = args.work / f"synth_{seed}.json"
        dets.write_text(json.dumps(fake_dets(args.kit, seed)))
        res = {}
        for name, extra in {
            "policy_off": ["--min-hits", "1", "--min-len", "1", "--confirm-score", "0", "--emit-thr", "0.25",
                           "--low-thr", "0.25", "--high-thr", "0.25", "--max-gap", "0"],
            "policy_on": [],
        }.items():
            out = args.work / f"mot_{name}_{seed}"
            subprocess.run([sys.executable, str(here / "track.py"), "--dets", str(dets), "--out-dir", str(out), *extra],
                           check=True, capture_output=True)
            s = score(args.kit, out)
            res[name] = {k: s[k] for k in ("MOTA", "IDF1", "track_continuity", "id_switches")}
        print(f"seed {seed}: {json.dumps(res)}")


if __name__ == "__main__":
    main()
