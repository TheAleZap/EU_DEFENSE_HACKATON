"""Siesta: pre-upload checker for all TEST submissions (Ch 6 COCO + MOT, Ch 8 JSONL, Ch 9 JSONL).

Checks the rules from the kits (ids, types, ranges, formats, duplicates) without needing
test labels. Exit code 0 only if everything passes. Run it before EVERY upload.

Usage (give only the parts you have):
  python check_submissions_before_upload.py \
     --c6-kit <challenge-1-cuas-tracking> --c6-preds preds_test.json --c6-mot pred_mot/ \
     --c8-kit <challenge-4-edge-triage>   --c8-preds triage_test.jsonl \
     --c9-kit <challenge-5-acoustic-visual-cuas> --c9-preds c9_test.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

SOUND = {"drone_buzz", "vehicle_mechanical", "explosion", "ambient_outdoor"}
problems: list[str] = []


def bad(msg):
    problems.append(msg)


def finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def check_c6(kit: Path, preds: Path | None, mot: Path | None):
    if preds:
        gt = json.loads((kit / "dataset" / "test" / "_annotations.coco.json").read_text())
        ids = {im["id"] for im in gt["images"]}
        cats = {c["id"] for c in gt["categories"]}
        P = json.loads(preds.read_text())
        if not isinstance(P, list):
            bad("c6 preds: top level must be a JSON list")
            P = []
        per_img = Counter()
        for i, p in enumerate(P):
            if p.get("image_id") not in ids:
                bad(f"c6 preds[{i}]: unknown image_id {p.get('image_id')}")
            if p.get("category_id") not in cats:
                bad(f"c6 preds[{i}]: category_id {p.get('category_id')} not in {sorted(cats)} (0-based)")
            b = p.get("bbox")
            if not (isinstance(b, list) and len(b) == 4 and all(finite(v) for v in b) and b[2] > 0 and b[3] > 0):
                bad(f"c6 preds[{i}]: bad bbox {b}")
            s = p.get("score")
            if not (finite(s) and 0 <= s <= 1):
                bad(f"c6 preds[{i}]: score {s} not in [0,1]")
            per_img[p.get("image_id")] += 1
            if len(problems) > 20:
                break
        over = [k for k, v in per_img.items() if v > 100]
        if over:
            bad(f"c6 preds: {len(over)} images have >100 detections (only top 100 are scored)")
        print(f"c6 COCO: {len(P)} detections over {len(per_img)}/{len(ids)} test images")
    if mot:
        man = json.loads((kit / "tracking_eval" / "sequences_manifest.json").read_text())
        for s in man["sequences"]:
            if s["split"] != "test":
                continue
            f = mot / f"{s['sequence']}.txt"
            if not f.exists():
                bad(f"c6 MOT: missing {f.name}")
                continue
            seen, n = set(), 0
            for ln, line in enumerate(f.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                r = line.split(",")
                n += 1
                try:
                    fr, tid = int(r[0]), int(r[1])
                    x, y, w, h, conf = map(float, r[2:7])
                    cls = int(r[7])
                except Exception:
                    bad(f"c6 MOT {f.name}:{ln}: unparsable row '{line}' (need frame,id,x,y,w,h,conf,class,1.0)")
                    continue
                if len(r) != 9:
                    bad(f"c6 MOT {f.name}:{ln}: expected 9 columns, got {len(r)}")
                if not (1 <= fr <= s["frame_count"]):
                    bad(f"c6 MOT {f.name}:{ln}: frame {fr} outside 1..{s['frame_count']} (1-based)")
                if not (w > 0 and h > 0 and all(map(math.isfinite, (x, y, w, h, conf)))):
                    bad(f"c6 MOT {f.name}:{ln}: non-positive or non-finite box")
                if cls not in (1, 2, 3):
                    bad(f"c6 MOT {f.name}:{ln}: class_id {cls} must be 1..3 (1-based)")
                if (fr, tid) in seen:
                    bad(f"c6 MOT {f.name}:{ln}: duplicate (frame, track_id) = ({fr},{tid})")
                seen.add((fr, tid))
                if len(problems) > 40:
                    break
            print(f"c6 MOT {f.name}: {n} rows, {len({t for _, t in seen})} track ids")


def check_jsonl(kit_labels: Path, preds: Path, name: str, validate):
    gt_ids = [json.loads(l)["id"] for l in kit_labels.read_text().splitlines() if l.strip()]
    rows = [json.loads(l) for l in preds.read_text().splitlines() if l.strip()]
    ids = [r.get("id") for r in rows]
    dup = [k for k, v in Counter(ids).items() if v > 1]
    if dup:
        bad(f"{name}: duplicate ids {dup[:3]}")
    if set(ids) != set(gt_ids):
        bad(f"{name}: ids differ from test labels (missing {len(set(gt_ids) - set(ids))}, extra {len(set(ids) - set(gt_ids))})")
    for r in rows:
        validate(r)
    print(f"{name}: {len(rows)} rows for {len(gt_ids)} test ids")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--c6-kit", type=Path); ap.add_argument("--c6-preds", type=Path); ap.add_argument("--c6-mot", type=Path)
    ap.add_argument("--c8-kit", type=Path); ap.add_argument("--c8-preds", type=Path)
    ap.add_argument("--c9-kit", type=Path); ap.add_argument("--c9-preds", type=Path)
    a = ap.parse_args()
    if a.c6_kit:
        check_c6(a.c6_kit, a.c6_preds, a.c6_mot)
    if a.c8_kit and a.c8_preds:
        def v8(r):
            if not isinstance(r.get("triage_keep"), bool):
                bad(f"c8 {r.get('id')}: triage_keep must be bool")
            if not finite(r.get("priority_score")):
                bad(f"c8 {r.get('id')}: priority_score must be a finite number")
        rows = check_jsonl(a.c8_kit / "dataset" / "test" / "labels.jsonl", a.c8_preds, "c8", v8)
        ties = len(rows) - len({round(float(r.get('priority_score', 0)), 6) for r in rows})
        if ties:
            print(f"c8 note: {ties} tied priority scores (ties are broken pessimistically by the scorer)")
    if a.c9_kit and a.c9_preds:
        def v9(r):
            if not isinstance(r.get("wake_triggered"), bool):
                bad(f"c9 {r.get('id')}: wake_triggered must be bool")
            if r.get("sound_class") not in SOUND:
                bad(f"c9 {r.get('id')}: sound_class {r.get('sound_class')} invalid")
            if isinstance(r.get("wake_triggered"), bool) and r["wake_triggered"] != (r.get("sound_class") == "drone_buzz"):
                print(f"c9 warning {r.get('id')}: wake and class disagree (labels always agree)")
        check_jsonl(a.c9_kit / "dataset" / "test" / "labels.jsonl", a.c9_preds, "c9", v9)
    if problems:
        print("\nFAIL:")
        for p in problems[:50]:
            print(" -", p)
        raise SystemExit(1)
    print("\nPASS: all checked submissions are well-formed.")


if __name__ == "__main__":
    main()
