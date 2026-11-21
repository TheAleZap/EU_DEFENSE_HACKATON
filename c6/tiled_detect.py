"""Siesta Ch 6: tiled COCO detection (idea A + D: native tiles, low threshold for mAP).

Writes the kit's preds.json format: [{"image_id", "category_id" (0-based), "bbox" [x,y,w,h], "score"}],
top 100 per image. Score it with the kit:
  python <kit>/starter-kit/evaluate_submissions.py detect --gt <kit>/dataset/valid/_annotations.coco.json --preds preds.json

Usage:
  python tiled_detect.py --kit <challenge-1-cuas-tracking> --onnx inference_model.onnx --split valid --out preds_valid.json
  python tiled_detect.py --kit <challenge-1-cuas-tracking> --onnx inference_model.onnx --split test  --out preds_test.json
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))  # find sibling modules

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from tiling import Detector, detect_tiled, load_kit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", type=Path, required=True, help="challenge-1-cuas-tracking folder")
    ap.add_argument("--onnx", type=Path, required=True, help="FULL exported RF-DETR ONNX (dets/labels outputs)")
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--no-full-pass", action="store_true", help="skip the whole-frame pass")
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N images")
    args = ap.parse_args()

    kit = load_kit(args.kit / "starter-kit")
    gt = json.loads((args.kit / "dataset" / args.split / "_annotations.coco.json").read_text(encoding="utf-8"))
    det = Detector(args.onnx, kit, num_classes=len(gt["categories"]))
    preds, t0 = [], time.perf_counter()
    images = gt["images"][: args.limit] if args.limit else gt["images"]
    for i, im in enumerate(images):
        with Image.open(args.kit / "dataset" / args.split / im["file_name"]) as pil:
            pil = pil.convert("RGB")
            for cls, score, box in detect_tiled(det, pil, full_pass=not args.no_full_pass, overlap=args.overlap):
                if box[2] > 0 and box[3] > 0:
                    preds.append({"image_id": int(im["id"]), "category_id": int(cls),
                                  "bbox": [float(v) for v in box], "score": float(score)})
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(images)} images, {det.calls} model calls")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(preds), encoding="utf-8")
    dt = time.perf_counter() - t0
    print(json.dumps({"images": len(images), "predictions": len(preds), "model_calls": det.calls,
                      "calls_per_image": round(det.calls / max(1, len(images)), 2),
                      "seconds": round(dt, 1)}))


if __name__ == "__main__":
    main()
