"""Build a native-resolution tiled COCO dataset for RF-DETR training (Siesta idea A).

Why: the kit resizes whole frames to 384x384, which shrinks a 7 px drone in a
1920 px frame to ~1.4 px. Training on native-resolution 384 crops keeps tiny
targets visible. Output keeps the RF-DETR layout (train/ valid/ test/ each with
_annotations.coco.json), so the kit's train_rfdetr.py works unchanged.

Per split (train, valid):
  * every original image (so large aircraft are still learned at full-frame scale)
  * `--pos-per-box` random native crops containing each box (box fully inside)
  * `--neg-per-image` random native crops containing no box
test/ is copied unchanged (labels are held out; never use --run-test).

Usage:
  python build_tiled_dataset.py --src <challenge-1-cuas-tracking>/dataset --out tiled_dataset
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image

MIN_VISIBLE = 0.5  # keep a box in a crop only if >= 50% of its area is inside


def clip_box(b, x0, y0, w, h):
    bx, by, bw, bh = b
    x1, y1 = max(bx, x0), max(by, y0)
    x2, y2 = min(bx + bw, x0 + w), min(by + bh, y0 + h)
    if x2 <= x1 or y2 <= y1:
        return None, 0.0
    vis = (x2 - x1) * (y2 - y1) / max(1e-9, bw * bh)
    return [x1 - x0, y1 - y0, x2 - x1, y2 - y1], vis


def crop_window_for_box(b, W, H, tile, rng):
    """Random tile-sized window (clipped to image) that fully contains box b if possible."""
    tw, th = min(tile, W), min(tile, H)
    bx, by, bw, bh = b
    lo_x = max(0, int(bx + bw) - tw)
    hi_x = min(W - tw, int(bx))
    lo_y = max(0, int(by + bh) - th)
    hi_y = min(H - th, int(by))
    x0 = rng.randint(lo_x, hi_x) if hi_x >= lo_x else max(0, min(W - tw, int(bx + bw / 2 - tw / 2)))
    y0 = rng.randint(lo_y, hi_y) if hi_y >= lo_y else max(0, min(H - th, int(by + bh / 2 - th / 2)))
    return x0, y0, tw, th


def overlaps_any(x0, y0, w, h, boxes):
    for bx, by, bw, bh in boxes:
        if bx < x0 + w and bx + bw > x0 and by < y0 + h and by + bh > y0:
            return True
    return False


def build_split(src: Path, out: Path, split: str, tile: int, pos_per_box: int,
                neg_per_image: int, rng: random.Random) -> dict:
    doc = json.loads((src / split / "_annotations.coco.json").read_text(encoding="utf-8"))
    by_img: dict[int, list] = {}
    for a in doc["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    (out / split).mkdir(parents=True, exist_ok=True)
    images, anns = [], []
    next_img, next_ann = 0, 0

    def add(img_pil, fname, boxes_cats):
        nonlocal next_img, next_ann
        img_pil.save(out / split / fname, quality=95)
        images.append({"id": next_img, "file_name": fname,
                       "width": img_pil.width, "height": img_pil.height})
        for box, cat in boxes_cats:
            anns.append({"id": next_ann, "image_id": next_img, "category_id": cat,
                         "bbox": [round(v, 2) for v in box], "area": round(box[2] * box[3], 2),
                         "iscrowd": 0})
            next_ann += 1
        next_img += 1

    n_pos = n_neg = 0
    for im in doc["images"]:
        stem = Path(im["file_name"]).stem
        with Image.open(src / split / im["file_name"]) as pil:
            pil = pil.convert("RGB")
            W, H = pil.size
            iann = by_img.get(im["id"], [])
            boxes = [a["bbox"] for a in iann]
            # 1) original image, unchanged
            add(pil, f"{stem}.jpg", [(a["bbox"], a["category_id"]) for a in iann])
            # 2) positive native crops
            for a in iann:
                for k in range(pos_per_box):
                    x0, y0, tw, th = crop_window_for_box(a["bbox"], W, H, tile, rng)
                    kept = []
                    for b in iann:
                        cb, vis = clip_box(b["bbox"], x0, y0, tw, th)
                        if cb is not None and vis >= MIN_VISIBLE and cb[2] >= 1 and cb[3] >= 1:
                            kept.append((cb, b["category_id"]))
                    if not kept:
                        continue
                    add(pil.crop((x0, y0, x0 + tw, y0 + th)), f"{stem}_p{a['id']}_{k}.jpg", kept)
                    n_pos += 1
            # 3) negative native crops (no box overlap at all)
            for k in range(neg_per_image):
                tw, th = min(tile, W), min(tile, H)
                for _ in range(20):
                    x0, y0 = rng.randint(0, W - tw), rng.randint(0, H - th)
                    if not overlaps_any(x0, y0, tw, th, boxes):
                        add(pil.crop((x0, y0, x0 + tw, y0 + th)), f"{stem}_n{k}.jpg", [])
                        n_neg += 1
                        break
    out_doc = {k: v for k, v in doc.items() if k not in ("images", "annotations")}
    out_doc["images"], out_doc["annotations"] = images, anns
    (out / split / "_annotations.coco.json").write_text(json.dumps(out_doc), encoding="utf-8")
    return {"split": split, "originals": len(doc["images"]), "pos_crops": n_pos,
            "neg_crops": n_neg, "total_images": len(images), "boxes": len(anns)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="kit dataset/ dir (train/ valid/ test/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tile", type=int, default=384)
    ap.add_argument("--pos-per-box", type=int, default=2)
    ap.add_argument("--neg-per-image", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid"):
        print(build_split(args.src, args.out, split, args.tile, args.pos_per_box,
                          args.neg_per_image, rng))
    # test: copy unchanged (labels held out). Needed only because RF-DETR expects the folder.
    if (args.out / "test").exists():
        shutil.rmtree(args.out / "test")
    shutil.copytree(args.src / "test", args.out / "test")
    print({"split": "test", "copied": True})


if __name__ == "__main__":
    main()
