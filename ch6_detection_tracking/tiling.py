"""Shared helpers for Siesta Ch 6 tiled inference.

Reuses the kit's own preprocessing and RF-DETR decoding (imported from the kit's
starter-kit/predict_onnx_baseline.py) so boxes/scores stay identical to the kit.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image

TILE = 384


def load_kit(kit_starter_dir: Path):
    """Import the kit's predict_onnx_baseline module from its starter-kit folder."""
    path = Path(kit_starter_dir) / "predict_onnx_baseline.py"
    spec = importlib.util.spec_from_file_location("kit_predict", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kit_predict"] = mod
    spec.loader.exec_module(mod)
    return mod


def tile_grid(W: int, H: int, tile: int = TILE, overlap: int = 64) -> list[tuple[int, int, int, int]]:
    """Overlapping native-resolution windows covering the whole image."""
    tw, th = min(tile, W), min(tile, H)
    stride_x, stride_y = max(1, tw - overlap), max(1, th - overlap)
    xs = list(range(0, max(1, W - tw + 1), stride_x))
    ys = list(range(0, max(1, H - th + 1), stride_y))
    if xs[-1] != W - tw:
        xs.append(W - tw)
    if ys[-1] != H - th:
        ys.append(H - th)
    return [(x, y, tw, th) for y in ys for x in xs]


def window_around(cx: float, cy: float, W: int, H: int, tile: int = TILE) -> tuple[int, int, int, int]:
    """Native tile-sized window centred on (cx, cy), clipped inside the image."""
    tw, th = min(tile, W), min(tile, H)
    x0 = int(round(min(max(cx - tw / 2, 0), W - tw)))
    y0 = int(round(min(max(cy - th / 2, 0), H - th)))
    return x0, y0, tw, th


class Detector:
    """Runs the exported full RF-DETR ONNX on crops/frames (CPU ONNX Runtime by default)."""

    def __init__(self, onnx_path: Path, kit, num_classes: int = 3, img_size: int = 384,
                 providers: list[str] | None = None):
        import onnxruntime as ort
        self.kit = kit
        self.num_classes = num_classes
        self.img_size = img_size
        self.sess = ort.InferenceSession(str(onnx_path), providers=providers or ["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_names = [o.name for o in self.sess.get_outputs()]
        self.calls = 0

    def run(self, pil: Image.Image, offset=(0, 0), top_k: int = 100, score_thr: float = 0.0):
        """Detect on one PIL image; returns [(cls, score, [x,y,w,h])] in parent-image coordinates."""
        inp = self.kit._preprocess_imagenet(pil, self.img_size)
        raw = self.sess.run(self.out_names, {self.in_name: inp})
        outs = dict(zip(self.out_names, raw))           # index by NAME (dets/labels share a shape)
        self.calls += 1
        dets = self.kit._decode_rfdetr_topk(outs["dets"][0], outs["labels"][0], pil.width, pil.height,
                                            self.num_classes, top_k=top_k, score_thr=score_thr)
        ox, oy = offset
        return [(c, s, [b[0] + ox, b[1] + oy, b[2], b[3]]) for c, s, b in dets]


def iou_xywh(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(1e-9, a[2] * a[3] + b[2] * b[3] - inter)


def classwise_nms(dets, iou_thr: float = 0.5, top_k: int = 100):
    """Greedy per-class NMS, then keep the global top_k by score."""
    kept = []
    for cls in sorted({d[0] for d in dets}):
        cand = sorted((d for d in dets if d[0] == cls), key=lambda d: -d[1])
        keep_c = []
        for d in cand:
            if all(iou_xywh(d[2], k[2]) < iou_thr for k in keep_c):
                keep_c.append(d)
        kept.extend(keep_c)
    kept.sort(key=lambda d: -d[1])
    return kept[:top_k]


def detect_tiled(det: Detector, pil: Image.Image, *, full_pass: bool = True, overlap: int = 64,
                 per_tile_k: int = 30, score_thr: float = 0.0, top_k: int = 100):
    """Full tiled sweep: every native tile + (optionally) one whole-frame pass for large aircraft."""
    W, H = pil.size
    out = []
    for x0, y0, tw, th in tile_grid(W, H, overlap=overlap):
        out += det.run(pil.crop((x0, y0, x0 + tw, y0 + th)), (x0, y0), top_k=per_tile_k, score_thr=score_thr)
    if full_pass and (W > TILE or H > TILE):
        out += det.run(pil, (0, 0), top_k=per_tile_k, score_thr=score_thr)
    return classwise_nms(out, top_k=top_k)
