"""Axelera `axcompile --transform` callback for the Ch 6 RF-DETR model (384x384).

Per the kit: RF-DETR expects ImageNet-normalised NCHW RGB (mean 0.485/0.456/0.406,
std 0.229/0.224/0.225), resized to 384x384, matching predict_onnx_baseline.py.
Calibrate ONLY with the kit's challenge-1-cuas-tracking/calibration/ folder (train-only).

Compile the BACKBONE export first (the full model's GridSample deformable-attention tail
is the likely blocker; the decoder then runs on the host):
  axcompile -i exports/rfdetr/backbone/inference_model.onnx --imageset <challenge-1>/calibration \
            --transform preprocess_rfdetr.py --input-shape 1,3,384,384 -o out_c6_backbone/
If the compiler reports a rank mismatch: --transform preprocess_rfdetr.py:get_preprocess_transform_batched
"""
import numpy as np
import torch
from PIL import Image

SIZE = 384
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def get_preprocess_transform(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image = image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(np.ascontiguousarray((arr - MEAN) / STD))      # CHW, normalised


def get_preprocess_transform_batched(image):
    return get_preprocess_transform(image).unsqueeze(0)
