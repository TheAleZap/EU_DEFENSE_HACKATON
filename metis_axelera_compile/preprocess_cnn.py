"""Axelera `axcompile --transform` callback for the Ch 8 and Ch 9 CNNs (224x224).

Per the Axelera compiler docs: `--transform preprocess_cnn.py` loads `get_preprocess_transform`,
which receives one image (PIL.Image or numpy array) and returns a torch.Tensor.
Per the kits: these CNN graphs ALREADY contain ImageNet normalisation, so we only
resize to 224x224 RGB, convert to float32 CHW and scale to [0, 1].

Shape: returns CHW [3,224,224]. If the compiler reports a rank/shape mismatch, use the
batched variant instead:  --transform preprocess_cnn.py:get_preprocess_transform_batched

Example (Ch 9; Ch 8 is the same with its own onnx and ../calibration folder):
  axcompile -i audio_wake_cnn.onnx --imageset <challenge-5>/calibration_spectrograms \
            --transform preprocess_cnn.py --input-shape 1,3,224,224 -o out_c9/
Check `axcompile --help` for your SDK version.
"""
import numpy as np
import torch
from PIL import Image

SIZE = 224


def get_preprocess_transform(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image = image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0          # [0,1], HWC
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))   # CHW


def get_preprocess_transform_batched(image):
    return get_preprocess_transform(image).unsqueeze(0)          # NCHW
