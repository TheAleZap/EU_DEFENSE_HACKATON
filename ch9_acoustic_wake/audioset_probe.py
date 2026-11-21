"""Siesta Ch 9: AudioSet-pretrained audio model (AST) + logistic regression on its 527 AudioSet scores.

Why: with 56 labelled clips, fine-tuning the kit's ImageNet MobileNet on spectrogram PNGs reaches only
wake F1 ~0.49 / class acc ~0.43 (out-of-fold). AST was pretrained on AudioSet, which already knows engines,
aircraft, buzzing and explosions; a small linear model on its scores generalises better.

What it does
  1. Runs MIT/ast-finetuned-audioset-10-10-0.4593 on every 5 s wav (train, valid, test) -> 527 AudioSet logits.
  2. Leave-one-clip-out over the 56 labelled clips (train + valid): logistic regression on 4 sound classes.
     wake_triggered = (sound_class == drone_buzz), so outputs never contradict.  Report THESE numbers.
  3. Fits on all 56 clips and writes the test JSONL.

Caveats: AST is an 86M-parameter transformer, not the kit's 3.6 MB CNN; Metis compilation is untested.
Whether outside pretrained audio models are allowed is an open question for the organisers.

Usage:
  python ch9_acoustic_wake/audioset_probe.py --challenge-dir <challenge-5-acoustic-visual-cuas> --out c9_test.jsonl
"""
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

CLASSES = ["drone_buzz", "vehicle_mechanical", "explosion", "ambient_outdoor"]
MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"


def load_wav(p: Path) -> np.ndarray:
    with wave.open(str(p)) as w:
        if w.getframerate() != 16000 or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{p}: expected 16 kHz mono 16-bit")
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def audioset_logits(challenge_dir: Path, cache: Path) -> dict:
    if cache.exists():
        return json.loads(cache.read_text())
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    fe = ASTFeatureExtractor.from_pretrained(MODEL)
    model = ASTForAudioClassification.from_pretrained(MODEL).eval()
    out = {}
    for split in ("train", "valid", "test"):
        for line in (challenge_dir / "dataset" / split / "labels.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            x = fe(load_wav(challenge_dir / r["audio"]), sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                logits = model(**x).logits[0].numpy()
            out[r["id"]] = {"split": split, "logits": logits.tolist(), "sound_class": r.get("sound_class")}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--challenge-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="test JSONL")
    ap.add_argument("--cache", type=Path, default=Path("runs/c9_audioset_logits.json"))
    ap.add_argument("--C", type=float, default=0.01, help="inverse L2 strength of the logistic regression")
    args = ap.parse_args()

    D = audioset_logits(args.challenge_dir, args.cache)
    lab = [k for k, v in D.items() if v["split"] != "test"]
    test = [k for k, v in D.items() if v["split"] == "test"]
    X = np.array([D[k]["logits"] for k in lab])
    c = np.array([CLASSES.index(D[k]["sound_class"]) for k in lab])
    y = c == 0

    def model():
        return make_pipeline(StandardScaler(), LogisticRegression(C=args.C, max_iter=5000, class_weight="balanced"))

    pred = np.zeros(len(c), int)
    for i in range(len(c)):
        tr = np.arange(len(c)) != i
        pred[i] = model().fit(X[tr], c[tr]).predict(X[i:i + 1])[0]
    wake = pred == 0
    tp, fp, fn = int((wake & y).sum()), int((wake & ~y).sum()), int((~wake & y).sum())
    print(json.dumps({"leave_one_clip_out_clips": len(c), "wake_F1": round(2 * tp / max(1, 2 * tp + fp + fn), 4),
                      "class_acc": round(float((pred == c).mean()), 4), "camera_on": round(float(wake.mean()), 4),
                      "tp": tp, "fp": fp, "fn": fn}))

    final = model().fit(X, c)
    tp_cls = final.predict(np.array([D[k]["logits"] for k in test]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps({"id": k, "wake_triggered": bool(p == 0), "sound_class": CLASSES[p]}) + "\n"
                                for k, p in zip(test, tp_cls)), encoding="utf-8")
    print(f"wrote {args.out}: {len(test)} clips, {int((tp_cls == 0).sum())} wakes")


if __name__ == "__main__":
    main()
