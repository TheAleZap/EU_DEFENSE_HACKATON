"""Siesta Ch 9: AudioSet-pretrained audio model (AST) + logistic regression on its 527 AudioSet scores.

Why: with 56 labelled clips, fine-tuning the kit's ImageNet MobileNet on spectrogram PNGs reaches only
wake F1 ~0.49 / class acc ~0.43 (out-of-fold). AST was pretrained on AudioSet, which already knows engines,
aircraft, buzzing and explosions; a small linear model on its scores generalises better.

What it does
  1. Runs MIT/ast-finetuned-audioset-10-10-0.4593 on every 5 s wav (train, valid, test) -> 527 AudioSet logits.
  2. Leave-one-clip-out over the 56 labelled clips (train + valid): logistic regression on 4 sound classes.
     wake_triggered = (sound_class == drone_buzz), so outputs never contradict.  Report THESE numbers.
  3. Fits on all 56 clips and writes the test JSONL.

Extra data (--extra <audio_export>): each clip folder holds audio.wav + analysis.json with a clip-level
has_drone_audio flag. Clips are cut into 5 s windows (at most --extra-max-windows per clip, evenly spaced)
and added to the TRAINING side only, for a separate drone/no-drone wake model. The 4-class model stays on
kit clips. Evaluation is always leave-one-clip-out on the 56 kit clips; kit-only and kit+extra are printed.

Caveats: AST is an 86M-parameter transformer, not the kit's 3.6 MB CNN; Metis compilation is untested.
Whether outside pretrained audio models and extra data are allowed is an open question for the organisers.

Usage:
  python ch9_acoustic_wake/audioset_probe.py --challenge-dir <challenge-5-acoustic-visual-cuas> --out c9_test.jsonl \
      [--extra ~/edth/extra/audio_export]
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


def _ast():
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    return ASTFeatureExtractor.from_pretrained(MODEL), ASTForAudioClassification.from_pretrained(MODEL).eval()


def _logits(fe, model, audio: np.ndarray) -> list:
    import torch
    x = fe(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        return model(**x).logits[0].numpy().tolist()


def extra_logits(extra_dir: Path, max_windows: int, cache: Path) -> list:
    """[(clip_id, has_drone, logits)] for evenly spaced 5 s windows of each labelled clip."""
    if cache.exists():
        return [tuple(r) for r in json.loads(cache.read_text())]
    fe, model = _ast()
    out, win = [], 5 * 16000
    for d in sorted(p for p in extra_dir.iterdir() if (p / "analysis.json").exists() and (p / "audio.wav").exists()):
        label = bool(json.loads((d / "analysis.json").read_text())["has_drone_audio"])
        a = load_wav(d / "audio.wav")
        n = max(1, len(a) // win)
        starts = sorted({int(round(s)) for s in np.linspace(0, n - 1, min(n, max_windows))})
        for s in starts:
            seg = a[s * win:(s + 1) * win]
            out.append((d.name, label, _logits(fe, model, seg)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def audioset_logits(challenge_dir: Path, cache: Path) -> dict:
    if cache.exists():
        return json.loads(cache.read_text())
    fe, model = _ast()
    out = {}
    for split in ("train", "valid", "test"):
        for line in (challenge_dir / "dataset" / split / "labels.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out[r["id"]] = {"split": split, "logits": _logits(fe, model, load_wav(challenge_dir / r["audio"])),
                            "sound_class": r.get("sound_class")}
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
    ap.add_argument("--extra", type=Path, default=None, help="audio_export folder (training only)")
    ap.add_argument("--extra-max-windows", type=int, default=6)
    ap.add_argument("--extra-cache", type=Path, default=Path("runs/c9_audioset_extra_logits.json"))
    ap.add_argument("--extra-pos-weight", type=float, default=1.0,
                    help="sample weight of extra drone windows (their labels are clip-level, so noisy)")
    args = ap.parse_args()

    D = audioset_logits(args.challenge_dir, args.cache)
    lab = [k for k, v in D.items() if v["split"] != "test"]
    test = [k for k, v in D.items() if v["split"] == "test"]
    X = np.array([D[k]["logits"] for k in lab])
    XT = np.array([D[k]["logits"] for k in test])
    c = np.array([CLASSES.index(D[k]["sound_class"]) for k in lab])
    y = c == 0

    E = extra_logits(args.extra, args.extra_max_windows, args.extra_cache) if args.extra else []
    XE = np.array([e[2] for e in E]) if E else np.zeros((0, X.shape[1]))
    yE = np.array([e[1] for e in E], bool)
    wE = np.where(yE, args.extra_pos_weight, 1.0)
    if E:
        print(f"extra: {len({e[0] for e in E})} clips, {len(E)} windows ({int(yE.sum())} drone, {int((~yE).sum())} not)")

    def lr():
        return make_pipeline(StandardScaler(), LogisticRegression(C=args.C, max_iter=5000, class_weight="balanced"))

    def fit_predict(tr, Xq, use_extra):
        """4-class on kit clips; with extra data, wake comes from a separate binary model on kit + extra."""
        cls = lr().fit(X[tr], c[tr])
        if not use_extra:
            p = cls.predict(Xq)
            return p == 0, p
        Xw = np.vstack([X[tr], XE]); yw = np.concatenate([y[tr], yE]); sw = np.concatenate([np.ones(tr.sum()), wE])
        wake = lr().fit(Xw, yw, logisticregression__sample_weight=sw).predict(Xq).astype(bool)
        probs = cls.predict_proba(Xq)
        probs[:, 0] = -1
        return wake, np.where(wake, 0, probs.argmax(1))

    def report(name, use_extra):
        wake, pred = np.zeros(len(c), bool), np.zeros(len(c), int)
        for i in range(len(c)):
            tr = np.arange(len(c)) != i
            w, p = fit_predict(tr, X[i:i + 1], use_extra)
            wake[i], pred[i] = w[0], p[0]
        tp, fp, fn = int((wake & y).sum()), int((wake & ~y).sum()), int((~wake & y).sum())
        print(name, json.dumps({"leave_one_clip_out_clips": len(c), "wake_F1": round(2 * tp / max(1, 2 * tp + fp + fn), 4),
                                "class_acc": round(float((pred == c).mean()), 4), "camera_on": round(float(wake.mean()), 4),
                                "tp": tp, "fp": fp, "fn": fn}))

    report("kit only   ", False)
    if E:
        report("kit + extra", True)

    wake_t, pred_t = fit_predict(np.ones(len(c), bool), XT, bool(E))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps({"id": k, "wake_triggered": bool(w), "sound_class": CLASSES[p]}) + "\n"
                                for k, w, p in zip(test, wake_t, pred_t)), encoding="utf-8")
    print(f"wrote {args.out} ({'kit + extra' if E else 'kit only'}): {len(test)} clips, {int(wake_t.sum())} wakes")


if __name__ == "__main__":
    main()
