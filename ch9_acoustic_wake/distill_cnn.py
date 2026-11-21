"""Siesta Ch 9: distil the AudioSet teacher into the kit's small Metis CNN, using the extra recordings.

Why: the kit CNN (MobileNetV3-Small, 3.6 MB, compiles for Metis) only has 48-56 labelled clips. The
extra audio_export clips have only clip-level drone/no-drone flags written by an AI, so instead of those
flags we label every 5 s window with the AudioSet teacher (audioset_probe.py: AST logits + logistic
regression on kit clips), which gives soft scores for the kit's 4 sound classes.

Spectrograms: our own log-mel (16 kHz, n_fft 1024, hop 358 -> 224 frames, 128 mel bands, 80 dB range,
scaled to [0,1], resized to 224x224, 3 channels), computed from the wavs for kit, test and extra clips
alike. The exported ONNX keeps the kit's names (spectrogram -> wake_logit, sound_class_logits); its
input is OUR spectrogram, so predict with this script, not the kit predictor.

Evaluation: stratified K-fold over the 56 kit clips. In each fold the teacher is refit on the training
clips only, extra windows are added to the training side only, and the held-out kit clips are scored.
Rule: sound_class = argmax, wake = (sound_class == drone_buzz), so outputs never contradict.

Usage:
  python ch9_acoustic_wake/distill_cnn.py --challenge-dir <c9> --extra ~/edth/extra/audio_export \
      --kit-logits runs/c9_audioset_logits.json --extra-logits runs/c9_audioset_extra_logits_w6.json \
      --device mps --out runs/c9_distill [--no-extra]
  (run audioset_probe.py with --extra first: it creates both logits caches)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import wave
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ["drone_buzz", "vehicle_mechanical", "explosion", "ambient_outdoor"]
SR, WIN, N_FFT, HOP, N_MELS = 16000, 80000, 1024, 358, 128


def load_wav(p: Path) -> np.ndarray:
    with wave.open(str(p)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768


def mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, sr=SR, fmin=20.0, fmax=8000.0) -> torch.Tensor:
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700)
    mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(bins)), np.float32)
    for m in range(n_mels):
        lo, ce, hi = pts[m], pts[m + 1], pts[m + 2]
        fb[m] = np.clip(np.minimum((bins - lo) / (ce - lo), (hi - bins) / (hi - ce)), 0, None)
    return torch.from_numpy(fb)


FB = mel_filterbank()


def spectrogram(audio: np.ndarray) -> torch.Tensor:
    a = np.zeros(WIN, np.float32)
    a[:min(WIN, len(audio))] = audio[:WIN]
    s = torch.stft(torch.from_numpy(a), N_FFT, HOP, window=torch.hann_window(N_FFT), return_complex=True).abs() ** 2
    db = 10 * torch.log10(FB @ s + 1e-10)
    db = torch.clamp(db, min=db.max() - 80)
    img = ((db - db.min()) / (db.max() - db.min() + 1e-6)).flip(0)[None, None]
    img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)[0]
    return img.repeat(3, 1, 1)


def load_kit(challenge_dir: Path):
    spec = importlib.util.spec_from_file_location("kit_c9_train", challenge_dir / "starter-kit" / "train_audio_cnn_axelera.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kit_c9_train"] = mod
    spec.loader.exec_module(mod)
    return mod


def extra_windows(extra_dir: Path, max_windows: int):
    """Same window positions as audioset_probe.extra_logits."""
    out = []
    for d in sorted(p for p in extra_dir.iterdir() if (p / "analysis.json").exists() and (p / "audio.wav").exists()):
        a = load_wav(d / "audio.wav")
        n = max(1, len(a) // WIN)
        for s in sorted({int(round(s)) for s in np.linspace(0, n - 1, min(n, max_windows))}):
            out.append(spectrogram(a[s * WIN:(s + 1) * WIN]))
    return out


def teacher(Xk, ck, Xq, C=0.01):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
    return m.fit(Xk, ck).predict_proba(Xq)


def train_student(kit, specs, targets, weights, args, device):
    model = kit.AcousticWakeCNN("mobilenet_v3_small", pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    X = torch.stack(specs); T = torch.tensor(np.array(targets), dtype=torch.float32); W = torch.tensor(weights, dtype=torch.float32)
    n = len(X)
    for _ in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            x = X[idx].clone()
            x = torch.roll(x, random.randint(-56, 56), dims=-1)
            x = (x + random.uniform(-0.05, 0.05)).clamp(0, 1)
            x, t, w = x.to(device), T[idx].to(device), W[idx].to(device)
            wl, cl = model(x)
            ce = -(t * F.log_softmax(cl, -1)).sum(-1)
            bce = F.binary_cross_entropy_with_logits(wl, t[:, 0], reduction="none")
            loss = ((ce + bce) * w).sum() / w.sum()
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def predict(model, specs, device):
    with torch.no_grad():
        out = []
        for i in range(0, len(specs), 32):
            _, cl = model(torch.stack(specs[i:i + 32]).to(device))
            out.append(cl.argmax(-1).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--challenge-dir", type=Path, required=True)
    ap.add_argument("--extra", type=Path, default=None)
    ap.add_argument("--kit-logits", type=Path, required=True)
    ap.add_argument("--extra-logits", type=Path, default=None)
    ap.add_argument("--extra-max-windows", type=int, default=6)
    ap.add_argument("--extra-weight", type=float, default=0.5)
    ap.add_argument("--extra-ids", type=Path, default=None, help="keep only these extra clip ids (select_drone_clips.py)")
    ap.add_argument("--no-extra", action="store_true", help="kit clips only (comparison)")
    ap.add_argument("--folds", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    kit = load_kit(args.challenge_dir)

    L = json.loads(args.kit_logits.read_text())
    lab = [k for k, v in L.items() if v["split"] != "test"]
    test = [k for k, v in L.items() if v["split"] == "test"]
    rows = {}
    for split in ("train", "valid", "test"):
        for line in (args.challenge_dir / "dataset" / split / "labels.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line); rows[r["id"]] = r
    S = {k: spectrogram(load_wav(args.challenge_dir / rows[k]["audio"])) for k in lab + test}
    Xk = np.array([L[k]["logits"] for k in lab]); ck = np.array([CLASSES.index(L[k]["sound_class"]) for k in lab]); y = ck == 0

    use_extra = not args.no_extra
    if use_extra:
        E = json.loads(args.extra_logits.read_text())
        XE = np.array([e[2] for e in E])
        SE = extra_windows(args.extra, args.extra_max_windows)
        assert len(SE) == len(E), (len(SE), len(E))
        if args.extra_ids:
            keep_ids = set(args.extra_ids.read_text().split())
            keep = [i for i, e in enumerate(E) if e[0] in keep_ids]
            XE, SE = XE[keep], [SE[i] for i in keep]
            print(f"--extra-ids: {len({E[i][0] for i in keep})} clips kept")
        print(f"extra windows: {len(SE)}")

    folds = [[] for _ in range(args.folds)]
    for ci in range(4):
        idx = [i for i in range(len(lab)) if ck[i] == ci]
        random.shuffle(idx)
        for j, i in enumerate(idx):
            folds[j % args.folds].append(i)

    def build(tr):
        specs = [S[lab[i]] for i in tr]
        targets = [np.eye(4)[ck[i]] for i in tr]
        weights = [1.0] * len(tr)
        if use_extra:
            specs += SE
            targets += list(teacher(Xk[tr], ck[tr], XE))
            weights += [args.extra_weight] * len(SE)
        return specs, targets, weights

    pred = np.zeros(len(lab), int)
    for f, val in enumerate(folds):
        tr = np.array([i for i in range(len(lab)) if i not in set(val)])
        m = train_student(kit, *build(tr), args, device)
        pred[val] = predict(m, [S[lab[i]] for i in val], device)
        print(f"fold {f + 1}/{args.folds} done")
    wake = pred == 0
    tp, fp, fn = int((wake & y).sum()), int((wake & ~y).sum()), int((~wake & y).sum())
    oof = {"clips": len(lab), "wake_F1": round(2 * tp / max(1, 2 * tp + fp + fn), 4), "class_acc": round(float((pred == ck).mean()), 4),
           "camera_on": round(float(wake.mean()), 4), "tp": tp, "fp": fp, "fn": fn, "extra": use_extra}
    print("OUT-OF-FOLD", json.dumps(oof))

    final = train_student(kit, *build(np.arange(len(lab))), args, device).cpu()
    args.out.mkdir(parents=True, exist_ok=True)
    pt = predict(final, [S[k] for k in test], torch.device("cpu"))
    (args.out / "c9_test.jsonl").write_text("".join(json.dumps({"id": k, "wake_triggered": bool(p == 0), "sound_class": CLASSES[p]}) + "\n"
                                                    for k, p in zip(test, pt)), encoding="utf-8")
    onnx_path = args.out / "audio_wake_cnn.onnx"
    torch.onnx.export(final, (torch.rand(1, 3, 224, 224),), str(onnx_path), opset_version=17,
                      input_names=["spectrogram"], output_names=["wake_logit", "sound_class_logits"], dynamo=False)
    import onnx
    mp = onnx.load(onnx_path)
    onnx.helper.set_model_props(mp, {"siesta_rule": "argmax_class", "spectrogram": f"logmel sr{SR} nfft{N_FFT} hop{HOP} mels{N_MELS} 80dB"})
    onnx.save(mp, onnx_path)
    (args.out / "cv_report.json").write_text(json.dumps({"oof": oof, "args": vars(args)}, default=str, indent=2))
    print(f"wrote {args.out}/c9_test.jsonl ({int((pt == 0).sum())} wakes) and {onnx_path}")


if __name__ == "__main__":
    main()
