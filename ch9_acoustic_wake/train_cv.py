"""Siesta Ch 9: augmented training + cross-validated threshold + consistent wake/class rule.

Why (team brief ideas G + H):
  * The kit trainer has no augmentation and picks its threshold on only 8 valid clips (2 positives).
  * Here we pool train+valid (56 labelled clips), run stratified K-fold, and choose the
    wake threshold on OUT-OF-FOLD predictions (56 clips instead of 8).
  * Consistency: in every labelled clip, has_drone_audio == (sound_class == "drone_buzz").
    We fuse both heads: p = mean(sigmoid(wake_logit), softmax(class)[drone_buzz]);
    wake = p >= thr; class = drone_buzz if wake else best non-drone class.
    -> outputs can never contradict each other. Use ch9_acoustic_wake/predict_consistent.py to predict.
  * Spectrogram augmentation in tensor space (time axis = image width, frequency = height):
    time roll, gain shift, time/frequency masking, and mixing a non-drone clip into a clip
    with an element-wise max (log-mel approximation of adding sounds).

Final model: retrained on all 56 clips; ONNX keeps the kit's input/output names
(spectrogram -> wake_logit, sound_class_logits) and stores metadata
wake_threshold (fused) and siesta_rule="fused_mean".

Usage (Mac):
  python train_cv.py --challenge-dir <challenge-5-acoustic-visual-cuas> --device mps --out runs/c9_cv
Quick smoke test:
  python train_cv.py --challenge-dir ... --folds 2 --epochs 2 --no-pretrained --out /tmp/c9
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

THRESHOLDS = [round(t, 2) for t in np.arange(0.20, 0.96, 0.05)]


def load_kit(challenge_dir: Path):
    path = challenge_dir / "starter-kit" / "train_audio_cnn_axelera.py"
    spec = importlib.util.spec_from_file_location("kit_c9_train", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kit_c9_train"] = mod
    spec.loader.exec_module(mod)
    return mod


class Rows(Dataset):
    def __init__(self, base_items, augment: bool, mix_pool=None):
        self.items = base_items          # list of (tensor, has_drone, class_idx)
        self.augment = augment
        self.mix_pool = mix_pool or []

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x, y, c = self.items[i]
        x = x.clone()
        if self.augment:
            W = x.shape[-1]
            x = torch.roll(x, shifts=random.randint(-W // 4, W // 4), dims=-1)       # time shift
            x = (x + random.uniform(-0.08, 0.08)).clamp(0, 1)                         # gain
            for _ in range(2):                                                         # time masks
                w = random.randint(0, W // 10)
                s = random.randint(0, W - w)
                x[..., :, s:s + w] = x.mean()
            Hh = x.shape[-2]
            h = random.randint(0, Hh // 12)                                            # freq mask
            s = random.randint(0, Hh - h)
            x[..., s:s + h, :] = x.mean()
            if self.mix_pool and random.random() < 0.4:                                # add background
                bx = random.choice(self.mix_pool)
                x = torch.maximum(x, bx * random.uniform(0.6, 0.95))
        return x, torch.tensor(float(y)), torch.tensor(int(c))


def fused_probs(model, xs, device, drone_idx):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(xs), 16):
            wl, cl = model(torch.stack(xs[i:i + 16]).to(device))
            pw = torch.sigmoid(wl).cpu().numpy()
            pc = torch.softmax(cl, dim=-1).cpu().numpy()
            out.append(np.column_stack([pw, pc]))
    return np.concatenate(out)                      # [:,0]=wake prob, [:,1:]=class probs


def decide(probs, thr, drone_idx):
    fused = 0.5 * (probs[:, 0] + probs[:, 1 + drone_idx])
    wake = fused >= thr
    cls = []
    for p, w in zip(probs[:, 1:], wake):
        if w:
            cls.append(drone_idx)
        else:
            q = p.copy()
            q[drone_idx] = -1
            cls.append(int(np.argmax(q)))
    return wake, np.array(cls), fused


def metrics(wake, cls, y, c):
    tp = int((wake & y).sum()); fp = int((wake & ~y).sum()); fn = int((~wake & y).sum())
    return {"wake_F1": round(2 * tp / max(1, 2 * tp + fp + fn), 4), "class_acc": round(float((cls == c).mean()), 4),
            "camera_on": round(float(wake.mean()), 4), "tp": tp, "fp": fp, "fn": fn}


def train_one(kit, items, args, device, mix_pool):
    model = kit.AcousticWakeCNN(args.arch, pretrained=not args.no_pretrained).to(device)
    ys = np.array([it[1] for it in items])
    pos, neg = ys.sum(), len(ys) - ys.sum()
    pos_w = torch.tensor([float(min(2.0, max(1.0, (neg / max(1, pos)) ** 0.5)))], dtype=torch.float32, device=device)
    bce, ce = nn.BCEWithLogitsLoss(pos_weight=pos_w), nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    dl = DataLoader(Rows(items, not args.no_augment, mix_pool), batch_size=args.batch_size, shuffle=True, num_workers=0)
    for _ in range(args.epochs):
        model.train()
        for x, y, c in dl:
            opt.zero_grad()
            wl, cl = model(x.to(device))
            loss = bce(wl, y.to(device)) + 0.5 * ce(cl, c.to(device))
            loss.backward()
            opt.step()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--challenge-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arch", default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18"])
    ap.add_argument("--folds", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-augment", action="store_true", help="kit-style training, no augmentation")
    ap.add_argument("--duty-penalty", type=float, default=0.15, help="same utility as the kit: F1 - p*camera_on")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    kit = load_kit(args.challenge_dir)
    classes = list(kit.SOUND_CLASSES)
    drone_idx = classes.index("drone_buzz")
    items, ids = [], []
    for split in ("train", "valid"):
        ds = kit.SpectrogramDataset(args.challenge_dir, split, 224)
        for i in range(len(ds)):
            r = ds[i]
            items.append((r["spectrogram"], int(r["has_drone"].item() > 0.5), int(r["sound_class"].item())))
            ids.append(ds.rows[i]["id"])
    y = np.array([it[1] for it in items]).astype(bool)
    c = np.array([it[2] for it in items])
    mix_pool = [it[0] for it in items if it[2] != drone_idx]

    # stratified folds by class
    folds = [[] for _ in range(args.folds)]
    for cls_i in sorted(set(c)):
        idx = [i for i in range(len(items)) if c[i] == cls_i]
        random.shuffle(idx)
        for k, i in enumerate(idx):
            folds[k % args.folds].append(i)
    oof = np.zeros((len(items), 1 + len(classes)))
    for k, val_idx in enumerate(folds):
        tr_idx = [i for i in range(len(items)) if i not in set(val_idx)]
        pool = [items[i][0] for i in tr_idx if items[i][2] != drone_idx]
        model = train_one(kit, [items[i] for i in tr_idx], args, device, pool)
        oof[val_idx] = fused_probs(model, [items[i][0] for i in val_idx], device, drone_idx)
        print(f"fold {k + 1}/{args.folds} done ({len(val_idx)} clips)")

    best = None
    for thr in THRESHOLDS:
        wake, cls, _ = decide(oof, thr, drone_idx)
        m = metrics(wake, cls, y, c)
        u = m["wake_F1"] - args.duty_penalty * m["camera_on"]
        if best is None or u > best[0]:
            best = (u, thr, m)
    _, thr, m = best
    print("OUT-OF-FOLD (56 clips) at fused thr", thr, json.dumps(m))
    best_k = None
    for thr2 in THRESHOLDS:
        wake2 = oof[:, 0] >= thr2
        cls2 = np.argmax(oof[:, 1:], axis=1)
        m2 = metrics(wake2, cls2, y, c)
        u2 = m2["wake_F1"] - args.duty_penalty * m2["camera_on"]
        if best_k is None or u2 > best_k[0]:
            best_k = (u2, thr2, m2)
    print("OUT-OF-FOLD kit-style rule (wake prob only, class argmax) at thr", best_k[1], json.dumps(best_k[2]))

    final = train_one(kit, items, args, device, mix_pool)
    final.eval().cpu()
    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / "audio_wake_cnn.onnx"
    torch.onnx.export(final, (torch.rand(1, 3, 224, 224),), str(onnx_path), opset_version=17,
                      input_names=["spectrogram"], output_names=["wake_logit", "sound_class_logits"], dynamo=False)
    import onnx
    mp = onnx.load(onnx_path)
    meta = {e.key: e.value for e in mp.metadata_props}
    meta.update({"wake_threshold": str(thr), "siesta_rule": "fused_mean"})
    onnx.helper.set_model_props(mp, meta)
    onnx.save(mp, onnx_path)
    (args.out / "cv_report.json").write_text(json.dumps({"fused_threshold": thr, "oof": m, "args": vars(args)},
                                                       default=str, indent=2))
    print(f"Saved {onnx_path} (fused wake_threshold={thr}). Predict with ch9_acoustic_wake/predict_consistent.py")


if __name__ == "__main__":
    main()
