"""Siesta Ch 8: ranking-aware triage training (idea F) on top of the kit's model.

Changes vs the kit trainer (which uses MSE on priority with no augmentation):
  * Pairwise RankNet loss on the priority head: within each batch, for every pair whose
    teacher priorities differ by > 0.05, push the higher one above the lower one.
    The eval is rank-based (Spearman + top-k capture), so we train for ORDER.
  * Image augmentation: random resized crop, horizontal flip, colour jitter, random
    darkening (night-like).
  * Checkpoint selection reports both frame-level and clip-pooled valid metrics
    (pooling: ch8_edge_triage/pool_clips.py). Selection uses the kit composite on clip-pooled scores.
  * --final-epochs N: after selection, optionally retrain on train+valid for N epochs.

ONNX export keeps the kit's names (image -> keep_logit, priority_score, modality_logits,
perspective_logits, scene_logits), so the kit's predict_onnx_baseline.py triage still works.

Usage (Mac):  python train_rank.py --kit <challenge-4-edge-triage> --device mps --out runs/c8_rank
Smoke test:   python train_rank.py --kit ... --epochs 1 --no-pretrained --out /tmp/c8
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms


def load_kit(kit: Path):
    path = kit / "starter-kit" / "train_triage_distill_axelera.py"
    spec = importlib.util.spec_from_file_location("kit_c8_train", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kit_c8_train"] = mod
    spec.loader.exec_module(mod)
    return mod


def ranknet_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    logit = torch.logit(pred.clamp(1e-4, 1 - 1e-4))
    diff_t = target[:, None] - target[None, :]
    mask = diff_t > margin                                   # i should rank above j
    if not mask.any():
        return pred.sum() * 0.0
    diff_p = logit[:, None] - logit[None, :]
    return nn.functional.softplus(-diff_p[mask]).mean()


def budget_capture(scores, high, frac):
    order = np.lexsort((np.arange(len(scores)), high.astype(int), -scores))   # pessimistic ties (kit)
    k = max(1, int(round(len(scores) * frac)))
    return float(high[order[:k]].sum() / max(1, min(k, int(high.sum()))))


def spearman(a, b):
    """Tie-aware Spearman (average ranks); 0.0 when undefined (e.g. constant scores), like the kit."""
    from scipy.stats import spearmanr
    if len(a) < 3 or np.all(np.asarray(a) == a[0]) or np.all(np.asarray(b) == b[0]):
        return 0.0
    rho = spearmanr(a, b).correlation
    return 0.0 if rho is None or not np.isfinite(rho) else float(rho)


def pool(scores, clip_ids, eps=1e-3):
    by = defaultdict(list)
    for s, c in zip(scores, clip_ids):
        by[c].append(s)
    mean = {c: float(np.mean(v)) for c, v in by.items()}
    return np.array([round(mean[c] + eps * s, 6) for s, c in zip(scores, clip_ids)])


def evaluate(model, ds, device):
    model.eval()
    scores, prios, high = [], [], []
    with torch.no_grad():
        for b in DataLoader(ds, batch_size=32, shuffle=False, num_workers=0):
            _, p, *_ = model(b["image"].to(device))
            scores += [round(float(v), 6) for v in p.cpu()]
            prios += b["priority"].tolist()
            high += ((b["keep"] > 0.5) & (b["priority"] >= 0.7)).tolist()
    scores, prios, high = np.array(scores), np.array(prios), np.array(high, bool)
    clips = [r["video_id"] for r in ds.rows]
    pooled = pool(scores, clips)
    f = {"r1": budget_capture(scores, high, 0.01), "r5": budget_capture(scores, high, 0.05), "rho": spearman(scores, prios)}
    p = {"r1": budget_capture(pooled, high, 0.01), "r5": budget_capture(pooled, high, 0.05), "rho": spearman(pooled, prios)}
    return f, p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", type=Path, required=True, help="challenge-4-edge-triage folder")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arch", default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet18"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--final-epochs", type=int, default=0, help="retrain on train+valid for N epochs after selection")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rank-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    kit = load_kit(args.kit)
    ddir = args.kit / "dataset"
    aug = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.03),
        transforms.RandomApply([transforms.Lambda(lambda im: im.point(lambda v: int(v * 0.45)))], p=0.15),
        transforms.ToTensor(),
    ])
    train_ds = kit.EdgeTriageDataset(ddir, "train", 224); train_ds.transform = aug
    valid_ds = kit.EdgeTriageDataset(ddir, "valid", 224)

    def fit(datasets, epochs, select):
        model = kit.EdgeTriageNet(args.arch, pretrained=not args.no_pretrained).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, epochs))
        bce, mse, ce = nn.BCEWithLogitsLoss(), nn.MSELoss(), nn.CrossEntropyLoss()
        dl = DataLoader(ConcatDataset(datasets), batch_size=args.batch_size, shuffle=True, num_workers=0)
        best, best_ep, log = -1.0, 0, []
        for ep in range(1, epochs + 1):
            model.train()
            for b in dl:
                opt.zero_grad()
                kl, pp, ml, pl, sl = model(b["image"].to(device))
                pr = b["priority"].to(device)
                loss = (bce(kl, b["keep"].to(device)) + 0.5 * mse(pp, pr) + args.rank_weight * ranknet_loss(pp, pr)
                        + 0.5 * ce(ml, b["modality"].to(device)) + 0.5 * ce(pl, b["perspective"].to(device))
                        + bce(sl, b["scene"].to(device)))
                loss.backward()
                opt.step()
            sched.step()
            if select:
                f, p = evaluate(model, valid_ds, device)
                comp = 0.5 * (p["r1"] + p["r5"]) + 0.25 * p["rho"]
                log.append({"epoch": ep, "frame": f, "pooled": p})
                print(f"epoch {ep}: frame {f} | pooled {p}")
                if comp >= best:
                    best, best_ep = comp, ep
                    torch.save(model.state_dict(), args.out / "triage_best.pt")
        if select:
            model.load_state_dict(torch.load(args.out / "triage_best.pt", map_location=device, weights_only=True))
        return model, best_ep, log

    args.out.mkdir(parents=True, exist_ok=True)
    model, best_ep, log = fit([train_ds], args.epochs, select=True)
    print(f"best epoch {best_ep} (selected on clip-pooled valid; valid budgets are 1 and 7 frames, so noisy)")
    if args.final_epochs > 0:
        valid_aug = kit.EdgeTriageDataset(ddir, "valid", 224); valid_aug.transform = aug
        model, _, _ = fit([train_ds, valid_aug], args.final_epochs, select=False)
        print(f"retrained on train+valid for {args.final_epochs} epochs (valid metrics no longer independent)")
    model.eval().cpu()
    onnx_path = args.out / "triage_distill.onnx"
    torch.onnx.export(model, (torch.rand(1, 3, 224, 224),), str(onnx_path), opset_version=17, input_names=["image"],
                      output_names=["keep_logit", "priority_score", "modality_logits", "perspective_logits", "scene_logits"],
                      dynamo=False)
    (args.out / "train_log.json").write_text(json.dumps({"best_epoch": best_ep, "log": log, "args": vars(args)},
                                                        default=str, indent=2))
    print(f"Saved {onnx_path}. Predict with the kit, then pool with ch8_edge_triage/pool_clips.py")


if __name__ == "__main__":
    main()
