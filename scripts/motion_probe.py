"""Quick feasibility probe: can camera-motion-compensated 3-frame differencing
find the tiny targets in the Challenge 6 validation tracking sequences?

Usage: python3 -I motion_probe.py <tracking_eval_dir> <out_pred_dir>
Writes a naive motion-only MOT file per valid sequence for the official scorer.
"""
import sys, json
from pathlib import Path
import numpy as np
import cv2

def read_frames(mp4):
    cap = cv2.VideoCapture(str(mp4))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    return out

def align(src, dst, scale=0.5):
    """Estimate affine warp mapping src onto dst (global camera motion)."""
    s = cv2.resize(src, None, fx=scale, fy=scale)
    d = cv2.resize(dst, None, fx=scale, fy=scale)
    p0 = cv2.goodFeaturesToTrack(s, 400, 0.01, 8)
    if p0 is None or len(p0) < 8:
        return np.eye(2, 3, dtype=np.float32)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(s, d, p0, None)
    good = st.ravel() == 1
    if good.sum() < 8:
        return np.eye(2, 3, dtype=np.float32)
    M, _ = cv2.estimateAffinePartial2D(p0[good], p1[good], method=cv2.RANSAC)
    if M is None:
        return np.eye(2, 3, dtype=np.float32)
    M[:, 2] /= scale
    return M.astype(np.float32)

def motion_candidates(prev, cur, nxt, k_sigma=4.0, min_area=2, max_area=4000):
    h, w = cur.shape
    a = cv2.warpAffine(prev, align(prev, cur), (w, h), borderMode=cv2.BORDER_REPLICATE)
    b = cv2.warpAffine(nxt, align(nxt, cur), (w, h), borderMode=cv2.BORDER_REPLICATE)
    c = cur.astype(np.int16)
    d = np.minimum(np.abs(c - a.astype(np.int16)), np.abs(c - b.astype(np.int16))).astype(np.uint8)
    d = cv2.GaussianBlur(d, (3, 3), 0)
    thr = float(d.mean() + k_sigma * d.std())
    m = (d > max(thr, 12)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, cents = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if min_area <= area <= max_area:
            out.append((float(x), float(y), float(bw), float(bh), float(d[y:y+bh, x:x+bw].max())))
    return out

def iou(a, b):
    ax2, ay2, bx2, by2 = a[0]+a[2], a[1]+a[3], b[0]+b[2], b[1]+b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0])); ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max(1e-9, a[2]*a[3] + b[2]*b[3] - inter)

def main():
    root, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    man = json.loads((root / "sequences_manifest.json").read_text())
    for s in man["sequences"]:
        if s["split"] != "valid":
            continue
        seq = s["sequence"]
        frames = read_frames(root / seq / "video.mp4")
        gt = {}
        for line in (root / seq / "gt" / "gt.txt").read_text().splitlines():
            r = line.split(",")
            gt[int(r[0])] = tuple(map(float, r[2:6]))
        hit = 0; best_iou = []; ncand = []
        cands_by_frame = {}
        for t in range(1, len(frames) - 1):
            cands = motion_candidates(frames[t-1], frames[t], frames[t+1])
            fid = t + 1
            cands_by_frame[fid] = cands
            ncand.append(len(cands))
            g = gt.get(fid)
            if g is None:
                continue
            gx, gy = g[0] + g[2]/2, g[1] + g[3]/2
            pad = max(4.0, 0.5 * max(g[2], g[3]))
            near = [c for c in cands if abs(c[0]+c[2]/2-gx) <= g[2]/2+pad and abs(c[1]+c[3]/2-gy) <= g[3]/2+pad]
            if near:
                hit += 1
                best_iou.append(max(iou(c[:4], g) for c in near))
        n_gt = len(gt)
        print(f"{seq} {s['width']}x{s['height']} target~{np.median([g[2] for g in gt.values()]):.0f}px | "
              f"motion finds target in {hit}/{n_gt} frames | median IoU of hit {np.median(best_iou) if best_iou else 0:.2f} | "
              f"candidates/frame median {np.median(ncand):.0f}")
        # naive motion-only track: per frame keep the strongest candidate nearest the previous pick
        lines = []; prev = None
        for fid in sorted(cands_by_frame):
            cs = cands_by_frame[fid]
            if not cs:
                continue
            if prev is None:
                c = max(cs, key=lambda c: c[4])
            else:
                px, py = prev[0]+prev[2]/2, prev[1]+prev[3]/2
                c = min(cs, key=lambda c: (c[0]+c[2]/2-px)**2 + (c[1]+c[3]/2-py)**2 - 50*c[4])
            prev = c
            lines.append(f"{fid},1,{c[0]:.2f},{c[1]:.2f},{c[2]:.2f},{c[3]:.2f},1.0,1,1.0")
        (out_dir / f"{seq}.txt").write_text("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
