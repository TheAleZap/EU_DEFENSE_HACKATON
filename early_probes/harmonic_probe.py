"""Feasibility probe for a physics-based drone-buzz cue on Challenge 9 audio.
Computes simple harmonicity features per 5 s clip and checks, with leave-one-out
on train+valid, how well they alone separate drone_buzz from everything else.

Usage: python3 -I harmonic_probe.py <challenge-5 dir>
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

def feats(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x /= (np.abs(x).max() + 1e-9)
    f, _, Z = stft(x, fs=sr, nperseg=4096, noverlap=3072)
    S = np.abs(Z) + 1e-9
    spec = np.median(S, axis=1)                       # robust long-term spectrum
    band = (f >= 60) & (f <= 4000)
    logs = np.log(spec[band]); fb = f[band]
    # peakiness: how much narrow peaks stand above a smoothed floor
    k = 25
    floor = np.convolve(logs, np.ones(k) / k, mode="same")
    resid = logs - floor
    peakiness = float(np.mean(np.sort(resid)[-20:]))
    # harmonic comb score: best f0 in 80-400 Hz whose first 8 harmonics sit on peaks
    best = -1e9
    for f0 in np.arange(80, 400, 1.0):
        idx = [np.argmin(np.abs(fb - h * f0)) for h in range(1, 9) if h * f0 <= 4000]
        if len(idx) >= 4:
            best = max(best, float(np.mean(resid[idx])))
    # temporal stability of the strongest peak (rotor tones are steady)
    peak_bin = np.argmax(S[band].mean(axis=1))
    track = np.log(S[band][peak_bin]); stability = float(-np.std(track))
    flat = float(np.exp(np.mean(np.log(spec[band]))) / np.mean(spec[band]))
    return [peakiness, best, stability, flat]

def main():
    root = Path(sys.argv[1])
    X, y, cls = [], [], []
    for split in ("train", "valid"):
        for line in (root / "dataset" / split / "labels.jsonl").read_text().splitlines():
            r = json.loads(line)
            wav = root / "audio" / split / f"{r['id']}.wav"
            X.append(feats(wav)); y.append(bool(r["has_drone_audio"])); cls.append(r["sound_class"])
    X = np.asarray(X); y = np.asarray(y)
    names = ["peakiness", "harmonic_comb", "peak_stability", "flatness"]
    for j, n in enumerate(names):
        print(f"{n:15s} drone mean {X[y, j].mean():+.3f}  other mean {X[~y, j].mean():+.3f}")
    # leave-one-out nearest-centroid on standardized features
    correct = 0; tp = fp = fn = 0
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        mu, sd = X[m].mean(0), X[m].std(0) + 1e-9
        Z = (X - mu) / sd
        c1, c0 = Z[m & y].mean(0), Z[m & ~y].mean(0)
        pred = np.linalg.norm(Z[i] - c1) < np.linalg.norm(Z[i] - c0)
        correct += pred == y[i]; tp += pred and y[i]; fp += pred and not y[i]; fn += (not pred) and y[i]
    print(f"LOO accuracy {correct}/{len(y)} | wake F1 {2*tp/max(1,2*tp+fp+fn):.2f} (tp={tp} fp={fp} fn={fn}) | positives {y.sum()}")

if __name__ == "__main__":
    main()
