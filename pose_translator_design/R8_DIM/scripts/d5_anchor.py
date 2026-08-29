"""Outside-motion anchors: image patches / whole images, and an audio-like proxy.

Honest note on what we can and cannot get locally:
 - sklearn digits (8x8, 1797 imgs) is a REAL image dataset but tiny + very constrained.
 - sklearn load_sample_images gives 2 natural photos -> we can cut thousands of 8x8 / 16x16
   natural-image patches, which is the classic "natural image patch manifold" benchmark.
 - No audio corpus is installed (no librosa/torchaudio, no wav files on disk). We therefore
   do NOT fabricate an audio number; instead we note it as missing.
"""
import json, numpy as np, torch
from d0_common import *
from sklearn.datasets import load_digits, load_sample_images

rows = []


def add(X, case, note=""):
    r, _ = all_estimators(torch.as_tensor(X, dtype=torch.float32), exclude_same_group=False)
    r["case"] = case; r["note"] = note
    rows.append(r); print(r)


# --- 1. sklearn digits: 8x8 = 64-D, 1797 samples ---
d = load_digits()
add(d.data / 16.0, "images_digits_8x8", "1797 imgs, 64-D ambient (small N -> NN estimators biased low)")

# --- 2. natural image patches from the 2 sample photos ---
imgs = load_sample_images().images
g = np.random.default_rng(0)
for P in [8, 16]:
    patches = []
    for im in imgs:
        a = im.astype(np.float32) / 255.0
        a = a.mean(-1)                       # grayscale
        H, W = a.shape
        ys = g.integers(0, H - P, 9000); xs = g.integers(0, W - P, 9000)
        for y, x in zip(ys, xs):
            patches.append(a[y:y + P, x:x + P].ravel())
    X = np.stack(patches)
    Xc = X - X.mean(1, keepdims=True)        # remove DC, the classic preprocessing
    add(Xc, f"images_natural_patch_{P}x{P}", f"{len(X)} grayscale patches, {P*P}-D ambient, DC removed")

# --- 3. colour natural patches, no DC removal (rawest possible image data) ---
patches = []
for im in imgs:
    a = im.astype(np.float32) / 255.0
    H, W = a.shape[:2]
    ys = g.integers(0, H - 8, 9000); xs = g.integers(0, W - 8, 9000)
    for y, x in zip(ys, xs):
        patches.append(a[y:y + 8, x:x + 8].ravel())
add(np.stack(patches), "images_natural_patch_8x8_rgb", "192-D ambient, raw RGB")

# --- 4. audio-like proxy, clearly labelled as SYNTHETIC ---
# 25 ms frames of a synthetic mixture of harmonic + noise sources, then a log-mel-ish
# filterbank. This is a proxy, NOT real speech/music. It exists only to show that a
# generic dense signal in a comparable ambient dimension does not read as ~7-D.
sr, T = 16000, 400        # 25 ms @ 16 kHz
n = 12000
t = torch.arange(T, device=DEV).float() / sr
f0 = 80 + 320 * torch.rand(n, 1, device=DEV)
sig = torch.zeros(n, T, device=DEV)
for h in range(1, 13):
    amp = torch.rand(n, 1, device=DEV) / h
    ph = 2 * np.pi * torch.rand(n, 1, device=DEV)
    sig += amp * torch.sin(2 * np.pi * f0 * h * t[None] + ph)
sig += 0.15 * torch.randn(n, T, device=DEV)
S = torch.fft.rfft(sig * torch.hann_window(T, device=DEV)[None]).abs()
edges = torch.linspace(0, S.shape[1] - 1, 65, device=DEV).long()
mel = torch.stack([S[:, edges[i]:edges[i + 1] + 1].mean(1) for i in range(64)], 1)
add(torch.log(mel + 1e-6).cpu().numpy(), "audio_PROXY_synth_logmel64",
    "SYNTHETIC harmonic+noise, 64-D log-mel. Not a real audio corpus -- proxy only.")

json.dump(rows, open(f"{SP}/nums_d5_anchor.json", "w"), indent=1)
print("done")
