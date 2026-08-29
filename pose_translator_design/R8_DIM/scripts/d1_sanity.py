"""Calibrate the estimators on data whose true intrinsic dimension we know."""
import json, numpy as np, torch
from d0_common import *

torch.manual_seed(0)
rows = []
N = 12000

# 1. isotropic Gaussian of true dim d, embedded (rotated) into a 200-dim ambient space
for d in [2, 5, 10, 20, 50, 100]:
    Z = torch.randn(N, d, device=DEV)
    Q = torch.linalg.qr(torch.randn(200, d, device=DEV))[0]
    X = Z @ Q.T
    r, _ = all_estimators(X, exclude_same_group=False)
    r["case"] = f"gauss_d{d}"; r["true_d"] = d
    rows.append(r); print(r)

# 2. same but with a fast-decaying spectrum (like real data: no hard cutoff)
for d, alpha in [(50, 1.0), (200, 1.0)]:
    Z = torch.randn(N, d, device=DEV) * (torch.arange(1, d + 1, device=DEV).float() ** (-alpha))
    Q = torch.linalg.qr(torch.randn(300, d, device=DEV))[0]
    X = Z @ Q.T
    r, _ = all_estimators(X, exclude_same_group=False)
    r["case"] = f"gauss_d{d}_pow{alpha}"; r["true_d"] = d
    rows.append(r); print(r)

# 3. curved manifold: d-dim sphere surface embedded in d+1, then rotated into 200
for d in [5, 20]:
    Z = torch.randn(N, d + 1, device=DEV)
    Z = Z / Z.norm(dim=1, keepdim=True)
    Q = torch.linalg.qr(torch.randn(200, d + 1, device=DEV))[0]
    r, _ = all_estimators(Z @ Q.T, exclude_same_group=False)
    r["case"] = f"sphere_d{d}"; r["true_d"] = d
    rows.append(r); print(r)

# 4. sample-size dependence of the NN estimators at true d=50
for n in [1000, 3000, 10000, 30000]:
    Z = torch.randn(n, 50, device=DEV)
    Q = torch.linalg.qr(torch.randn(200, 50, device=DEV))[0]
    r, _ = all_estimators(Z @ Q.T, exclude_same_group=False)
    r["case"] = f"gauss_d50_n{n}"; r["true_d"] = 50
    rows.append(r); print(r)

json.dump(rows, open(f"{SP}/nums_d1_sanity.json", "w"), indent=1)
print("done")
