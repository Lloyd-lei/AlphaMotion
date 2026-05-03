"""
Evaluation: how well does a model's learned joint-joint structure recover the
ground-truth motor synergies?

The ground-truth synergies are rank-1 outer products of temporal × spatial
patterns. The spatial part is a [J, C] matrix whose joint-level "activation
strength" can be written as a [J, J] affinity via ψ · ψᵀ after collapsing
channels. A model has "recovered" a synergy if the principal components of its
learned pair structure align with those of the ground-truth affinity.

We quantify this by:
    1. Computing the ground-truth synergy affinity S_gt  ∈ R^{J × J}
        (a weighted sum of outer products of spatial patterns, one per synergy)
    2. Reading off the model's learned pair structure S_model ∈ R^{J × J}
    3. Principal subspace alignment: cosine similarity of the top-K eigenspaces
       of S_gt and S_model.
"""

from __future__ import annotations

import numpy as np
import torch


def ground_truth_affinity(
    temporal: np.ndarray,   # [K, T]
    spatial: np.ndarray,    # [K, J, C]
) -> np.ndarray:
    """Build a [J, J] affinity capturing which joints co-vary under the same synergy.

    For each synergy k, collapse channels via ψ_k · ψ_kᵀ  (a [J, J] rank-C matrix).
    Weight the K resulting matrices by the temporal-profile energy and sum.
    """
    K, J, C = spatial.shape
    energy = (temporal ** 2).sum(axis=1)              # [K]
    affinity = np.zeros((J, J), dtype=np.float64)
    for k in range(K):
        psi = spatial[k]                               # [J, C]
        affinity += energy[k] * (psi @ psi.T)
    # Normalise
    norm = np.linalg.norm(affinity) + 1e-8
    return (affinity / norm).astype(np.float32)


def principal_subspace_similarity(
    A: np.ndarray, B: np.ndarray, K: int
) -> float:
    """Grassmann-style similarity: cosine between top-K eigenspaces of A, B.

    Returns a scalar in [0, 1]: the average squared canonical correlation of
    the two K-dimensional subspaces. 1.0 iff the subspaces coincide.
    """
    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)
    _, U_A = np.linalg.eigh(A)
    _, U_B = np.linalg.eigh(B)
    # Top-K eigenvectors (eigh returns ascending order)
    U_A = U_A[:, -K:]
    U_B = U_B[:, -K:]
    # Cosine of principal angles
    M = U_A.T @ U_B               # [K, K]
    s = np.linalg.svd(M, compute_uv=False)
    return float(np.mean(s ** 2))


def eigen_spectrum(A: np.ndarray, top_k: int) -> np.ndarray:
    A = 0.5 * (A + A.T)
    w, _ = np.linalg.eigh(A)
    w = np.sort(np.abs(w))[::-1]
    return w[:top_k]


def evaluate_model(
    model,
    temporal: np.ndarray,
    spatial: np.ndarray,
    K_top: int,
    device: torch.device | str = "cuda",
) -> dict:
    """Run a full evaluation given a trained model and the ground-truth synergies.

    The model must expose `.extract_pair_structure()` returning a [J, J] tensor
    after a forward pass on any batch (so the model's internal pair cache is fresh).
    """
    S_gt = ground_truth_affinity(temporal, spatial)                   # [J, J]
    S_model_t = model.extract_pair_structure().detach().cpu().numpy()  # [J, J]
    S_model_t = S_model_t.astype(np.float32)
    S_model_t = S_model_t / (np.linalg.norm(S_model_t) + 1e-8)

    sim = principal_subspace_similarity(S_gt, S_model_t, K_top)
    gt_spec = eigen_spectrum(S_gt, K_top)
    mo_spec = eigen_spectrum(S_model_t, K_top)
    # Correlation between eigenvalue spectra (normalised)
    gt_n = gt_spec / (np.linalg.norm(gt_spec) + 1e-8)
    mo_n = mo_spec / (np.linalg.norm(mo_spec) + 1e-8)
    spec_corr = float(gt_n @ mo_n)

    # Frobenius-normalised inner product between the affinity matrices themselves
    S_gt_n = S_gt / (np.linalg.norm(S_gt) + 1e-8)
    frob_align = float((S_gt_n * S_model_t).sum())

    return dict(
        subspace_sim=sim,
        spectrum_corr=spec_corr,
        frobenius_align=frob_align,
        gt_spectrum=gt_spec.tolist(),
        model_spectrum=mo_spec.tolist(),
    )
