"""§3 variable-length sequence dataset.

Reads the rebuilt full-length AMASS (data/amass_seq/manifest.json + per-seq
.npz) and serves VARIABLE-LENGTH windows for learning self-segmentation.

Design (per the conversation + architecture-discussion.md §3/§6):
  * full continuous sequences on disk; on-the-fly variable-length windows
  * pad + mask per batch (NEVER pre-chop to a fixed length)
  * BABEL frame labels -> per-frame TARGETS (boundary signal + action id);
    boundaries are supervision/eval, NOT model input (§3 = model learns to cut)
  * continuous time provided (time_s, dt) for continuous-time positional enc
  * filter ultra-short seqs; a 'boundary' split keeps only the 4507 seqs that
    actually carry BABEL frame boundaries (the real §3 training material)

Self-test:  python data/amass_seq_dataset.py --viz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except Exception as e:                                    # pragma: no cover
    raise SystemExit(f"torch required: {e}")

REPO = Path("/home/arenalabs/Desktop/\"be water, robot\"")
AMASS_SEQ = REPO / "data" / "amass_seq"
BABEL_LABELS = REPO / "data" / "babel_labels.json"


# --------------------------------------------------------------------------- #
def matrix_to_rot6d(R: np.ndarray) -> np.ndarray:
    """[...,3,3] -> [...,6] = first two columns (matches soma rot6d_to_matrix)."""
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def _cat(fl: dict):
    """Primary BABEL action category (act_cat is the ~250-class taxonomy;
    proc_label is free-text and far too granular for a class target)."""
    ac = fl.get("act_cat")
    if isinstance(ac, list) and ac:
        return ac[0]
    return None


def build_action_vocab(babel_json: Path) -> dict:
    """act_cat[0] -> id. id 0 reserved for <unlabeled>."""
    labels = set()
    if babel_json.exists():
        bl = json.loads(babel_json.read_text())
        for e in bl.values():
            for fl in e.get("frame_labels", []):
                c = _cat(fl)
                if c:
                    labels.add(c)
    vocab = {"<unlabeled>": 0}
    for i, lab in enumerate(sorted(labels), start=1):
        vocab[lab] = i
    return vocab


# --------------------------------------------------------------------------- #
class AmassSeqDataset(Dataset):
    def __init__(
        self,
        manifest: Path = AMASS_SEQ / "manifest.json",
        vocab: dict | None = None,
        require_boundaries: bool = True,   # §3: only seqs with BABEL frame labels
        min_frames: int = 90,              # drop ultra-short (no phases to learn)
        min_win: int = 120,
        max_win: int = 900,
        boundary_sigma_s: float = 0.10,    # soft boundary bump (seconds)
        seed: int = 0,
    ):
        self.man = json.loads(Path(manifest).read_text())
        self.vocab = vocab if vocab is not None else build_action_vocab(BABEL_LABELS)
        self.min_win, self.max_win = min_win, max_win
        self.sigma_s = boundary_sigma_s
        self.rng = np.random.default_rng(seed)

        self.items = [
            m for m in self.man
            if m["T"] >= min_frames
            and (not require_boundaries or m["n_frame_labels"] > 0)
        ]
        if not self.items:
            raise RuntimeError("no sequences after filtering — check manifest")

    def __len__(self):
        return len(self.items)

    # ---- per-frame BABEL targets ---------------------------------------- #
    def _targets(self, frame_labels, T, fps):
        """Per-frame coarse action id (act_cat of the shortest covering
        segment) + boundary = soft bump only where that id CHANGES."""
        action = np.zeros(T, np.int64)                    # 0 = <unlabeled>
        # longest first so the SHORTEST (most specific) segment overwrites
        segs = sorted(frame_labels, key=lambda s: -(s["end_t"] - s["start_t"]))
        for s in segs:
            a = int(round(s["start_t"] * fps))
            b = int(round(s["end_t"] * fps))
            a, b = max(0, min(a, T - 1)), max(0, min(b, T - 1))
            action[a:b + 1] = self.vocab.get(_cat(s), 0)

        change = np.flatnonzero(np.diff(action) != 0) + 1   # true change-points
        boundary = np.zeros(T, np.float32)
        if len(change):
            sigma_f = max(1.0, self.sigma_s * fps)
            idx = np.arange(T)
            for c in change:
                boundary = np.maximum(
                    boundary, np.exp(-0.5 * ((idx - c) / sigma_f) ** 2))
        return boundary, action

    def __getitem__(self, i):
        m = self.items[i]
        d = np.load(m["file"], allow_pickle=True)
        T = int(d["joints_world"].shape[0])
        fps = float(d["fps"])
        fl = list(d["babel_frame_labels"])

        boundary, action = self._targets(fl, T, fps)

        # variable-length window that CONTAINS at least one boundary
        win = int(self.rng.integers(self.min_win, self.max_win + 1))
        win = min(win, T)
        bnd_frames = [int(round(s["start_t"] * fps)) for s in fl] or [T // 2]
        anchor = int(self.rng.choice(bnd_frames))
        lo = int(np.clip(anchor - self.rng.integers(0, win),
                         0, max(0, T - win)))
        hi = lo + win

        pos = d["joints_world"][lo:hi].astype(np.float32)          # [w,77,3]
        body6d = matrix_to_rot6d(d["local_rot_mats"][lo:hi]).astype(np.float32)
        root_pos = d["root_positions"][lo:hi].astype(np.float32)   # [w,3]
        root6d = matrix_to_rot6d(
            d["global_rot_mats"][lo:hi, 0]).astype(np.float32)      # [w,6]
        root_token = np.concatenate([root_pos, root6d], -1)        # [w,9] §6
        time_s = (np.arange(lo, hi) / fps).astype(np.float32)

        return {
            "pos":        torch.from_numpy(pos),
            "body6d":     torch.from_numpy(body6d),
            "root_token": torch.from_numpy(root_token),
            "time_s":     torch.from_numpy(time_s),
            "boundary":   torch.from_numpy(boundary[lo:hi]),
            "action":     torch.from_numpy(action[lo:hi]),
            "length":     win,
            "fps":        fps,
            "name":       m["sample_name"],
        }


# --------------------------------------------------------------------------- #
def collate_varlen(batch):
    """Pad to the batch max length; build a bool pad_mask (True = real)."""
    B = len(batch)
    L = max(b["length"] for b in batch)
    J = batch[0]["pos"].shape[1]

    out = {
        "pos":        torch.zeros(B, L, J, 3),
        "body6d":     torch.zeros(B, L, J, 6),
        "root_token": torch.zeros(B, L, 9),
        "time_s":     torch.zeros(B, L),
        "boundary":   torch.zeros(B, L),
        "action":     torch.zeros(B, L, dtype=torch.long),
        "pad_mask":   torch.zeros(B, L, dtype=torch.bool),
        "length":     torch.tensor([b["length"] for b in batch]),
        "fps":        torch.tensor([b["fps"] for b in batch]),
        "name":       [b["name"] for b in batch],
    }
    for i, b in enumerate(batch):
        n = b["length"]
        for k in ("pos", "body6d", "root_token", "time_s", "boundary", "action"):
            out[k][i, :n] = b[k]
        out["pad_mask"][i, :n] = True
    return out


# --------------------------------------------------------------------------- #
def _selftest(viz: bool):
    vocab = build_action_vocab(BABEL_LABELS)
    ds = AmassSeqDataset(vocab=vocab)
    print(f"vocab size           : {len(vocab)}  (incl <unlabeled>)")
    print(f"sequences after filter: {len(ds)}  (require BABEL boundaries)")

    dl = DataLoader(ds, batch_size=4, shuffle=True,
                    collate_fn=collate_varlen, num_workers=0)
    bt = next(iter(dl))
    print("\nbatch tensor shapes:")
    for k in ("pos", "body6d", "root_token", "time_s",
              "boundary", "action", "pad_mask"):
        print(f"  {k:11s} {tuple(bt[k].shape)}")
    print(f"  lengths      {bt['length'].tolist()}")
    real = bt["pad_mask"]
    print(f"\npad_mask真实帧占比 : {real.float().mean():.2f} "
          f"(变长 -> 有 padding 才对)")
    print(f"boundary 正信号均值: {bt['boundary'][real].mean():.4f} "
          f"(稀疏才对)")
    cov = (bt['action'][real] != 0).float().mean()
    print(f"有动作标签的帧占比 : {cov:.2f}")

    if viz:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        b0 = {k: v[0] for k, v in bt.items()
              if isinstance(v, torch.Tensor)}
        n = int(b0["length"])
        pos = b0["pos"][:n].numpy()
        spd = np.linalg.norm(np.diff(pos, axis=0), axis=-1).mean(-1)
        spd = spd / (spd.max() + 1e-8)
        t = b0["time_s"][:n].numpy()
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(t[1:], spd, lw=0.8, label="joint speed (norm)")
        ax.plot(t, b0["boundary"][:n].numpy(), lw=1.5,
                label="BABEL boundary target")
        ax.set_xlabel("time (s)")
        ax.set_title(f"{bt['name'][0]} — boundaries should land at motion "
                     f"changes")
        ax.legend(fontsize=8)
        out = REPO / "data" / "amass_seq_sanity.png"
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        print(f"\nsanity 图 -> {out} (肉眼核对边界是否落在动作切换处)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--viz", action="store_true")
    _selftest(ap.parse_args().viz)
