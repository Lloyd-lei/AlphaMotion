"""Post-hoc probes for a trained DualTranslator (hooked into eval_dual.py).

1. leakage_probe — small MLP classifies source embodiment FROM the quantized codes.
   Val accuracy ~ chance (1/N_emb) => codes are embodiment-invariant. This is the
   E6 longitudinal metric; harsher than code_align_dist (which only checks paired
   distance, not decodability of identity).
2. partial_obs_stability — encode full pose vs masked pose (families from
   masking.py); report integer-code flip rate, z distance (relative to code
   scale), and the E0b acceptance metric: full-body decode error from
   masked-input codes.
"""
from __future__ import annotations
import torch
from dual_common import geom
from common import rot6d_to_matrix, geodesic_deg
from masking import sample_joint_mask, leaf_mask


def _flat_codes(model, data, name, idx, chunk=2048):
    """Quantized grid values for rows idx: [N, S*K*f * n_streams]."""
    outs = []
    for i in range(0, len(idx), chunk):
        sub = idx[i:i + chunk]
        with torch.no_grad():
            _, _, (cp, cr) = model.encode(data[name]["pose"][sub], geom(data, name, len(sub)),
                                          return_codes=True)
        # cp/cr: [S,B,K,f] (ResidualFSQ) or [B,K,f] (FSQ) -> [B, -1]
        def flat(c):
            if c.dim() == 4:
                c = c.permute(1, 0, 2, 3)
            return c.reshape(c.shape[0], -1)
        outs.append(torch.cat([flat(cp), flat(cr)], -1))
    return torch.cat(outs, 0)


def leakage_probe(model, data, names, tr_idx, val_idx, steps=2000, bs=512, n_train=20000, device="cuda"):
    """Train a 2x256 MLP: codes -> embodiment id. Returns val accuracy and chance."""
    tr = tr_idx[:n_train]
    Xtr = torch.cat([_flat_codes(model, data, n, tr) for n in names], 0)
    ytr = torch.cat([torch.full((len(tr),), i, dtype=torch.long, device=device) for i in range(len(names))], 0)
    Xva = torch.cat([_flat_codes(model, data, n, val_idx) for n in names], 0)
    yva = torch.cat([torch.full((len(val_idx),), i, dtype=torch.long, device=device) for i in range(len(names))], 0)
    mlp = torch.nn.Sequential(
        torch.nn.Linear(Xtr.shape[1], 256), torch.nn.GELU(),
        torch.nn.Linear(256, 256), torch.nn.GELU(),
        torch.nn.Linear(256, len(names))).to(device)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    g = torch.Generator(device="cpu").manual_seed(1)
    for _ in range(steps):
        sel = torch.randint(0, Xtr.shape[0], (bs,), generator=g)
        loss = torch.nn.functional.cross_entropy(mlp(Xtr[sel]), ytr[sel])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = float((mlp(Xva).argmax(-1) == yva).float().mean())
    return {"val_acc": round(acc, 4), "chance": round(1.0 / len(names), 4)}


def partial_obs_stability(model, data, names, val_idx, device="cuda", chunk=2048):
    """Per embodiment x mask family: code flip rate, relative z distance, and
    full-body decode error (rot deg / pos cm) from masked-input codes."""
    out = {}
    for name in names:
        spec = data[name]["spec"]
        fams = {"ee": None, "root": None, "rand50": None}
        for fam in fams:
            flips, zds, rots, poss, zsc = [], [], [], [], []
            for i in range(0, len(val_idx), chunk):
                sub = val_idx[i:i + chunk]; B = len(sub)
                if fam == "ee":
                    m = (~leaf_mask(spec).to(device))[None].expand(B, -1).clone()
                elif fam == "root":
                    m = torch.ones(B, spec.J, dtype=torch.bool, device=device)
                else:
                    gen = torch.Generator(device=device).manual_seed(2)
                    m = sample_joint_mask(spec, B, {"rand": 1.0}, device=device, generator=gen)
                import numpy as np
                m[:, int(np.where(spec.parents < 0)[0][0])] = False
                g = geom(data, name, B); x = data[name]["pose"][sub]
                with torch.no_grad():
                    zp_f, zr_f, (cpf, crf) = model.encode(x, g, return_codes=True)
                    zp_m, zr_m, (cpm, crm) = model.encode(x, g, joint_mask=m, return_codes=True)
                    prot, ppos = model.decode(zp_m, zr_m, g)
                gt = x
                flips.append(float((~torch.isclose(torch.cat([cpf.flatten(), crf.flatten()]),
                                                   torch.cat([cpm.flatten(), crm.flatten()]),
                                                   atol=1e-5)).float().mean()))
                zds.append(float(((zp_f - zp_m) ** 2).mean() + ((zr_f - zr_m) ** 2).mean()))
                zsc.append(float((zp_f ** 2).mean() + (zr_f ** 2).mean()))
                rots.append(float(geodesic_deg(rot6d_to_matrix(prot), rot6d_to_matrix(gt[..., :6])).mean()))
                poss.append(float((ppos - gt[..., 6:9]).norm(dim=-1).mean() * data[name]["reach"]))
            k = len(flips)
            out[f"{name[:4]}_{fam}"] = {
                "code_flip": round(sum(flips) / k, 4),
                "z_dist_rel": round(sum(zds) / max(sum(zsc), 1e-9), 4),
                "decode_rot": round(sum(rots) / k, 1),
                "decode_pos_cm": round(sum(poss) / k, 2)}
    return out
