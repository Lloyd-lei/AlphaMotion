"""JOB 1b/2c: does E19 encode the pelvis-rooted soma_body cache SANELY?

Self-reconstruction (encode soma_body -> decode soma_body) with E19_K128_0805,
scored exactly like the run's own eval (dual_common.run_dir): mean geodesic deg
on global rotations + decoder position error (reach units -> cm).

Reference points:
  * E19 val human_smpl->human_smpl: 8.7 deg / 3.8 cm   (trained embodiment+data)
  * old floor-rooted soma77 -> g1: 118.9 deg           (the defect being fixed)
  * expectation for a zero-shot human-like body: ~10-16 deg.

Also runs a human_smpl control on the same checkpoint so the two numbers are
directly comparable, and a per-R7-family breakdown over the first batch.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE_ROOT, SkeletonSpec, rot6d_to_matrix, geodesic_deg  # noqa: E402
from train_ae import static_cond                               # noqa: E402
from train_global import build_global                          # noqa: E402
from dual_common import build_model, geom                      # noqa: E402


def load_spec(cache, name):
    d = np.load(f"{cache}/{name}_spec.npz", allow_pickle=True)
    return SkeletonSpec(str(d["name"]), d["parents"], d["rest_offsets"],
                        [str(x) for x in d["joint_names"]])

CACHE = f"{CACHE_ROOT}/cache_bones_body"
XWBC = f"{CACHE_ROOT}/cache_xwbc_lafan"
RUN = f"{CACHE_ROOT}/runs/E19_K128_0805"


def entry(cache, name, spec, device, lut):
    from semantics import embed_matrix
    from dof_head import build_kin
    e = {"spec": spec, "st": static_cond(spec, device),
         "sem": embed_matrix(spec.joint_names, lut).to(device),
         "dof": torch.as_tensor(np.load(f"{cache}/{name}_dof.npz")["dof"],
                                dtype=torch.float32, device=device),
         "kin": build_kin(cache, name, spec, device=device)}
    return e


@torch.no_grad()
def self_recon(net, data, name, pose, bs=64):
    """-> per-row rot deg / pos (reach units), per-joint rot deg, per-row swing deg.
    rot/pos exactly like dual_common.run_dir; swing = bone-direction error only
    (position-observable part; rot - swing ~ twist residual)."""
    st = data[name]["st"]
    a0 = torch.nn.functional.normalize(st["rest_off"], dim=-1)[None]
    rot, pos, pj, sw = [], [], [], []
    for i in range(0, len(pose), bs):
        x = pose[i:i + bs]; B = len(x)
        zp, zr = net.encode(x, geom(data, name, B))
        prot, ppos = net.decode(zp, zr, geom(data, name, B))
        Rp, Rt = rot6d_to_matrix(prot), rot6d_to_matrix(x[..., :6])
        g = geodesic_deg(Rp, Rt)
        vp = (Rp @ a0.expand(B, -1, -1).unsqueeze(-1)).squeeze(-1)
        vg = (Rt @ a0.expand(B, -1, -1).unsqueeze(-1)).squeeze(-1)
        s = torch.rad2deg(torch.arccos((vp * vg).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)))
        rot.append(g.mean(-1).cpu()); pj.append(g.mean(0).cpu())
        sw.append(s[:, 1:].mean(-1).cpu())            # root has no bone dir
        pos.append((ppos - x[..., 6:9]).norm(dim=-1).mean(-1).cpu())
    return (torch.cat(rot), torch.cat(pos),
            torch.stack(pj).mean(0), torch.cat(sw))


def stats(v):
    v = np.asarray(v)
    return {"mean": round(float(v.mean()), 2), "median": round(float(np.median(v)), 2),
            "p90": round(float(np.percentile(v, 90)), 2), "max": round(float(v.max()), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--control_rows", type=int, default=2000)
    args = ap.parse_args()
    dev = args.device

    man = json.load(open(f"{CACHE}/manifest.json"))
    spec_b = load_spec(CACHE, "soma_body")
    spec_h = load_spec(XWBC, "human_smpl")
    from semantics import build_name_embeddings
    lut = build_name_embeddings(list(spec_b.joint_names) + list(spec_h.joint_names),
                                device=dev)
    data = {"soma_body": entry(CACHE, "soma_body", spec_b, dev, lut),
            "human_smpl": entry(XWBC, "human_smpl", spec_h, dev, lut)}

    net = build_model(json.load(open(f"{RUN}/config.json")), dev)
    net.load_state_dict(torch.load(f"{RUN}/model.pt", map_location=dev), strict=False)
    net.eval()

    # ---- soma_body: every row of the first batch ----
    g9 = torch.load(f"{CACHE}/soma_body_global9d.pt")            # pos in cm
    reach = man["reach_cm"]
    pose = torch.cat([g9[..., :6], g9[..., 6:9] / reach], -1).to(dev)
    rot, pos, pj, sw = self_recon(net, data, "soma_body", pose, args.bs)
    pos_cm = pos * reach
    print(f"[soma_body self-recon] rows={len(rot)} reach={reach:.1f} cm")
    print("  rot deg:", stats(rot), "\n  pos cm :", stats(pos_cm))
    print(f"  swing (bone-dir) deg: {sw.mean():.1f}  -> twist residual ~ "
          f"{rot.mean() - sw.mean():.1f} deg")
    order = torch.argsort(pj, descending=True)
    names_b = list(spec_b.joint_names)
    print("  worst joints:", [(names_b[i], round(float(pj[i]), 1)) for i in order[:6]])
    print("  best joints :", [(names_b[i], round(float(pj[i]), 1)) for i in order[-4:]])

    # per-family breakdown
    seqs = json.load(open(f"{CACHE}/sequences.json"))
    fam_rows = {}
    for s in seqs:
        fam_rows.setdefault(s["family"], []).append((s["start"], s["start"] + s["len"]))
    per_fam = {}
    for f, spans in sorted(fam_rows.items()):
        m = torch.zeros(len(rot), dtype=torch.bool)
        for a, b in spans:
            m[a:b] = True
        per_fam[f] = {"rows": int(m.sum()), "rot_deg": round(float(rot[m].mean()), 2),
                      "pos_cm": round(float(pos_cm[m].mean()), 2)}
        print(f"  {f:6s} rows={int(m.sum()):6d}  rot {rot[m].mean():5.1f} deg  "
              f"pos {pos_cm[m].mean():5.2f} cm")

    # ---- human_smpl control (same checkpoint, same metric) ----
    l6h = torch.load(f"{XWBC}/human_smpl_paired_local6d.pt")
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(l6h), generator=g)[:args.control_rows]
    gr6, gp, reach_h = build_global(l6h[idx], spec_h, "cpu")
    pose_h = torch.cat([gr6, gp / reach_h], -1).to(dev)
    rot_h, pos_h, _, sw_h = self_recon(net, data, "human_smpl", pose_h, args.bs)
    print(f"[human_smpl control]  rows={len(rot_h)} reach={reach_h:.1f} cm")
    print("  rot deg:", stats(rot_h), "\n  pos cm :", stats(pos_h * reach_h))

    out = {"run": RUN, "cache": CACHE, "rows": len(rot), "reach_cm": reach,
           "soma_body_self": {"rot_deg": stats(rot), "pos_cm": stats(pos_cm),
                              "swing_deg": round(float(sw.mean()), 2),
                              "per_joint_rot_deg": {n: round(float(v), 1)
                                                    for n, v in zip(names_b, pj)}},
           "per_family": per_fam,
           "human_smpl_control": {"rows": len(rot_h),
                                  "rot_deg": stats(rot_h),
                                  "pos_cm": stats(pos_h * reach_h),
                                  "swing_deg": round(float(sw_h.mean()), 2)}}
    json.dump(out, open(f"{CACHE}/e19_sanity.json", "w"), indent=2)
    print("SAVED", f"{CACHE}/e19_sanity.json")


if __name__ == "__main__":
    main()
