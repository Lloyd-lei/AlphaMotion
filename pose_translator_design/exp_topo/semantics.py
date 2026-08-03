"""SigLIP joint-name semantics: the cross-topology alignment signal (SAMoR-style).
Maps each joint NAME -> a 768-d frozen SigLIP text embedding, so functionally-similar
joints across different skeletons (human hand / robot wrist) land near each other in a
SHARED semantic space. This is the bridge that makes the code embodiment-invariant --
the thing I wrongly dropped earlier.
"""
from __future__ import annotations
import os, re, json
import numpy as np, torch

HF_HOME = "/var/tmp/alphamotion/any_body/hf"
CACHE = "/var/tmp/alphamotion/topo_exp/siglip_names.pt"
MODEL = "google/siglip-base-patch16-224"


def normalize_joint_name(n: str) -> str:
    """'left_hip_pitch_skel' / 'LeftHandThumb1' -> readable words for the text tower."""
    n = re.sub(r"_skel$|_joint$|_dof$", "", n)
    n = re.sub(r"([a-z])([A-Z])", r"\1 \2", n)   # camelCase -> spaced
    n = n.replace("_", " ").strip().lower()
    return n or "root"


@torch.no_grad()
def build_name_embeddings(names, device="cuda"):
    """Return {name: [768] tensor} for the given joint names, cached to disk."""
    uniq = list(dict.fromkeys(names))
    if os.path.exists(CACHE):
        d = torch.load(CACHE, map_location="cpu")
        if set(uniq).issubset(set(d.keys())):
            return d
    os.environ["HF_HOME"] = HF_HOME
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=HF_HOME)
    model = AutoModel.from_pretrained(MODEL, cache_dir=HF_HOME).to(device).eval()
    prompts = [f"the {normalize_joint_name(n)} joint of a body" for n in uniq]
    inp = tok(prompts, padding="max_length", max_length=64, return_tensors="pt").to(device)
    feats = model.get_text_features(**inp)
    if not torch.is_tensor(feats):
        po = getattr(feats, "pooler_output", None)
        feats = po if po is not None else feats[0]
    feats = feats.float().cpu()   # [U,768]
    feats = torch.nn.functional.normalize(feats, dim=-1)
    out = {n: feats[i] for i, n in enumerate(uniq)}
    torch.save(out, CACHE)
    return out


def embed_matrix(names, lookup):
    """[J,768] matrix for an ordered joint-name list."""
    return torch.stack([lookup[n] for n in names], 0)


if __name__ == "__main__":
    import numpy as np
    c = "/var/tmp/alphamotion/topo_exp/cache_paired"
    alln = []
    for f in ["soma77_spec.npz", "g1_30_spec.npz"]:
        alln += list(np.load(f"{c}/{f}", allow_pickle=True)["joint_names"])
    lut = build_name_embeddings(alln)
    print("cached", len(lut), "joint-name SigLIP embeddings, dim", next(iter(lut.values())).shape)
    for n in ["Hips", "LeftHand", "pelvis_skel", "left_wrist_roll_skel"]:
        if n in lut:
            print(f"  {n} -> {normalize_joint_name(n)}")
