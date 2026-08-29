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
CACHE = "/var/tmp/alphamotion/topo_exp/siglip_names.pt"   # siglip cache; imported by scratchpad/carnival.py
MODEL = "google/siglip-base-patch16-224"

# E27 encoder swap: SigLIP text tower vs Qwen3-Embedding-0.6B, SAME prompt, so the
# encoder is the only variable. Defaults keep every prior run bit-identical.
QWEN_HF_HOME = "/var/tmp/alphamotion/hf_cache"
QWEN_MODEL = "Qwen/Qwen3-Embedding-0.6B"
SEM_DIM = {"siglip": 768, "qwen3": 1024}                  # qwen3 = config.json hidden_size
CACHES = {"siglip": CACHE, "qwen3": "/var/tmp/alphamotion/topo_exp/qwen3_names.pt"}


def joint_prompt(n: str) -> str:
    """The one prompt template, shared by every encoder."""
    return f"the {normalize_joint_name(n)} joint of a body"


def normalize_joint_name(n: str) -> str:
    """'left_hip_pitch_skel' / 'LeftHandThumb1' -> readable words for the text tower."""
    n = re.sub(r"_skel$|_joint$|_dof$", "", n)
    n = re.sub(r"([a-z])([A-Z])", r"\1 \2", n)   # camelCase -> spaced
    n = n.replace("_", " ").strip().lower()
    return n or "root"


@torch.no_grad()
def _encode_siglip(prompts, device):
    """SigLIP text tower, pooled output. Unchanged from every prior run."""
    os.environ["HF_HOME"] = HF_HOME
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=HF_HOME)
    model = AutoModel.from_pretrained(MODEL, cache_dir=HF_HOME).to(device).eval()
    inp = tok(prompts, padding="max_length", max_length=64, return_tensors="pt").to(device)
    feats = model.get_text_features(**inp)
    if not torch.is_tensor(feats):
        po = getattr(feats, "pooler_output", None)
        feats = po if po is not None else feats[0]
    return feats.float().cpu()    # [U,768]


@torch.no_grad()
def _encode_qwen3(prompts, device, batch=16):
    """Qwen3-Embedding-0.6B. It is a CAUSAL LM encoder: the official recipe pools the
    LAST token (never the mean), so we left-pad and read position -1."""
    os.environ["HF_HOME"] = QWEN_HF_HOME
    from transformers import AutoTokenizer, AutoModel
    hub = f"{QWEN_HF_HOME}/hub"
    tok = AutoTokenizer.from_pretrained(QWEN_MODEL, cache_dir=hub, padding_side="left")
    model = AutoModel.from_pretrained(QWEN_MODEL, cache_dir=hub, dtype=torch.float16).to(device).eval()
    out = []
    for i in range(0, len(prompts), batch):
        inp = tok(prompts[i:i + batch], padding=True, truncation=True, max_length=64,
                  return_tensors="pt").to(device)
        assert int(inp["attention_mask"][:, -1].min()) == 1, "left padding broken: last token is a pad"
        h = model(**inp).last_hidden_state          # [B,T,D]
        out.append(h[:, -1].float().cpu())          # last-token pooling
    feats = torch.cat(out, 0)
    assert feats.shape == (len(prompts), SEM_DIM["qwen3"]), f"bad qwen3 shape {tuple(feats.shape)}"
    return feats


@torch.no_grad()
def build_name_embeddings(names, device="cuda", encoder="siglip"):
    """Return {name: [D] tensor} for the given joint names, cached to disk.
    encoder: "siglip" (D=768, every run through E19) or "qwen3" (D=1024, E27 arm Q).
    A full cache hit returns immediately and consumes NO torch RNG -- load-bearing for
    the split reproduction contract in dual_common's module docstring."""
    cache = CACHES[encoder]
    uniq = list(dict.fromkeys(names))
    if os.path.exists(cache):
        d = torch.load(cache, map_location="cpu")
        if set(uniq).issubset(set(d.keys())):
            return d
    prompts = [joint_prompt(n) for n in uniq]       # SAME prompt for both encoders
    feats = _encode_siglip(prompts, device) if encoder == "siglip" else _encode_qwen3(prompts, device)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    out = {n: feats[i] for i, n in enumerate(uniq)}
    torch.save(out, cache)
    return out


def embed_matrix(names, lookup):
    """[J,D] matrix for an ordered joint-name list."""
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
