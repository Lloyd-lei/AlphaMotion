"""Qualitative probe: NOVEL free-text prompts -> motion retrieval.

Embeds hand-written prompts (never in training) with frozen Qwen, retrieves
top-3 clips from a mixed amass-val + bones-holdout gallery, prints each hit's
own caption so lloyd can judge semantic match by eye.
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("V5_AMASS", "/var/tmp/alphamotion/topo_exp/cache_xwbc_amass_full")
from align_text import TextCodec, MotionEnc, Pairs, motion_tokens
from bridge_a import load_bridge_a
import v5_common as V
from semantics import _encode_qwen3

TA = "/var/tmp/alphamotion/topo_exp/text_align"
RUN = "/var/tmp/alphamotion/topo_exp/runs/TEXT_ALIGN_v2_0810"
dev = "cuda"
PROMPTS = ["a person kicks with their right leg",
           "someone dances gracefully",
           "a man walks forward slowly",
           "a person punches like a boxer",
           "someone jumps up and down",
           "a person crawls on the ground",
           "someone waves their hand to say hello",
           "a person sits down on a chair"]

torch.manual_seed(0)
ck = torch.load(f"{RUN}/model.pt", map_location=dev)
tc = TextCodec().to(dev); tc.load_state_dict(ck["text"]); tc.eval()
me = MotionEnc().to(dev); me.load_state_dict(ck["motion"]); me.eval()
ma, _ = load_bridge_a(f"{V.RUNS}/BRIDGE_A3_prod_0810", dev)

g = np.random.default_rng(3)
gal, caps = [], []
for split in ("val_amass", "val_bones"):
    pr = Pairs(split)
    byclip = {}
    for p in pr.items:
        byclip.setdefault(p["name"], p)
    for p in list(byclip.values()):
        gal.append(pr.window(p, g)); caps.append((p["src"], p["caption"]))
with torch.no_grad():
    zm = []
    for i in range(0, len(gal), 128):
        zm.append(me(motion_tokens(ma, torch.stack(gal[i:i+128]), dev)))
    zm = torch.cat(zm)
    pe = _encode_qwen3(PROMPTS, dev, batch=8).float()
    zt = tc(pe.to(dev))
    sim = zt @ zm.T
print(f"gallery {len(gal)} clips (amass val + bones holdout)\n")
for i, q in enumerate(PROMPTS):
    top = sim[i].topk(3).indices.tolist()
    print(f'PROMPT: "{q}"')
    for r, j in enumerate(top):
        src, cap = caps[j]
        print(f"  #{r+1} [{src}] {cap[:90]}")
    print()
