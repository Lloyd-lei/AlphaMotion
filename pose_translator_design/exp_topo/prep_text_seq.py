"""Cache the WORD-LEVEL Qwen states for every caption (not the pooled vector).

    prep_text_seq.py

The wrapper has been feeding Stage B one 1024-d vector per caption — the
last-token pool. "a person punches with their RIGHT arm QUICKLY" collapses into
that single vector before the motion model ever sees it. Keep the sequence so
cross-attention can look up individual words.
Stored fp16 with a length array; captions are short so 32 BPE tokens is ample.
"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from semantics import QWEN_HF_HOME, QWEN_MODEL
TA = "/var/tmp/alphamotion/topo_exp/text_align"
MAXL, dev = 32, "cuda"
os.environ["HF_HOME"] = QWEN_HF_HOME
from transformers import AutoTokenizer, AutoModel
hub = f"{QWEN_HF_HOME}/hub"
tok = AutoTokenizer.from_pretrained(QWEN_MODEL, cache_dir=hub, padding_side="right")
model = AutoModel.from_pretrained(QWEN_MODEL, cache_dir=hub,
                                  dtype=torch.float16).to(dev).eval()
caps = sorted(torch.load(f"{TA}/caption_emb.pt", map_location="cpu").keys())
print(f"captions {len(caps)}", flush=True)
S = torch.zeros(len(caps), MAXL, 1024, dtype=torch.float16)
L = torch.zeros(len(caps), dtype=torch.int16)
with torch.no_grad():
    for i in range(0, len(caps), 64):
        b = caps[i:i + 64]
        inp = tok(b, padding="max_length", truncation=True, max_length=MAXL,
                  return_tensors="pt").to(dev)
        h = model(**inp).last_hidden_state              # [b, MAXL, 1024]
        S[i:i + 64] = h.cpu().half()
        L[i:i + 64] = inp["attention_mask"].sum(1).cpu().short()
        if i % 6400 == 0:
            print(f"  {i}/{len(caps)}", flush=True)
torch.save({"seq": S, "len": L, "keys": caps}, f"{TA}/caption_seq.pt")
print(f"-> {TA}/caption_seq.pt  {S.shape}  median len {int(L.median())}"
      f"  {S.numel()*2/1e9:.2f} GB", flush=True)
