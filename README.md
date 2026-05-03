# AlphaMotion / MotionFormer

> *"Be water, my friend."* — Bruce Lee

A pair-first inductive bias for whole-body humanoid control, ported from
AlphaFold 2's Evoformer to motion modelling.

---

## TL;DR

We move the `[joints × joints]` **pair tensor** from a transient attention
matrix (as in standard Transformers) to a first-class, persistently-updated
backbone state — exactly as Evoformer does for `[residue × residue]` in
AlphaFold 2 — then train it with masked motion modelling on a mixture of
real mocap (AMASS, HumanML3D, 100STYLE) and synthesized motion (Kimodo,
HY Motion).

**Key Stage-1 finding** (2026-05-02):

> The pair tensor learns an **anatomically-meaningful modular structure**
> (spine cluster + 4 limb clusters + L.Hand / R.Hand hub joints) entirely
> from masked-motion-modelling supervision, with no explicit anatomical
> labels. **Triangle attention** is the unique mechanism that selects this
> modular geometry — without it (OPM-only ablation) the pair tensor still
> learns *some* structure but a completely different one (Pearson
> correlation between full vs. opm-only pair-norm heatmaps = **0.077**).
>
> Reconstruction loss is **invariant** to {distributed, modular} pair
> geometry — they reach the same MSE on AMASS held-out (0.122 vs 0.121).
> But downstream tasks needing anatomical structure (cross-morphology
> transfer, interpretable style control, family-aware retrieval) should
> strongly prefer the modular geometry. **Triangle attention's role is
> therefore as a "representation-geometry selector", not a loss optimizer.**

→ Full writeup with sanity-check figures: [`note/triangle_sanity_check.md`](note/triangle_sanity_check.md)

→ Research vision and 4-stage roadmap: [`note/motionformer-research-vision.md`](note/motionformer-research-vision.md)

---

## Repo structure

```
be water, robot/
├── agent.md                                — short syllabus
├── note/
│   ├── motionformer-research-vision.md     — full research vision (1.4k lines)
│   ├── triangle_sanity_check.md            — Stage-1 retrospective: did
│   │                                          triangle attention learn?
│   └── figures/                            — figures referenced by the notes
└── experiment/
    ├── dataset/                             — data prep + ingestion (data not committed)
    │   ├── README.md                        — download / preparation instructions
    │   ├── prepare_amass.py
    │   ├── build_mixed_dataset.py
    │   ├── action_labels.py                 — 12-family regex caption labelling
    │   ├── build_split.py
    │   ├── fix_amass_axis.py / diag_axis.py — coordinate-frame normalisation
    │   └── splits.json                      — train / val_id / val_subject / val_source
    └── stage1/                              — Stage-1 (motion-only, single morphology)
        ├── motionformer.py                  — pair-first backbone (~430 lines)
        ├── baseline.py                       — sequence-first reference Transformer
        ├── data.py                           — datasets + 5 mask modes (incl. kinematic_chain)
        ├── soma_skeleton.py                  — SOMA-77 kinematic tree + FK
        ├── train.py                          — joint pos+rot+root+FK loss training loop
        ├── eval.py / eval_on_amass_heldout.py
        ├── diagnose_stage15.py               — 9-plot diagnostic suite
        ├── style_cluster.py                  — pair-tensor t-SNE by action family
        ├── pair_heatmap_sanity.py            — 4-way control for the triangle finding
        ├── visualize_reconstruction.py       — npz/png/gif comparison viz
        ├── viser_compare.py                  — interactive 3-skeleton browser viewer
        ├── viser_render.py                   — single-npz viser viewer (legacy)
        └── runs/                             — training logs + diag figures (no checkpoints)
```

---

## What's in this repo vs not

**Committed:**
- All source code (Python, shell)
- Research notes (Markdown, 1.5k+ lines)
- Training logs (plaintext) showing every run end-to-end
- Diagnostic figures (training curves, per-joint error, mask-mode comparison,
  style PCA / t-SNE, pair-tensor heatmaps)
- summary.json with quantitative results

**Excluded (see `.gitignore`):**
- Mocap data — AMASS / HumanML3D / 100STYLE / Motion-X / Kimodo each have
  their own license and registration. Download instructions in
  `experiment/dataset/README.md`.
- Trained checkpoints (`.pt`, ~8-13 MB each) — to be hosted via Releases
  or LFS in a follow-up; for now reviewers can re-train from the
  documented commands below.
- Cache `.npz` files (rebuilt from raw data on demand).
- 3D animation `.gif` files (regenerable from `visualize_reconstruction.py`).
- Reference papers PDF directory (`papers-docs/` ~600 MB).

---

## Reproduce Stage-1.5 (axis-fixed mocap mix)

After installing data per `experiment/dataset/README.md`:

```bash
cd experiment/stage1

# Train both ablation variants (~50-60 min each on RTX Pro 6000)
python train.py --mixed-data ../dataset/mixed_soma77.npz \
                --splits ../dataset/splits.json \
                --model opm_only --epochs 30 --batch-size 32 \
                --mask-mode mixed --run-tag mixed_fixed30

python train.py --mixed-data ../dataset/mixed_soma77.npz \
                --splits ../dataset/splits.json \
                --model full --epochs 30 --batch-size 32 \
                --mask-mode mixed --run-tag mixed_fixed30

# Diagnostic suite per checkpoint (training curves, pair heatmap, per-joint
# error, style PCA / t-SNE, source-stratified eval, etc.)
python diagnose_stage15.py --checkpoint runs/full__mixed_fixed30/final.pt --model full
python diagnose_stage15.py --checkpoint runs/opm_only__mixed_fixed30/final.pt --model opm_only

# AMASS held-out evaluation (separate from training subjects)
python eval_on_amass_heldout.py --checkpoint runs/full__mixed_fixed30/final.pt \
       --model full --subsets cmu bmlrub kit hdm05 \
       --out-json runs/full__mixed_fixed30/amass_heldout.json

# 4-way control to test whether the pair-tensor's anatomical structure is
# real (this is the experiment that produced the headline finding).
python pair_heatmap_sanity.py
```

## Interactive viewer

```bash
python viser_compare.py --checkpoint runs/full__mixed_fixed30/final.pt --model full --port 7860
# open http://localhost:7860
```

---

## Stage-1 ablation table (Kimodo-only N=160, 30 epochs, seed=0)

| Variant | params | val_id | **val_kc (joint-failure OOD)** | val ≤ 0.3 | val ≤ 0.2 |
|---|---|---|---|---|---|
| baseline (sequence Transformer) | 2.40 M | 0.419 | 0.484 | never | never |
| axial_only (no pair) | 1.80 M | 0.349 | 0.283 | 55 ep | 120 ep |
| pair_static (frozen pair) | 1.99 M | 0.304 | 0.234 | 55 ep | 95 ep |
| triangle_only (no OPM) | 2.10 M | **0.192** | 0.465 ← OOD trap | 55 ep | 55 ep |
| **opm_only** | 2.12 M | 0.296 | 0.184 | 50 ep | 120 ep |
| **full** | 2.18 M | 0.246 | **0.165** ← best OOD | 55 ep | 120 ep |

→ Two mechanistic conclusions: (1) **OPM** is the OOD-key — gating the
pair tensor by per-sample input is what matters; (2) **triangle alone**
is a trap — refining a static pair leads to ID memorisation but OOD
collapse (val_kc 0.465 ~= baseline). Any future "pair refiner" must be
paired with a "pair update gateway".

## Stage-1.5 (mixed AMASS + Kimodo + HY Motion, axis-corrected)

| | val_id pos | val_subject pos | AMASS heldout pos | Pair-tensor PC1+PC2 var |
|---|---|---|---|---|
| full | 0.154 | 0.168 | **0.122** | **0.538** |
| opm_only | 0.160 | 0.164 | **0.121** | 0.307 |

Reconstruction loss is **invariant** to triangle (Δ = 0.001), but pair
representation geometry is **completely different** (off-diagonal Pearson
*r* between full vs. opm-only pair-norm heatmaps = **0.077**). See
[`note/triangle_sanity_check.md`](note/triangle_sanity_check.md).

---

## Roadmap

- **Stage 1** ✓ (this repo, Apr–May 2026): pair-first ablation on a single
  morphology, anatomical-structure emergence sanity-checked
- **Stage 2** (planned): cross-morphology zero-shot transfer (4-leg dog →
  3-leg) — does modular pair geometry transfer better than distributed?
- **Stage 3**: 40-DoF Octopus Doctor whole-body control — pair-first's
  defensible niche where standard Transformers' attention cost explodes
- **Stage 4**: full brain ↔ cerebellum two-tier architecture with
  cross-attention command interface (compare to π0 / HEX / SONIC)

Detailed plan: [`note/motionformer-research-vision.md`](note/motionformer-research-vision.md) §9.

---

## Open questions / how to contribute

The following are open and would benefit from outside collaboration:

1. **Modular vs distributed pair on cross-morphology** — does the
   anatomical structure that triangle attention introduces actually
   transfer better when the morphology changes? Stage 2 plan.
2. **Other refiners** — triangle attention came from AlphaFold's protein
   distance constraint. For motion the equivalent is **not** kinematic
   chain (we have evidence that whole-body force generation does **not**
   follow kinematic chain — see traditional martial arts and ballet
   examples in `note/triangle_sanity_check.md`). What is the right
   motion-native refiner? SE(3)-equivariant? Contact-graph?
3. **Better mask modes** — random/joint mask MMM trains "fill missing
   sensors" but not "fill missing actuators with downstream" —
   `kinematic_chain` mask is too rigid (4× degradation in Stage 1.5).
   Per-bodypart group masks?
4. **Flow matching head** — the current head is direct regression with
   FK consistency loss. Replacing with a flow-matching head should give
   us controllable generation (Kimodo-editor-style spatial keyframe
   conditioning at inference).
5. **AMASS caption coverage** — 77 % of AMASS samples fall to "other" in
   our 12-family regex labeller because most AMASS subsets only have
   `Subject_XX_F_YY_poses` filenames. Integrating HumanML3D's text
   annotations would unblock proper family-emergence analysis on
   ~4 700 well-labelled samples instead of just 200 Kimodo prompts.

PRs welcome. Issues with reproductions, failures, or counter-evidence
also very welcome — the negative results section in the sanity-check
note (`§4 Open Questions`) is where this is headed honestly.

---

## Citation / inspiration

- **Architecture**: AlphaFold 2 (Jumper et al., Nature 2021), Evoformer block; OpenFold reimplementation as architectural reference (not copied verbatim).
- **Training paradigm**: Masked Motion Modelling — MaskedMimic (NVIDIA SIGGRAPH 2024), Kimodo (NVIDIA 2026); 6D rotation representation (Zhou et al., CVPR 2019).
- **Whole-body control prior art**: HumanPlus (Stanford), OmniH2O (CMU), HOVER (NVIDIA + CMU), HEX (Beijing Humanoid), SONIC (NVIDIA), π0 / π0.5 (Physical Intelligence).
- **Motor primitives biological motivation**: Bizzi & d'Avella's spinal motor synergy literature; Maravita & Iriki (2004) on body-schema extension.
- **Adaptive control comparison**: RMA (Kumar, Fu, Pathak, Malik, 2021); Skild AI's productisation thereof.

---

## License

TBD — likely permissive (Apache 2.0 / MIT) for code, with the explicit note
that the data this repo trains on (AMASS, HumanML3D, etc.) carries its
own non-commercial restrictions and is **not redistributed** here.
