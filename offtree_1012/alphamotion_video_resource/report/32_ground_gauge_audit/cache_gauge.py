"""Cache label floor gauge per embodiment per corpus, from the contact npz `gap` field (height above the labeler's floor, m) and h0.
Stance frames = labeler's hard contact on any foot-group joint. Sampled every STRIDE rows."""
import numpy as np, json, sys
STRIDE = 8
CACHES = {'lafan': '/var/tmp/alphamotion/topo_exp/cache_lafan_v3', 'amass': '/var/tmp/alphamotion/topo_exp/cache_amass_lconv_tfix'}
man = json.load(open(f"{CACHES['lafan']}/manifest.json"))['embodiments']
out = {}
for emb in man:
    out[emb] = {}
    for corpus, c in CACHES.items():
        try: z = np.load(f'{c}/{emb}_contact.npz', allow_pickle=True)
        except Exception as e: out[emb][corpus] = {'err': str(e)[:80]}; print(emb, corpus, 'ERR', e); continue
        hdr = json.loads(str(z['header'])); gap = z['gap'][::STRIDE].astype(np.float32); hard = z['contact'][::STRIDE]; trust = z['trust'][::STRIDE] if 'trust' in z.files else np.ones(len(gap), bool)
        groups = hdr['limb_groups']; feet = [i for i, g in enumerate(groups) if g in ('L_foot', 'R_foot')]
        if not feet:  # generic-name caches (h2): use the labeler's j_low / j_floor
            feet = sorted({hdr['j_low'], hdr['j_floor']})
        # lowest foot-group joint height above the labeler's floor (gap already has h0 removed); raw height = gap + h0
        low = gap[:, feet].min(1); raw_low = low + hdr['h0_m']
        stance = hard[:, feet].any(1) & trust
        s = low[stance]; r = raw_low[stance]
        out[emb][corpus] = {'N_sampled': int(len(gap)), 'stance_frac': float(stance.mean()), 'h0_m': hdr['h0_m'], 'sole_drop_cm': hdr.get('sole_drop_cm'), 'feet_joints': [hdr['joint_names'][i] for i in feet],
                            'stance_low_gap_cm': {'p10': float(np.percentile(s, 10) * 100), 'p50': float(np.percentile(s, 50) * 100), 'p90': float(np.percentile(s, 90) * 100)},
                            'stance_low_raw_cm': {'p10': float(np.percentile(r, 10) * 100), 'p50': float(np.percentile(r, 50) * 100), 'p90': float(np.percentile(r, 90) * 100)},
                            'all_low_raw_cm_p50': float(np.percentile(raw_low, 50) * 100), 'pen_frac_raw_lt_-0.5cm': float((raw_low < -0.005).mean()), 'pen_frac_gap_lt_-0.5cm': float((low < -0.005).mean()),
                            'untrusted_frac': float((~trust).mean())}
        print(f"{emb:18s} {corpus:6s} h0 {hdr['h0_m']*100:6.1f}  stance-low raw p50 {out[emb][corpus]['stance_low_raw_cm']['p50']:6.1f} (p10 {out[emb][corpus]['stance_low_raw_cm']['p10']:6.1f}) gap p50 {out[emb][corpus]['stance_low_gap_cm']['p50']:6.1f}  pen(raw) {out[emb][corpus]['pen_frac_raw_lt_-0.5cm']*100:5.1f}%  feet {out[emb][corpus]['feet_joints'][:2]} sole_drop {hdr.get('sole_drop_cm')}", flush=True)
json.dump(out, open('cache_gauge.json', 'w'), indent=1)
