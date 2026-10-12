"""Refiner candidates. All take the decoded global rotations R [T,J,3,3], decoded root height, decoder contact logits and the source's
horizontal root trajectory; return q [T,J,3] (spec slots), root_R [T,3,3], root_cm [T,3], per-frame timing and extras."""
import time, numpy as np, torch
from common import *
from alphamotion.engine.constraints import _fit_angles_greedy, lm_fit, fk_from_angles

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

class Latch:
    """Causal foot-plant state per foot.
    planted(t) = decoder contact probability on (hysteresis +-0.1 around its 0.5 decision point)
                 AND foot lowest point within band_cm of the lower foot (band = 2 x projection foot-position noise)
                 AND foot world speed since the previous frame below v_gate(t) = max(source in-contact speed p95, 0.5 x source pelvis speed)
                 (a planted foot moves slower than half the pelvis; the p95 keeps standing clips alive)
    root rule (per frame, before the pose solve): xz shifted so latched feet stay where they landed; y so the lowest planted sole is on the floor,
    or, with nothing planted, so the lowest foot sits at the source's own foot clearance; penetration guard afterwards."""
    def __init__(self, pk, fps, band_cm, clearance_cm, v_src_cm_s, root_speed_cm_s, on=0.6, off=0.4):
        self.pk, self.on_thr, self.off_thr, self.band, self.fps, self.clear = pk, on, off, band_cm, fps, clearance_cm
        self.v_gate = np.maximum(v_src_cm_s, 0.5 * root_speed_cm_s)
        self.on = [False, False]; self.logit_on = [False, False]; self.L = [None, None]; self.preset = [None, None]; self.window_on = None; self.w0 = 0
        self.off = np.zeros(3); self.prev_fpos = None; self.t = 0
    def tick(self, t): self.t = t
    def _feet(self, gR_t, gp_t, root):
        low, fpos = foot_low(self.pk, gR_t[None], (gp_t + torch.as_tensor(root, dtype=gp_t.dtype))[None]); return low[0].numpy().astype(np.float64), fpos[0].numpy().astype(np.float64)
    def logits(self, logit_t):
        for i in range(2):
            p = sigmoid(max(float(logit_t[j]) for j in self.pk.contact_nodes[i]))
            if self.logit_on[i] and p < self.off_thr: self.logit_on[i] = False
            elif not self.logit_on[i] and p > self.on_thr: self.logit_on[i] = True
    def step_root(self, logit_t, gR_t, gp_t, prov_root):
        """decide planted feet from the current (warm-start / R0) pose and return this frame's root"""
        self.logits(logit_t); root = prov_root + self.off; low, fpos = self._feet(gR_t, gp_t, root)
        for i in range(2):
            on_i = bool(self.window_on[self.t - self.w0, i]) if self.window_on is not None else self.logit_on[i]
            ok = on_i and (low[i] - low.min() < self.band)
            if self.prev_fpos is not None: ok = ok and (np.linalg.norm(fpos[i, [0, 2]] - self.prev_fpos[i, [0, 2]]) * self.fps < self.v_gate[self.t])
            self.on[i] = bool(ok)
        planted = [i for i in range(2) if self.on[i]]
        dy = -min(low[i] for i in planted) if planted else float(self.clear[self.t].min()) - low.min()
        root[1] += dy; self.off[1] += dy; low += dy; fpos[:, 1] += dy
        shifts = [self.L[i][0][[0, 2]] - fpos[i, [0, 2]] for i in planted if self.L[i] is not None]
        if shifts: sh = np.mean(shifts, 0); root[[0, 2]] += sh; self.off[[0, 2]] += sh; fpos[:, [0, 2]] += sh
        return root
    def bookkeep(self, gR_t, gp_t, root):
        """after the pose solve: latch new touchdowns at their solved world pose, release lifted feet, penetration guard; returns final root"""
        low, fpos = self._feet(gR_t, gp_t, root); m = low.min()
        if m < 0: root = root.copy(); root[1] -= m; self.off[1] -= m; fpos[:, 1] -= m
        for i in range(2):
            if not self.on[i]: self.L[i] = None; continue
            if self.L[i] is None:
                pre = self.preset[i]
                self.L[i] = (pre[0].copy(), pre[1].copy()) if (pre is not None and pre[2] <= self.t < pre[3]) else (fpos[i].copy(), gR_t[self.pk.feet[i]].numpy().copy())
            if self.preset[i] is not None and self.t >= self.preset[i][3]: self.preset[i] = None
        self.prev_fpos = fpos.copy(); return root

def make_latch(pk, dec, src, base):
    w = measured_weights(pk, dec, src, base); prov = src_root_xz(src, pk)
    rs = np.linalg.norm(np.diff(prov[:, [0, 2]], axis=0), axis=-1) * src['fps']; rs = np.concatenate([rs[:1], rs])
    return Latch(pk, src['fps'], band_cm(pk, dec, src, base), clearance(pk, src), src['v_thr_cm_s'], rs)

def r0(pk, dec, src):
    """today's path: greedy + LM25 batched; decoded root height; no correction"""
    T = len(dec['R']); t = time.perf_counter(); _, _, q = project(matrix_to_rot6d(dec['R']), pk.spec, pk.dof, rest=pk.rest); ms = 1000 * (time.perf_counter() - t)
    q = q.numpy(); root_R = dec['R'][:, pk.root].numpy(); root = src_root_xz(src, pk); root[:, 1] = dec['root_h_cm']
    return dict(q=q, root_R=root_R, root_cm=root, ms_per_frame=ms / T, mode='batched, non-causal only through the batch (each frame independent)', extra={})

def _unused_v_thr(src):
    """source's own foot speed while it says the foot is in contact: p95 (cm/s) = what the source calls "planted" """
    P = src['root_cm']; st = src['stance']
    fh = src.get('foot_xz_cm')
    if fh is None: return float('nan')
    sp = np.linalg.norm(np.diff(fh, axis=0), axis=-1) * src['fps']; m = st[1:] & st[:-1]
    return float(np.percentile(sp[m], 95)) if m.any() else float('nan')

def r1(pk, dec, src, base):
    """geometric snap, causal, no optimization: R0 pose + root rule (planted sole on the floor, xz foot latch)"""
    q, root_R = base['q'], base['root_R']; T = len(q); gR, gp = fk_from_angles(torch.as_tensor(q), pk.spec, pk.dof, rest=pk.rest, root_R=torch.as_tensor(root_R))
    prov = src_root_xz(src, pk); prov[:, 1] = dec['root_h_cm']; latch = make_latch(pk, dec, src, base); root = np.zeros((T, 3)); xs = []
    for t in range(T):
        s = time.perf_counter(); latch.tick(t); rt = latch.step_root(dec['con'][t], gR[t], gp[t], prov[t]); root[t] = latch.bookkeep(gR[t], gp[t], rt); xs.append(time.perf_counter() - s)
    drift = root[:, [0, 2]] - prov[:, [0, 2]]
    return dict(q=q.copy(), root_R=root_R.copy(), root_cm=root, ms_per_frame=1000 * float(np.median(xs)) + base['ms_per_frame'], mode='causal (per frame, on top of R0 projection)',
                extra=dict(band_cm=latch.band, root_xz_drift_final_cm=float(np.linalg.norm(drift[-1])), root_xz_drift_max_cm=float(np.linalg.norm(drift, axis=1).max()), root_y_offset_mean_cm=float((root[:, 1] - prov[:, 1]).mean())))

def band_cm(pk, dec, src, base):
    """2 x mean foot-position deviation between the projected pose and the decoded (unprojected) FK: the codec's own foot placement noise"""
    _, gp = fk_from_angles(torch.as_tensor(base['q']), pk.spec, pk.dof, rest=pk.rest, root_R=torch.as_tensor(base['root_R']))
    return 2.0 * float((gp[:, pk.feet] - dec['gp_dec'][:, pk.feet]).norm(dim=-1).mean())
def clearance(pk, src):
    """source lowest-foot-joint height above the source's own planted level (per side: the clip's minimum), scaled to the robot's leg"""
    h = np.asarray(src['foot_h_cm'], np.float64); return np.maximum(h - h.min(0, keepdims=True), 0.0) * (pk.leg_cm / src['leg_cm'])

def measured_weights(pk, dec, src, base):
    """sigma_r: decoder-vs-projection rotation noise (rad) from R0; sigma_a: joint-angle second-difference std (rad) of the source's own motion
    (its native joint angles when the source is a robot, else R0's)."""
    gR, _ = fk_from_angles(torch.as_tensor(base['q']), pk.spec, pk.dof, rest=pk.rest, root_R=torch.as_tensor(base['root_R']))
    sig_r = float(np.radians(geodesic_deg(gR, dec['R']).mean()))
    qa = src['q_native'] if src.get('q_native') is not None and src['q_native'].shape[1] == pk.spec.J else base['q']
    lim = pk.limited.bool().numpy(); acc = np.diff(qa, n=2, axis=0)[:, lim]; sig_a = float(acc.std() + 1e-9)
    J = pk.spec.J
    # w_ref: the AR(2) prior carries one joint's share of the decoder rotation noise (ablation: the full-body share overshoots and triples slip on H2)
    return dict(sigma_r_rad=sig_r, sigma_a_rad=sig_a, w_ref=2 * sig_r ** 2 / (J * sig_a ** 2), w_pin_pos=J * 2 * sig_r ** 2 / 0.5 ** 2, w_pin_rot=float(J))

def _pins(pk, latch, R_t, prov_root, w, hand_targets=None):
    tR = R_t.clone(); tp = torch.zeros(pk.spec.J, 3); wpos = torch.zeros(pk.spec.J); wrot = torch.ones(pk.spec.J)
    for i in range(2):
        if latch.on[i] and latch.L[i] is not None:
            j = pk.feet[i]; tp[j] = torch.as_tensor(latch.L[i][0] - prov_root, dtype=torch.float32); wpos[j] = w['w_pin_pos']; tR[j] = torch.as_tensor(latch.L[i][1]); wrot[j] = w['w_pin_rot']
    if hand_targets is not None:
        for i, j in enumerate(pk.hands): tp[j] = hand_targets[i]; wpos[j] = w['w_pin_pos']
    return tR, tp, wpos, wrot

def r2(pk, dec, src, base, iters=2, hand_pin=False):
    """recursive LM, causal: warm start from the previous frame, AR(2) prior, latched-foot pins, root from latched feet"""
    R = dec['R']; T = len(R); w = measured_weights(pk, dec, src, base); prov = src_root_xz(src, pk); prov[:, 1] = dec['root_h_cm']
    latch = make_latch(pk, dec, src, base); q = np.zeros_like(base['q']); root = np.zeros((T, 3)); xs = []; hand_err = []
    q_prev = torch.as_tensor(base['q'][:1]); q_prev2 = q_prev.clone(); lo, hi = pk.lo, pk.hi
    for t in range(T):
        s = time.perf_counter(); latch.tick(t); gRw, gpw = fk_from_angles(q_prev, pk.spec, pk.dof, rest=pk.rest, root_R=R[t, pk.root][None]); rt = latch.step_root(dec['con'][t], gRw[0], gpw[0], prov[t])
        q_ref = torch.where(pk.limited.bool()[None], (2 * q_prev - q_prev2).clamp(lo[None], hi[None]), 2 * q_prev - q_prev2)
        ht = dec['gp_dec'][t, pk.hands] if hand_pin else None
        tR, tp, wpos, wrot = _pins(pk, latch, R[t], rt, w, ht)
        qt = lm_fit(tR[None], pk.spec, pk.dof, q_prev, rest=pk.rest, root_R=R[t, pk.root][None], target_p=tp[None] if wpos.any() else None, w_rot=wrot, w_pos=wpos if wpos.any() else None, q_ref=q_ref, w_ref=w['w_ref'], iters=iters)
        gR, gp = fk_from_angles(qt, pk.spec, pk.dof, rest=pk.rest, root_R=R[t, pk.root][None]); root[t] = latch.bookkeep(gR[0], gp[0], rt); q[t] = qt[0].numpy()
        if hand_pin: hand_err.append(float((gp[0, pk.hands] - dec['gp_dec'][t, pk.hands]).norm(dim=-1).mean()))
        q_prev2, q_prev = q_prev, qt; xs.append(time.perf_counter() - s)
    ex = dict(weights=w, root_xz_drift_final_cm=float(np.linalg.norm(root[-1, [0, 2]] - prov[-1, [0, 2]])), root_xz_drift_max_cm=float(np.linalg.norm(root[:, [0, 2]] - prov[:, [0, 2]], axis=1).max()))
    if hand_pin: ex['hand_ee_err_cm_mean'] = float(np.mean(hand_err)); ex['hand_ee_err_cm_p95'] = float(np.percentile(hand_err, 95))
    return dict(q=q, root_R=R[:, pk.root].numpy(), root_cm=root, ms_per_frame=1000 * float(np.median(xs)), mode=f'causal, {iters} LM iteration(s) per frame' + (' + hand pins' if hand_pin else ''), extra=ex)

def r3(pk, dec, src, base, iters=3):
    """receding-horizon window LM: ARDY's own horizon (52 frames at 25 fps, 40 at 20, 60 at 30 = 2.0-2.1 s); plant on/off labels are decided with
    lookahead over the whole window (hysteresis run ahead on a provisional greedy pose), so a plant starts on its true first frame; feet pinned at the solved
    touchdown pose; AR(2) prior; 3 LM iterations; windows chained through the last two frames and the latches"""
    R = dec['R']; T = len(R); fps = src['fps']; W = {25.0: 52, 20.0: 40}.get(float(fps), int(round(2 * fps))); w = measured_weights(pk, dec, src, base)
    prov = src_root_xz(src, pk); prov[:, 1] = dec['root_h_cm']; latch = make_latch(pk, dec, src, base); q = np.zeros_like(base['q']); root = np.zeros((T, 3)); win_ms = []
    q_prev = torch.as_tensor(base['q'][:1]); q_prev2 = q_prev.clone(); lo, hi = pk.lo, pk.hi
    for w0 in range(0, T, W):
        s = time.perf_counter(); w1 = min(T, w0 + W)
        # lookahead inside the window: provisional pose (greedy, batched), provisional roots, stance labels, plant segments -> median poses
        qg, _ = _fit_angles_greedy(R[w0:w1], pk.spec, pk.dof, 1.0, rest=pk.rest, clamp=True, iters=32)
        gRp, gpp = fk_from_angles(qg, pk.spec, pk.dof, rest=pk.rest, root_R=R[w0:w1, pk.root]); pr = prov[w0:w1] + latch.off
        low, fpos = foot_low(pk, gRp, gpp + torch.as_tensor(pr, dtype=torch.float32)[:, None]); fpos = fpos.numpy()
        st = np.zeros((w1 - w0, 2), bool); on = list(latch.logit_on); lowp = low.numpy()
        for k in range(w1 - w0):
            for i in range(2):
                p = sigmoid(max(float(dec['con'][w0 + k][j]) for j in pk.contact_nodes[i]))
                if on[i] and p < latch.off_thr: on[i] = False
                elif not on[i] and p > latch.on_thr: on[i] = True
                sp = np.linalg.norm(fpos[k, i, [0, 2]] - fpos[k - 1, i, [0, 2]]) * fps if k > 0 else 0.0
                st[k, i] = on[i] and (lowp[k, i] - lowp[k].min() < latch.band) and sp < latch.v_gate[w0 + k]
        latch.window_on = st; latch.w0 = w0                           # lookahead: plant labels for the whole window (band/speed gates still apply per frame)
        for t in range(w0, w1):
            latch.tick(t); gRw, gpw = fk_from_angles(q_prev, pk.spec, pk.dof, rest=pk.rest, root_R=R[t, pk.root][None]); rt = latch.step_root(dec['con'][t], gRw[0], gpw[0], prov[t])
            q_ref = torch.where(pk.limited.bool()[None], (2 * q_prev - q_prev2).clamp(lo[None], hi[None]), 2 * q_prev - q_prev2)
            tR, tp, wpos, wrot = _pins(pk, latch, R[t], rt, w)
            qt = lm_fit(tR[None], pk.spec, pk.dof, q_prev, rest=pk.rest, root_R=R[t, pk.root][None], target_p=tp[None] if wpos.any() else None, w_rot=wrot, w_pos=wpos if wpos.any() else None, q_ref=q_ref, w_ref=w['w_ref'], iters=iters)
            gR, gp = fk_from_angles(qt, pk.spec, pk.dof, rest=pk.rest, root_R=R[t, pk.root][None]); root[t] = latch.bookkeep(gR[0], gp[0], rt); q[t] = qt[0].numpy()
            q_prev2, q_prev = q_prev, qt
        win_ms.append(1000 * (time.perf_counter() - s))
    return dict(q=q, root_R=R[:, pk.root].numpy(), root_cm=root, ms_per_frame=float(np.sum(win_ms) / T), mode=f'receding horizon, window {W} frames ({W / fps:.2f} s), {iters} LM iterations',
                extra=dict(weights=w, window_frames=W, window_ms_median=float(np.median(win_ms)), root_xz_drift_final_cm=float(np.linalg.norm(root[-1, [0, 2]] - prov[-1, [0, 2]])), root_xz_drift_max_cm=float(np.linalg.norm(root[:, [0, 2]] - prov[:, [0, 2]], axis=1).max())))

def eric(pk, dec, src, base):
    """Eric's offline support-aware contact residual (100 Adam steps over the whole clip), starting from R0's projection; his root-rotation smoothing kept"""
    from scipy.ndimage import gaussian_filter1d
    from soma_contact_refine import refine
    T = len(base['q']); fps = src['fps']; root = src_root_xz(src, pk); root[:, 1] = dec['root_h_cm']
    rr = gaussian_filter1d(base['root_R'].astype(np.float64), sigma=fps * .04, axis=0, mode='nearest'); u, _, v = np.linalg.svd(rr); corr = np.tile(np.eye(3), (T, 1, 1)); corr[:, 2, 2] = np.linalg.det(u @ v); rr = u @ corr @ v
    support = src['stance'].astype(bool); mask = np.concatenate([support, np.zeros((T, 2), bool)], 1)
    dt, rs = pk.dof.cuda().double(), pk.rest.cuda().double(); t = time.perf_counter()
    q, rot, world, root_t, info = refine(torch.as_tensor(base['q']).cuda().double(), root.copy(), torch.as_tensor(rr).cuda(), pk.spec, dt, rs, pk.sole, support, fps, 100,
                                         source_clearance=src['foot_h_cm'].astype(np.float64), config={}, support_surface=pk.support, support_mask=mask)
    torch.cuda.synchronize(); ms = 1000 * (time.perf_counter() - t)
    return dict(q=q.cpu().float().numpy(), root_R=rr.astype(np.float32), root_cm=np.asarray(root_t, np.float64), ms_per_frame=ms / T, mode='offline, whole clip, 100 Adam steps (his first stage; arm stage not run)',
                extra=dict(seconds=info['seconds'], max_root_adjustment_cm=info['max_root_adjustment_cm'], joint_adjustment_p95_deg=info['joint_adjustment_p95_deg']))
