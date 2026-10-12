"""UniML3D reload-with-ground prototype.

Export NPZ (stage 1-3, source rig, Y-up, per-frame local TRS) + source GLB mesh/skin
(raw mirrors) -> LBS surface point cloud per frame -> ground plane -> strict contact + slip.

No third-party glTF lib: minimal GLB reader with numpy.
"""
import json, struct, sys, os
import numpy as np

CT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}


def read_glb(path):
    b = open(path, 'rb').read()
    assert b[:4] == b'glTF'
    off = 12; js = None; binc = None
    while off < len(b):
        ln, ty = struct.unpack('<II', b[off:off + 8]); ch = b[off + 8: off + 8 + ln]
        if ty == 0x4E4F534A: js = json.loads(ch)
        elif ty == 0x004E4942: binc = ch
        off += 8 + ln
    return js, binc


def accessor(js, binc, i):
    a = js['accessors'][i]; n = NC[a['type']]; dt = CT[a['componentType']]
    cnt = a['count']
    if 'bufferView' not in a:
        return np.zeros((cnt, n), dt)
    bv = js['bufferViews'][a['bufferView']]
    start = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
    item = np.dtype(dt).itemsize * n
    stride = bv.get('byteStride', item)
    if stride == item:
        arr = np.frombuffer(binc, dt, cnt * n, start).reshape(cnt, n)
    else:
        raw = np.frombuffer(binc, np.uint8, stride * (cnt - 1) + item, start)
        arr = np.stack([np.frombuffer(raw[k * stride:k * stride + item].tobytes(), dt) for k in range(cnt)])
    arr = arr.astype(np.float64) if dt == np.float32 else arr
    if a.get('normalized') and dt in (np.uint8, np.uint16):
        arr = arr.astype(np.float64) / np.iinfo(dt).max
    if 'sparse' in a:
        raise NotImplementedError('sparse accessor')
    return arr


def quat_wxyz_to_mat(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                  2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                  2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)
    return R.reshape(q.shape[:-1] + (3, 3))


def node_local(n):
    if 'matrix' in n:
        return np.array(n['matrix'], float).reshape(4, 4).T
    M = np.eye(4)
    t = n.get('translation', [0, 0, 0]); r = n.get('rotation', [0, 0, 0, 1]); s = n.get('scale', [1, 1, 1])
    M[:3, :3] = quat_wxyz_to_mat(np.array([r[3], r[0], r[1], r[2]])) * np.array(s)[None]
    M[:3, 3] = t
    return M


def glb_skinned_mesh(path):
    """Return bind vertices (mesh space), faces, per-vertex joint weights over skin joint names."""
    js, binc = read_glb(path)
    nodes = js['nodes']
    parent = {c: i for i, n in enumerate(nodes) for c in n.get('children', [])}

    def world(i):
        M = node_local(nodes[i])
        while i in parent:
            i = parent[i]; M = node_local(nodes[i]) @ M
        return M

    V, F, W, vofs = [], [], [], 0
    skin_names = None; skin_ibm = None; skin_joints = None; mesh_worlds = []
    for ni, n in enumerate(nodes):
        if 'mesh' not in n or 'skin' not in n:
            continue
        sk = js['skins'][n['skin']]
        names = [nodes[j].get('name', f'node{j}') for j in sk['joints']]
        ibm = accessor(js, binc, sk['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1) \
            if 'inverseBindMatrices' in sk else np.tile(np.eye(4), (len(names), 1, 1))
        if skin_names is None:
            skin_names, skin_ibm, skin_joints = names, ibm, sk['joints']
        jmap = [skin_names.index(nm) if nm in skin_names else -1 for nm in names]
        if -1 in jmap:  # second skin with other joints: extend
            for k, nm in enumerate(names):
                if nm not in skin_names:
                    skin_names.append(nm); skin_ibm = np.concatenate([skin_ibm, ibm[k:k + 1]])
            jmap = [skin_names.index(nm) for nm in names]
        jmap = np.array(jmap)
        for p in js['meshes'][n['mesh']]['primitives']:
            if p.get('mode', 4) != 4 or 'JOINTS_0' not in p['attributes']:
                continue
            pos = accessor(js, binc, p['attributes']['POSITION'])
            if 'indices' in p:
                idx = accessor(js, binc, p['indices']).reshape(-1, 3).astype(np.int64)
            else:
                idx = np.arange(len(pos)).reshape(-1, 3)
            w = np.zeros((len(pos), len(skin_names) + 64))
            for s in range(4):
                if f'JOINTS_{s}' not in p['attributes']:
                    break
                J = accessor(js, binc, p['attributes'][f'JOINTS_{s}']).astype(np.int64)
                Wt = accessor(js, binc, p['attributes'][f'WEIGHTS_{s}'])
                for c in range(4):
                    np.add.at(w, (np.arange(len(pos)), jmap[J[:, c]]), Wt[:, c])
            V.append(pos); F.append(idx + vofs); W.append(w); vofs += len(pos)
            mesh_worlds.append(world(ni))
    nJ = len(skin_names)
    W = np.concatenate([w[:, :nJ] for w in W])
    W = W / np.clip(W.sum(1, keepdims=True), 1e-9, None)
    name_parent = {}
    for i, n in enumerate(nodes):
        if i in parent:
            name_parent[n.get('name', f'node{i}')] = nodes[parent[i]].get('name', f'node{parent[i]}')
    return dict(V=np.concatenate(V), F=np.concatenate(F), W=W, names=skin_names, ibm=skin_ibm, name_parent=name_parent,
                joint_world=np.stack([world(j) for j in skin_joints] + [np.eye(4)] * (nJ - len(skin_joints))))


def export_fk(d, trim=0):
    """Global 4x4 per frame from export local TRS (w-first quats). Returns (T,J,4,4) and rest (J,4,4)."""
    par = d['parents']

    def fk(lp, lr):
        L = np.zeros(lp.shape[:-1] + (4, 4)); L[..., :3, :3] = quat_wxyz_to_mat(lr); L[..., :3, 3] = lp; L[..., 3, 3] = 1
        G = np.zeros_like(L)
        for j, p in enumerate(par):
            G[..., j, :, :] = L[..., j, :, :] if p < 0 else G[..., p, :, :] @ L[..., j, :, :]
        return G
    return fk(d['anim_local_pos'][trim:], d['anim_local_rot'][trim:]), fk(d['rest_local_pos'], d['rest_local_rot'])


def umeyama(src, dst):
    ms, md = src.mean(0), dst.mean(0); s0, d0 = src - ms, dst - md
    U, S, Vt = np.linalg.svd(d0.T @ s0 / len(src)); D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0: D[2, 2] = -1
    R = U @ D @ Vt; sc = np.trace(np.diag(S) @ D) / (s0 ** 2).sum(1).mean()
    return sc, R, md - sc * R @ ms


def sample_surface(V, F, W, n_cand=40000, n=4096, seed=0):
    rng = np.random.default_rng(seed)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    fi = rng.choice(len(F), n_cand, p=area / area.sum())
    u, v = rng.random(n_cand), rng.random(n_cand); m = u + v > 1; u[m], v[m] = 1 - u[m], 1 - v[m]
    bc = np.stack([1 - u - v, u, v], 1)
    P = (bc[:, :, None] * V[F[fi]]).sum(1)
    Wp = (bc[:, :, None] * W[F[fi]]).sum(1)
    # farthest point sampling to n
    sel = np.zeros(n, int); dmin = np.full(n_cand, np.inf); sel[0] = rng.integers(n_cand)
    for k in range(1, n):
        dmin = np.minimum(dmin, ((P - P[sel[k - 1]]) ** 2).sum(1)); sel[k] = dmin.argmax()
    return P[sel], Wp[sel]


def lbs(Pbind_e, Wp, G, Grest):
    """Pbind_e: points in export frame at rest; G (T,J,4,4) export globals; Grest (J,4,4)."""
    Rel = G @ np.linalg.inv(Grest)[None]                       # (T,J,4,4)
    Ph = np.concatenate([Pbind_e, np.ones((len(Pbind_e), 1))], 1)
    # (T,N,4) = sum_j w_nj Rel_tj @ p_n
    M = np.einsum('nj,tjab->tnab', Wp, Rel)
    return np.einsum('tnab,nb->tna', M, Ph)[..., :3]


def weights_to_export(m, names):
    """Skin weights onto export joint order; a skin joint pruned from the export hands its
    weight to its nearest glTF ancestor that the export kept (pruned fingers -> hand)."""
    We = np.zeros((len(m['V']), len(names))); moved = 0.0
    for k, n in enumerate(m['names']):
        a = n; hops = 0
        while a not in names and a in m['name_parent'] and hops < 64:
            a = m['name_parent'][a]; hops += 1
        if a in names:
            We[:, names.index(a)] += m['W'][:, k]
            if a != n: moved += m['W'][:, k].sum()
    lost = 1 - We.sum(1)
    We /= np.clip(We.sum(1, keepdims=True), 1e-9, None)
    return We, float(lost.max()), moved / len(m['V'])


def part_slip(con, vt, part, thr):
    """Per frame: True if some contacting body part (dominant skin joint) has its 10th-percentile
    tangential speed above thr -- a rolling foot keeps a near-zero pivot, a sliding one does not."""
    out = np.zeros(len(con), bool)
    for i in range(len(con)):
        c = con[i]
        if not c.any(): continue
        for p in np.unique(part[c]):
            sel = c & (part == p)
            if sel.sum() >= 3 and np.percentile(vt[i][sel], 10) > thr:
                out[i] = True; break
    return out
