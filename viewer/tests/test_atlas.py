"""Atlas extension tests: gpu_npz_v2 adapter, SMPL-X browser, progress scanner, routes.

Synthetic fixtures live under qa/adapter-fixtures (project-local); real-data
checks are read-only and skip when the source files are absent.
"""
from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
from pathlib import Path
import json
import tempfile
import unittest
import zipfile

import numpy as np

from backend import npzhead
from backend.adapters import FactoryService, UnsupportedFormat, AdapterError, SnapshotChanged
from backend.gpu_adapter import (GpuService, split_export_name, quat_wxyz_to_mat, FLAG_BITS, guess_source)
from backend.human_smplx import rodrigues, floor_from_points, RID, part_to_region
from backend.progress import ProgressScanner, SOURCES
from backend.human_smplx import SmplxIndex, HumanService
from backend.wire import pack, unpack

PROJECT = Path(__file__).resolve().parents[1]
FIXTURES = PROJECT / 'qa' / 'adapter-fixtures'
FIXTURES.mkdir(parents=True, exist_ok=True)
REAL_EXPORT = Path(f'{SRCFG.work}/results/export/'
                   'case3_kick_right__20260330000586_1_1775269234__unitree_g1_29dof.npz')
REAL_PROBLEM = Path(f'{SRCFG.work}/problems/hiphi_getup__getup0004_neutral_30fps/'
                    'unitree_g1_29dof.stock.cpu.npz')
REAL_SOURCE = Path(f'{SRCFG.work}/sources/hiphi_getup/getup0004_neutral_30fps.npz')


def make_gpu_fixture(base):
    """Tiny robot (free body + box geom offset from the body) and one gpu_npz_v2 export."""
    data = base / 'data'; robot = base / 'robots' / 'fixbot'; export = base / 'export'
    for d in (data / 'robots' / 'fixbot', robot, export):
        d.mkdir(parents=True)
    xml = robot / 'fixbot.xml'
    xml.write_text('<mujoco><worldbody><body name="torso" pos="0 0 1"><freejoint/>'
                   '<geom type="box" size=".1 .2 .3" pos=".05 0 0" euler="0 0 90" contype="0" conaffinity="0"/>'
                   '<body name="leg" pos="0 0 -.4"><joint type="hinge" axis="0 1 0"/>'
                   '<geom type="capsule" size=".05 .2" contype="0" conaffinity="0"/></body></body></worldbody></mujoco>')
    pts = np.array([[.1, .2, .3], [-.1, -.2, -.3], [0, 0, -.25]], np.float32)
    np.savez(robot / 'surface.npz', points_local=pts, normals_local=np.tile([[0, 0, -1.]], (3, 1)).astype(np.float32),
             geom=np.array([0, 0, 1], np.int32), body=np.array([1, 1, 2], np.int32),
             region=np.array([0, 5, 6], np.int16), regions=np.array(['trunk', 'L_thigh', 'R_thigh', 'L_shin', 'R_shin', 'L_foot', 'R_foot',
                                                                   'L_upper_arm', 'R_upper_arm', 'L_forearm', 'R_forearm', 'L_hand', 'R_hand']))
    T = 4
    q = np.zeros((T, 8), np.float32); q[:, 2] = 1.0; q[:, 3] = 1.0; q[:, 0] = np.arange(T) * .01; q[3, 0] = .5
    # frame 0: point 2 ground contact; frame 1: none; frame 2: points 1 (ground) and 2 (object); frame 3: point 0
    contact_ptr = np.array([0, 1, 1, 3, 4], np.int64)
    contact_idx = np.array([2, 1, 2, 0], np.uint16); contact_partner = np.array([0, 0, 1, 0], np.uint8)
    near_ptr = np.array([0, 2, 2, 3, 3], np.int64)
    near_idx = np.array([2, 1, 0], np.uint16); near_partner = np.array([0, 0, 1], np.uint8)
    near_dist_q = np.array([5, 205, 0], np.uint8)  # 0 mm, 20 mm, <= -0.5 mm
    gc = np.zeros((T, 13), bool); gc[0, 6] = True; gc[2, 5] = True
    flags = np.array([0, 1 | 32, 0, 64], np.int32)
    qc = dict(frames=T, flagged=2, flags_hist={str(b): int(((flags & b) > 0).sum()) for b in (1, 2, 4, 8, 16, 32, 64)},
              ground_pen_max_mm=1.5, object_pen_max_mm=0., ground_slip_frac_of_contact=0.5, human_contact_robot_solid_frac=0.75)
    pose = np.zeros((T, 7), np.float32); pose[:, 3] = 1; pose[:, 0] = 2.
    pose[1, 3:] = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
    clip = 'case9_demo__123_4_567'
    path = export / f'{clip}__fixbot.npz'
    np.savez_compressed(path, qpos=q, fps=np.float32(30.), floor_offset_removed=np.float32(.012), floor_height=np.float32(0),
                        contact_ptr=contact_ptr, contact_idx=contact_idx, contact_partner=contact_partner,
                        near_ptr=near_ptr, near_idx=near_idx, near_partner=near_partner, near_dist_q=near_dist_q,
                        ground_contact=gc, ground_slip=np.zeros((T, 13), bool),
                        ground_slip_speed=np.full((T, 13), np.nan, np.float16),
                        ground_pen_mm=np.zeros((T, 13), np.float16), ground_clearance_mm=np.ones((T, 13), np.float16),
                        human_ground_contact=gc.copy(), human_ground_clearance_mm=np.ones((T, 13), np.float16),
                        flags=flags, qc_json=np.array(json.dumps(qc)), object_pose=pose,
                        surface_asset=str(robot / 'surface.npz'), robot_xml=str(xml))
    return data, export, path, clip, robot


class NpzHeadTests(unittest.TestCase):
    def test_header_small_member_and_stored_mmap(self):
        with tempfile.TemporaryDirectory(dir=FIXTURES) as folder:
            p = Path(folder) / 'a.npz'
            big = np.arange(60, dtype=np.float32).reshape(5, 4, 3)
            np.savez(p, big=big, s=np.array('hello'), f=np.float32(90.0009))
            self.assertEqual(npzhead.member_header(p, 'big'), ((5, 4, 3), np.dtype('float32')))
            self.assertEqual(str(npzhead.small_member(p, 's')), 'hello')
            m = npzhead.stored_memmap(p, 'big')
            np.testing.assert_array_equal(np.asarray(m[2:4]), big[2:4])
            c = Path(folder) / 'c.npz'
            np.savez_compressed(c, big=big)
            with self.assertRaises(ValueError):
                npzhead.stored_memmap(c, 'big')
            o = Path(folder) / 'o.npz'
            np.savez(o, obj=np.array([{'a': 1}], dtype=object))
            with self.assertRaises(ValueError):
                npzhead.small_member(o, 'obj')


class GpuAdapterTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(dir=FIXTURES)
        base = Path(self._tmp.name)
        self.data, self.export, self.path, self.clip, self.robot = make_gpu_fixture(base)
        self.s = GpuService(self.data, base / 'cache', export_dirs=(self.export,), problems=base / 'problems',
                            extra_roots=(base,))

    def tearDown(self):
        self._tmp.cleanup()

    def test_name_split_keeps_double_underscore_clip(self):
        self.assertEqual(split_export_name('case3_kick_right__2026_1_17__unitree_g1_29dof', ['unitree_g1_29dof', 'g1']),
                         ('case3_kick_right__2026_1_17', 'unitree_g1_29dof'))
        self.assertEqual(guess_source('case3_kick_right__x'), 'omnicontact')
        self.assertEqual(guess_source('hiphi_getup__getup0004'), 'hiphi')
        self.assertEqual(guess_source('lafan1_smplx__dance1'), 'lafan')

    def test_catalog_detail_flags_and_consistency(self):
        cat = self.s.catalog()
        self.assertEqual(len(cat['clips']), 1)
        clip = cat['clips'][0]
        self.assertEqual(clip['clip_id'], self.clip); self.assertEqual(clip['robots'][0]['robot'], 'fixbot')
        self.assertEqual([b['bit'] for b in cat['flag_bits']], [1, 2, 4, 8, 16, 32, 64])
        d = self.s.detail(clip['robots'][0]['key'])
        self.assertEqual(d['format'], 'gpu_npz_v2'); self.assertEqual(d['frames'], 4)
        self.assertEqual(d['flagged_frames'], 2)
        self.assertEqual(d['flag_counts'], {'1': 1, '2': 0, '4': 0, '8': 0, '16': 0, '32': 1, '64': 1})
        # region labels: frame0 R_foot (point 2) ok, frame2 L_foot (point 1) ok
        self.assertEqual(d['consistency']['ground']['labeled_without_point'], 0)
        self.assertEqual(d['consistency']['ground']['points_without_label'], 1)  # frame 3 point 0 (trunk) unlabeled
        self.assertEqual(d['root_motion']['frames_over_10cm'], 1)
        self.assertFalse(d['has_human'])

    def test_chunk_fk_csr_slices_and_dequantisation(self):
        key = self.s.catalog()['clips'][0]['robots'][0]['key']
        d = self.s.detail(key)
        meta, arrays = self.s.scene(key, d['sha256'])
        _, arrays = unpack(pack(meta, arrays))
        self.assertIn('timeline_ground_contact', arrays); self.assertIn('timeline_root_jump_cm', arrays)
        cm, ca = self.s.chunk(key, 1, 3, d['sha256'])
        cm, ca = unpack(pack(cm, ca))
        # CSR rebased to the chunk: frames 1..3 -> contact counts 0,2,1
        np.testing.assert_array_equal(ca['contact_ptr'], [0, 0, 2, 3])
        np.testing.assert_array_equal(ca['contact_idx'], [1, 2, 0])
        np.testing.assert_array_equal(ca['contact_partner'], [0, 1, 0])
        np.testing.assert_array_equal(ca['near_ptr'], [0, 0, 1, 1])
        dmm = ca['near_dist_q'].astype(float) * 0.1 - 0.5
        np.testing.assert_allclose(dmm, [-0.5])
        # FK: geom-local surface points through geom world pose == MuJoCo world positions
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(self.robot / 'fixbot.xml')); dd = mujoco.MjData(m)
        q = np.load(self.path)['qpos']; sp = np.load(self.robot / 'surface.npz')
        for j, f in enumerate(range(1, 4)):
            dd.qpos[:] = q[f]; mujoco.mj_forward(m, dd)
            for i, (p, g) in enumerate(zip(sp['points_local'], sp['geom'])):
                expect = dd.geom_xmat[g].reshape(3, 3) @ p + dd.geom_xpos[g]
                got = ca['geom_rot'][j, g] @ p + ca['geom_pos'][j, g]
                np.testing.assert_allclose(got, expect, atol=1e-6)
        # object quaternion (wxyz) -> rotation
        np.testing.assert_allclose(ca['object_rot'][0], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-6)
        np.testing.assert_allclose(quat_wxyz_to_mat(np.array([1., 0, 0, 0])), np.eye(3))
        with self.assertRaises(AdapterError):
            self.s.chunk(key, 0, 65, d['sha256'])
        with self.assertRaises(SnapshotChanged):
            self.s.chunk(key, 0, 2, '0' * 64)

    def test_missing_required_field_is_reported(self):
        z = dict(np.load(self.path))
        z.pop('near_dist_q')
        np.savez(self.path, **z)
        key = self.s.catalog()['clips'][0]['robots'][0]['key']
        with self.assertRaises(AdapterError) as ctx:
            self.s.detail(key)
        self.assertIn('near_dist_q', str(ctx.exception))

    def test_factory_hook_points_to_gpu_page(self):
        fs = FactoryService(self.data, Path(self._tmp.name) / 'fcache')
        with self.assertRaises(UnsupportedFormat) as ctx:
            fs.adapters['gpu_npz_v2'].open(fs, 'x')
        self.assertIn('GPU 输出', str(ctx.exception))


class HumanTests(unittest.TestCase):
    def test_rodrigues_matches_scipy(self):
        from scipy.spatial.transform import Rotation
        aa = np.random.default_rng(3).normal(size=(20, 3))
        aa[0] = 0
        np.testing.assert_allclose(rodrigues(aa), Rotation.from_rotvec(aa).as_matrix(), atol=1e-12)

    def test_floor_rule_and_region_map(self):
        reg = np.array([RID['L_foot']] * 5 + [RID['R_foot']] * 5 + [0] * 3)
        X = np.zeros((40, 13, 3)); X[..., 2] = 0.5; X[:, :10, 2] = 0.031
        floor, n, how = floor_from_points(X, reg, 30.)
        self.assertAlmostEqual(floor, 0.031); self.assertGreaterEqual(n, 10)
        self.assertEqual(part_to_region([1, 6, 8, 4]).tolist(), [RID['R_hand'], RID['L_foot'], RID['R_foot'], RID['trunk']])

    @unittest.skipUnless(REAL_PROBLEM.exists() and REAL_SOURCE.exists(), 'real UMR capture not available')
    def test_numpy_smplx_slots_match_umr_capture(self):
        from backend.human_smplx import SmplxModel, Y_UP_TO_Z_UP
        m = SmplxModel()
        with zipfile.ZipFile(REAL_PROBLEM) as z:
            ld = lambda n: npzhead.small_member(REAL_PROBLEM, n, z, max_bytes=10 ** 9)
            fid, bary = ld('x_bind_face_ids'), ld('x_bind_bary').astype(np.float64)
            v0, v1 = ld('x_vertices_world0').reshape(-1, 10475, 3)[0], ld('x_vertices_scaled0').reshape(-1, 10475, 3)[0]
        src = npzhead.stored_memmap(REAL_PROBLEM, 'src')
        d = np.load(REAL_SOURCE)
        vs, J = m.shaped(d['betas'])
        fr = np.array([0, 700, 1500])
        verts, _ = m.forward(vs, J, d['poses'][fr, :66], d['trans'][fr], np.arange(10475))
        np.testing.assert_allclose(verts[0], v0, atol=1e-5)
        v0z = v0 @ Y_UP_TO_Z_UP.T
        scale = np.median(v1[np.abs(v0z) > 1e-3] / v0z[np.abs(v0z) > 1e-3])
        slots = np.einsum('nk,cnkd->cnd', bary, (verts @ Y_UP_TO_Z_UP.T)[:, m.faces[fid]]) * scale
        self.assertLess(float(np.abs(slots - np.asarray(src[fr])).max()), 1e-5)

    def test_index_on_fixture_tree(self):
        with tempfile.TemporaryDirectory(dir=FIXTURES) as folder:
            root = Path(folder) / 'data'
            clip = root / 'smplx' / 'hiphi' / 'data' / 'Taking' / 'take' / 'Taking-take_0001'
            clip.mkdir(parents=True)
            np.savez_compressed(clip / 'motion_actor_smplx.npz', poses=np.zeros((90, 165), np.float32),
                                trans=np.zeros((90, 3), np.float32), betas=np.zeros(10, np.float32),
                                gender=np.array('neutral'), fps=np.float32(30.), output_up=np.array('y'))
            (clip / 'qa.json').write_text(json.dumps({'ik_residual_cm': {'mean': 1.5, 'p95': 2.0, 'max': 3.0}}))
            idx = SmplxIndex(root, Path(folder) / 'cache')
            entries = idx.scan()
            self.assertEqual(len(entries), 1)
            e = entries[0]
            self.assertEqual((e['source'], e['variant'], e['category'], e['clip']), ('hiphi', 'cpu', 'Taking/take', 'Taking-take_0001'))
            self.assertEqual(e['frames'], 90); self.assertAlmostEqual(e['duration_s'], 3.0); self.assertEqual(e['residual_cm_mean'], 1.5)
            # cached second scan returns identical entry without re-reading
            self.assertEqual(SmplxIndex(root, Path(folder) / 'cache').scan()[0]['frames'], 90)


class ProgressTests(unittest.TestCase):
    def test_empty_root_lists_every_known_source(self):
        with tempfile.TemporaryDirectory(dir=FIXTURES) as folder:
            root = Path(folder) / 'data'; (root / 'robots').mkdir(parents=True)
            idx = SmplxIndex(root, Path(folder) / 'cache')
            r = ProgressScanner(root, Path(folder) / 'cache', idx).scan()
            self.assertEqual([f['id'] for f in r['funnel']], [s['id'] for s in SOURCES])
            self.assertTrue(all(f['smplx']['hours'] == 0 for f in r['funnel']))

    def test_background_scan_is_rate_limited(self):
        with tempfile.TemporaryDirectory(dir=FIXTURES) as folder:
            root = Path(folder) / 'data'; (root / 'robots').mkdir(parents=True)
            ps = ProgressScanner(root, Path(folder) / 'cache', SmplxIndex(root, Path(folder) / 'cache'))
            first = ps.get(force=True)
            self.assertTrue(first['scanning'])
            done = ps.wait(60)
            self.assertIsNotNone(done['result'])
            again = ps.get()
            self.assertFalse(again['scanning'])  # < 5 min since last scan: no new thread


class RouteTests(unittest.TestCase):
    def test_atlas_routes_with_fixture_services(self):
        from fastapi.testclient import TestClient
        from backend.app import create_app
        with tempfile.TemporaryDirectory(dir=FIXTURES) as folder:
            base = Path(folder)
            data, export, path, clip, robot = make_gpu_fixture(base)
            (data / 'audit').mkdir()
            gpu = GpuService(data, base / 'cache', export_dirs=(export,), problems=base / 'problems', extra_roots=(base,))
            human = HumanService(data, base / 'hcache')
            prog = ProgressScanner(data, base / 'hcache', human.index)
            fs = FactoryService(data, base / 'fcache')
            app = create_app(service=fs, audit_root=PROJECT / 'qa' / 'fixture_audit', data_root=data, mode='test',
                             gpu_service=gpu, human_service=human, progress=prog)
            c = TestClient(app)
            cat = c.get('/api/gpu/catalog').json()
            key = cat['clips'][0]['robots'][0]['key']
            d = c.get('/api/gpu/items/' + key).json()
            self.assertEqual(d['frames'], 4)
            r = c.get(f'/api/gpu/items/{key}/chunk?sha256={d["sha256"]}&start=0&count=4')
            self.assertEqual(r.status_code, 200)
            meta, arrays = unpack(r.content)
            self.assertEqual(arrays['geom_pos'].shape[0], 4)
            self.assertEqual(c.get(f'/api/gpu/items/{key}/chunk?sha256={d["sha256"]}&start=0&count=65').status_code, 422)
            self.assertEqual(c.get(f'/api/gpu/items/{key}/chunk?sha256={"0" * 64}&start=0&count=2').status_code, 409)
            p = c.get('/api/atlas/progress?force=true').json()
            self.assertIn('scanning', p)
            prog.wait(120)  # let the background scan finish before the temp tree is removed
            self.assertEqual(c.get('/api/human/catalog').json()['total'], 0)
            self.assertEqual(c.get('/api/human/items/nope').status_code, 404)


class ProductionLayoutTests(unittest.TestCase):
    def test_provenance_fallback_and_self_invalidated_engine(self):
        with tempfile.TemporaryDirectory(dir=FIXTURES) as folder:
            base = Path(folder)
            data, export, path, clip, robot = make_gpu_fixture(base)
            z = dict(np.load(path))
            prov = dict(robot_xml=str(z.pop('robot_xml')), surface_asset=str(z.pop('surface_asset')), engine='umr_gpu-abc123def456',
                        clip_id='box__demo__1')
            z['provenance'] = np.array(json.dumps(prov))
            out = data / 'production' / 'v1' / 'omnicontact' / 'box__demo__1'
            out.mkdir(parents=True)
            np.savez_compressed(out / 'fixbot.npz', **z)
            np.savez_compressed(out / '.fixbot.tmp.npz', **z)            # atomic-write temp: must be ignored
            (data / 'production' / 'v1' / '_assets' / 'fixbot').mkdir(parents=True)
            np.savez(data / 'production' / 'v1' / '_assets' / 'fixbot' / 'surface.npz', a=np.zeros(1))
            inv = data / 'production' / 'v1' / '_invalid'; inv.mkdir()
            (inv / 'README.txt').write_text('Outputs written by engine umr_gpu-abc123def456 are INVALID: root jumps.')
            s = GpuService(data, base / 'cache', export_dirs=(), problems=base / 'problems', extra_roots=(base,))
            cat = s.catalog(pipeline='production_v1')
            self.assertEqual(cat['total'], 1)
            c = cat['clips'][0]
            self.assertEqual((c['source'], c['clip_id'], c['robots'][0]['robot']), ('omnicontact', 'box__demo__1', 'fixbot'))
            self.assertTrue(c['invalid'])
            d = s.detail(c['robots'][0]['key'])
            self.assertTrue(d['warnings'][0].startswith('⚠'))
            self.assertEqual(d['robot_xml'], str((robot / 'fixbot.xml').resolve()))
            idx = SmplxIndex(data, base / 'hcache')
            r = ProgressScanner(data, base / 'hcache', idx).scan()
            self.assertEqual(r['invalid_files'], 1)
            self.assertNotIn('fixbot', r['matrix'].get('omnicontact', {}))
            self.assertIn('umr_gpu-abc123def456', r['production']['invalid_engines'])

    @unittest.skipUnless(any(Path(f'{SRCFG.data_root}/production/v1').glob('*/*/*.npz'))
                         if Path(f'{SRCFG.data_root}/production/v1').exists() else False,
                         'no production outputs on disk')
    def test_real_production_human_rebuild_matches_file_clearance(self):
        from backend.atlas_routes import atlas_services
        data = Path(f'{SRCFG.data_root}')
        with tempfile.TemporaryDirectory(dir=FIXTURES) as folder:
            gpu, human, _ = atlas_services(data, Path(folder))
            cat = gpu.catalog(pipeline='production_v1', limit=1)
            key = cat['clips'][0]['robots'][0]['key']
            d = gpu.detail(key)
            if not d['has_human']:
                self.skipTest('production file without reconstructable human source')
            meta, arrays = gpu.chunk(key, 0, 8, d['sha256'])
            self.assertEqual(meta['human_mode'], 'smplx')
            self.assertLess(meta['human_check_max_abs_mm'], 0.5)
