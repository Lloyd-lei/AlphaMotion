"""Read-only real-source checks plus isolated, project-local mutation fixtures.

Run using the factory Python; no writes to decisions/issues or factory sources.
"""
from pathlib import Path
import gc
import json
import os
import resource
import tempfile
import unittest
import numpy as np
import mujoco
from backend.adapters import (FactoryService, NpyArchive, SnapshotChanged,
                              UnsupportedFormat, AdapterError, sha256_file)
from backend import geometry
from backend.wire import pack, unpack

PROJECT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.getenv('MOTION_AUDIT_DATA_ROOT', '/home/arenalabs/umr_factory_data'))
FIXTURES = PROJECT / 'qa' / 'adapter-fixtures'
FIXTURES.mkdir(parents=True, exist_ok=True)


def make_fixture(base):
    root = base / 'data'; robot = root / 'robots' / 'fixture'
    robot.mkdir(parents=True); (root/'audit').mkdir()
    clip = root/'out'/'test'; clip.mkdir(parents=True)
    xml = robot/'fixture.xml'
    xml.write_text('<mujoco><worldbody><body pos="0 0 1"><freejoint/><geom type="box" size=".1 .2 .3" contype="0" conaffinity="0"/></body></worldbody></mujoco>')
    points = np.array([[.1,.2,.3],[-.1,-.2,-.3]], np.float32)
    np.savez(robot/'surface.npz', points_local=points, normals_local=points,
             geom=np.zeros(2,np.int32), region=np.zeros(2,np.uint8))
    q = np.array([[0,0,1,1,0,0,0],[.1,.1,1,1,0,0,0]],np.float32)
    args = dict(qpos=q, qpos_umr=q.copy(),fps=np.float32(30),robot_xml=str(xml),regions=['trunk'],
                floor_offset_removed=np.float32(.03),ground_contact=np.zeros((2,1),bool),
                ground_point_contact=np.zeros((2,1),np.uint8),ground_slip_speed=np.zeros((2,1),np.float32),
                provenance=json.dumps({'source_kind':'fixture','source_data':str(root),'seq_key':'test'}))
    np.savez(clip/'fixture.npz', **args)
    return root,args


class FixtureTests(unittest.TestCase):
    def test_wire_point_roundtrip_precision_and_big_endian_bit_indices(self):
        rng=np.random.default_rng(13)
        p=rng.normal(size=(4096,3))*.2
        rotations=np.tile(np.eye(3),(3,1,1)); translations=np.array([[12.34,-8.11,.9],[0,0,0],[1e2,-1e2,0.]])
        b=pack({'coordinate':'metres'}, {'p':p,'r':rotations,'t':translations,'labels':np.array([[128,1]],np.uint8)})
        _,a=unpack(b)
        original=np.einsum('cij,pj->cpi',rotations,p)+translations[:,None,:]
        decoded=np.einsum('cij,pj->cpi',a['r'],a['p'])+a['t'][:,None,:]
        self.assertLess(float(np.max(np.abs(original-decoded))),1e-4)
        bits=np.unpackbits(a['labels'],axis=1,bitorder='big')
        self.assertEqual(np.flatnonzero(bits).tolist(),[0,15])
        with self.assertRaises(ValueError): unpack(b[:-1])

    def test_stale_npz_is_refused_without_silent_rebind(self):
        with tempfile.TemporaryDirectory(dir=FIXTURES) as folder:
            base=Path(folder);root,args=make_fixture(base);s=FactoryService(root,base/'cache')
            item=s.catalog()['items'][0]; sha=item['sha256']
            s.chunk(item['key'],0,1,'corrected',sha)
            args['qpos']=args['qpos']+.001
            np.savez(root/'out/test/fixture.npz',**args)
            with self.assertRaises(SnapshotChanged):s.chunk(item['key'],0,1,'corrected',sha)
            self.assertEqual(SnapshotChanged.code,'stale_snapshot')
            self.assertNotEqual(s.detail(item['key'])['sha256'],sha)

    def test_changed_or_deleted_companion_asset_is_refused(self):
        for change in ('touch','delete'):
            with self.subTest(change=change),tempfile.TemporaryDirectory(dir=FIXTURES) as folder:
                base=Path(folder);root,_=make_fixture(base);s=FactoryService(root,base/'cache')
                item=s.catalog()['items'][0]
                path=root/'robots/fixture/surface.npz'
                if change=='touch':os.utime(path,None)
                else:path.unlink()
                with self.assertRaises(SnapshotChanged):s.chunk(item['key'],0,1,'corrected',item['sha256'])

    def test_unknown_format_is_explicit_and_queue_identity_preserved(self):
        with tempfile.TemporaryDirectory(dir=FIXTURES) as folder:
            base=Path(folder);root,_=make_fixture(base);npz=root/'out/test/fixture.npz'
            queue={'item_id':'pipeline/exact::robot','npz':str(npz),'format':'gpu_npz_v2','priority':.9}
            (root/'audit/queue.jsonl').write_text(json.dumps(queue)+'\n')
            s=FactoryService(root,base/'cache');item=s.catalog()['items'][0]
            self.assertEqual(item['item_id'],queue['item_id']);self.assertTrue(item['queued'])
            self.assertIn('gpu_npz_v2',item['error'])
            with self.assertRaises(UnsupportedFormat):s.detail(item['key'])
            queue['format']='factory_npz_v1';(root/'audit/queue.jsonl').write_text(json.dumps(queue)+'\n')
            self.assertEqual(s.detail(item['key'])['item_id'],queue['item_id'])

    def test_bounded_ranges_and_source_path_allowlist(self):
        with tempfile.TemporaryDirectory(dir=FIXTURES) as folder:
            base=Path(folder);root,_=make_fixture(base);s=FactoryService(root,base/'cache');i=s.catalog()['items'][0]
            for start,count in [(-1,1),(0,0),(0,65),(2,1)]:
                with self.assertRaises(AdapterError):s.chunk(i['key'],start,count,'corrected',i['sha256'])
            with self.assertRaises(AdapterError):s.chunk(i['key'],0,1,'inferred',i['sha256'])
            with self.assertRaises(SnapshotChanged):s.chunk(i['key'],0,1,'corrected','')
            other=base/'unrelated.txt';other.write_text('not a source')
            snap=s._snapshot(i['key'])
            with self.assertRaises(AdapterError):s._bind(snap,other)

    def test_sparse_arrows_are_true_contacts_and_object_local_velocity(self):
        with tempfile.TemporaryDirectory(dir=FIXTURES) as folder:
            base=Path(folder);root,args=make_fixture(base)
            obj=root/'robots/fixture/object.xml'
            obj.write_text('<mujoco><worldbody><body><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
            args.update(ground_slip=np.ones((2,1),bool),ground_point_contact=np.full((2,1),128,np.uint8),
                        object_xml=str(obj),object_pos=np.zeros((2,3),np.float32),
                        object_rot=np.array([np.eye(3),[[0,-1,0],[1,0,0],[0,0,1]]],np.float32),
                        object_point_contact=np.full((2,1),128,np.uint8),object_slip=np.ones((2,1),bool))
            np.savez(root/'out/test/fixture.npz',**args)
            s=FactoryService(root,base/'cache');i=s.catalog()['items'][0]
            m,a=s.chunk(i['key'],0,2,'corrected',i['sha256']);_,a=unpack(pack(m,a))
            self.assertEqual(a['ui_arrow_frame'].tolist(),[0,1,0,1])
            self.assertEqual(a['ui_arrow_point'].tolist(),[0,0,0,0])
            self.assertEqual(a['ui_arrow_kind'].tolist(),[0,0,1,1])
            np.testing.assert_allclose(a['ui_arrow_velocity'][:2],[[3,3,0],[3,3,0]],atol=1e-5)
            np.testing.assert_allclose(a['ui_arrow_velocity'][2:],[[6,-12,0],[6,-12,0]],atol=1e-5)
            for removed in ('ui_point_velocity','ui_object_point_velocity','ui_ground_distance'):
                self.assertNotIn(removed,a)


@unittest.skipUnless(DATA_ROOT.exists(),'real ArenaLabs sources available only on remote host')
class ActualSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s=FactoryService(DATA_ROOT,PROJECT/'cache')
        cls.items=cls.s.catalog()['items']
        cls.evidence={'source_count':len(cls.items),'sources':[], 'fk_max_m':0.,'wire_point_max_m':0.,'human_floor_max_m':0.}

    @classmethod
    def tearDownClass(cls):
        cls.evidence['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (PROJECT/'qa'/'adapters-validation.json').write_text(json.dumps(cls.evidence,indent=2,ensure_ascii=False))

    def test_all_existing_npz_read_and_report_actual_contact_mismatches(self):
        source_paths=sorted(DATA_ROOT.glob('out/*/*.npz'))
        self.assertGreaterEqual(len(source_paths),7)
        by_path={i['npz']:i for i in self.items}
        for path in source_paths:
            item=by_path[str(path)]
            self.assertNotIn('error',item)
            detail=self.s.detail(item['key'],item['sha256'])
            meta,arrays=self.s.scene(item['key'],item['sha256'])
            self.assertTrue(meta['geoms']);self.assertTrue(meta['human_region_available'])
            scene_meta,scene_arrays=unpack(pack(meta,arrays))
            self.assertEqual(scene_meta['sha256'],item['sha256'])
            np.testing.assert_array_equal(scene_arrays['surface_geom'],arrays['surface_geom'])
            with np.load(path,allow_pickle=False) as z:
                for prefix in ('ground','object'):
                    if f'{prefix}_point_contact' not in z:continue
                    count=0
                    for lo in range(0,len(z['qpos']),128):
                        bits=np.unpackbits(z[f'{prefix}_point_contact'][lo:lo+128],axis=1,bitorder='big')[:,:meta['surface_point_count']]
                        for r in range(len(meta['regions'])):
                            hit=bits[:,arrays['surface_region']==r].any(axis=1)
                            count+=int((hit != z[f'{prefix}_contact'][lo:lo+128,r]).sum())
                    self.assertEqual(count,detail['contact_consistency'][prefix]['mismatch_region_frames'])
                if 'kick' in str(path):self.assertAlmostEqual(meta['fps'],90.00090026855469)
            self.evidence['sources'].append({'npz':str(path),'sha256':item['sha256'],'frames':meta['frames'],
                    'fps':meta['fps'],'visual_geom_count':len(meta['geoms']),
                    'human_region_available':meta['human_region_available'],'contact_consistency':detail['contact_consistency'],
                    'object_distance_description':meta['object_distance_description'],'warnings':detail['warnings']})

    def test_fk_wire_and_human_floor_against_independent_reads(self):
        for item in self.items:
            d=self.s.detail(item['key'],item['sha256'])
            with np.load(item['npz'],allow_pickle=False) as source:
                model=mujoco.MjModel.from_xml_path(str(source['robot_xml']));data=mujoco.MjData(model)
                surface_path=Path(str(source['robot_xml'])).parent/'surface.npz'
                with np.load(surface_path,allow_pickle=False) as sf:
                    local=sf['points_local'].astype(np.float64);ids=sf['geom'].astype(int)
                for mode in ('raw','corrected'):
                    q=source['qpos_umr' if mode=='raw' else 'qpos']
                    for frame in (0,len(q)//2,len(q)-1):
                        meta,a=self.s.chunk(item['key'],frame,1,mode,item['sha256']);_,received=unpack(pack(meta,a))
                        data.qpos[:]=q[frame];mujoco.mj_forward(model,data)
                        err=float(np.max(np.abs(received['geom_pos'][0]-data.geom_xpos)))
                        self.evidence['fk_max_m']=max(self.evidence['fk_max_m'],err);self.assertLess(err,1e-4)
                        reference=np.einsum('pij,pj->pi',data.geom_xmat.reshape(-1,3,3)[ids],local)+data.geom_xpos[ids]
                        from_wire=np.einsum('pij,pj->pi',received['geom_rot'][0,ids],local.astype(np.float32))+received['geom_pos'][0,ids]
                        err=float(np.max(np.abs(reference-from_wire)))
                        self.evidence['wire_point_max_m']=max(self.evidence['wire_point_max_m'],err);self.assertLess(err,1e-4)
                        self.assertTrue(meta['human_floor_shift_applied'])
                        # Independent direct npz read checks exactly one floor shift.
                        if mode=='corrected' and frame==0:
                            prov=json.loads(str(source['provenance']))
                            with np.load(prov['human_slots'],allow_pickle=False) as hz:
                                expected=hz['source_slots'][frame].copy()
                            expected[:,2]-=float(source['floor_offset_removed'])
                            err=float(np.max(np.abs(received['human_points'][0]-expected)))
                            self.evidence['human_floor_max_m']=max(self.evidence['human_floor_max_m'],err)
                            self.assertLess(err,1e-7)
                del model,data
                gc.collect()
        self.assertLess(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,4*1024**2)

    def test_bounded_64_frame_chunks_and_object_scale(self):
        for item in self.items:
            meta,scene=self.s.scene(item['key'],item['sha256'])
            m,a=self.s.chunk(item['key'],8,64,'corrected',item['sha256'])
            self.assertEqual(m['count'],64)
            self.assertEqual(a['human_points'].shape,(64,4096,3))
            encoded=pack(m,a);_,decoded=unpack(encoded)
            self.assertLess(len(encoded),16*1024**2)
            self.assertEqual(decoded['geom_pos'].dtype,np.dtype('float32'))
            if meta['has_object']:
                unscaled=mujoco.MjModel.from_xml_path(str(self.s._snapshot(item['key'])['archive'].scalar('object_xml')))
                verts=unscaled.mesh_vert.astype(np.float64)
                # Compiler's inertial mesh frame may rotate; distances are invariant.
                original_diameter=float(np.linalg.norm(verts[0]-verts[-1]))
                g=meta['object_geoms'][0];out=scene[g['mesh']+'_vertices']
                scaled_diameter=float(np.linalg.norm(out[0]-out[-1]))
                self.assertAlmostEqual(scaled_diameter/original_diameter,meta['human_scale'],places=5)
                if meta['object_name'].startswith('Box'):
                    self.assertIn('凸包',m['object_distance_description'])
        self.assertLess(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,4*1024**2)


if __name__=='__main__':
    unittest.main(verbosity=2)
