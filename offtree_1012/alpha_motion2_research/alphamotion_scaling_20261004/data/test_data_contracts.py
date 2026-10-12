import unittest,tempfile,json,copy
from pathlib import Path
import numpy as np
from trajectory_reader import npz_rows,static_reach,local_fk
from mujoco_dataset_adapter import group_split
from build_data_manifest import build
HERE=Path(__file__).resolve().parent

class Contracts(unittest.TestCase):
 def test_npz_numeric_rows_bounded(self):
  with tempfile.TemporaryDirectory() as td:
   p=Path(td)/'z.npz';x=np.arange(60,dtype=np.float16).reshape(20,3);np.savez_compressed(p,x=x)
   np.testing.assert_array_equal(npz_rows(p,'x',7,4),x[7:11])
   with self.assertRaises(IndexError):npz_rows(p,'x',19,2)
 def test_static_reach_ignores_root_translation(self):
  p=np.array([-1,0,1,1]);off=np.array([[0,900,0],[1,0,0],[0,2,0],[0,0,0.]])
  self.assertEqual(static_reach(p,off),3.)
  R,P=local_fk(np.tile([1,0,0,0,1,0],(2,4,1)).astype(float),p,off)
  np.testing.assert_array_equal(P[0],[[0,0,0],[1,0,0],[1,2,0],[1,0,0]])
 def test_episode_alias_cannot_leak(self):
  ks=['a']*10+['b']+['c']*2;sp=group_split(ks)
  self.assertEqual(sp,group_split(list(reversed(ks))));self.assertEqual(len(sp),3)
  self.assertEqual(group_split(['same','same']),{'same':2})
 def test_real_manifest_holdouts_and_aliases(self):
  a=json.loads((HERE/'coverage_audit.json').read_text());e=json.loads((HERE/'external_manifest.json').read_text());m=build(a,e)
  self.assertEqual(len(m['xwbc']['heldout_bodies']),4)
  for b in m['xwbc']['heldout_bodies']:self.assertEqual(m['xwbc']['train_eligible_sparse_rows_by_body'][b],0)
  broken=copy.deepcopy(a);broken['xwbc']['groups'][1]['group_id']=broken['xwbc']['groups'][0]['group_id']
  with self.assertRaises(AssertionError):build(broken,e)
 def test_external_selected_counts_frozen(self):
  d=json.loads((HERE/'external_manifest.json').read_text());a,b=d['bodies'];self.assertEqual((a['selected_frames'],b['selected_frames']),(5025,3699));self.assertEqual(a['native_independent_episodes'],25);self.assertEqual(b['native_independent_episodes'],19)
  for body in d['bodies']:
   self.assertLess(body['serialized_float32_fk_max_error_m'],2e-6)
   self.assertEqual(body['selected_labels']['contact_valid_joint_frames'],0)
   self.assertEqual(body['selected_labels']['ground_valid_frames'],0)
class MixedContracts(unittest.TestCase):
 def test_two_slide_branches_respond_to_q(self):
  from build_relay_v2 import mixed_fk
  parents=np.array([-1,0,1,1]);off=np.array([[0,0,0],[0,1,0],[1,0,0],[1,0,0.]])
  R=np.broadcast_to(np.eye(3),(2,4,3,3)).copy();rot=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
  R[:,1:]=rot;axis=np.array([[0,0,0],[0,0,0],[1,0,0],[-1,0,0.]])
  q=np.array([[0,0,.02,.03],[0,0,.04,.01]]);P=mixed_fk(R,off,parents,axis,q,np.zeros(4))
  np.testing.assert_allclose(P[:,2],[[0,2.02,0],[0,2.04,0]])
  np.testing.assert_allclose(P[:,3],[[0,1.97,0],[0,1.99,0]])
  np.testing.assert_allclose(np.linalg.norm(P[:,2]-P[:,3],axis=-1),[.05,.05])
 def test_v2_actual_mapping_and_no_v1_duplication(self):
  d=json.loads((HERE/'external_manifest_v2.json').read_text());self.assertEqual([b['body_id'] for b in d['bodies']],['adroit_pen','relay_franka_full_v2'])
  mp=HERE/'relay_franka_full_v2.json';mp=mp if mp.exists() else HERE/'payloads/relay_franka_full_v2.json';m=json.loads(mp.read_text());self.assertEqual(m['model']['parents'],[-1,0,1,2,3,4,5,6,7,7,7]);slides=[j for j in m['model']['mechanical_joints'] if j['type']=='slide'];self.assertEqual([j['skeleton_index'] for j in slides],[8,9]);self.assertTrue(m['validation']['v1_unchanged']);self.assertLess(m['validation']['all_frame_serialized_mixed_fk_max_error_m'],2e-6);self.assertGreater(m['source_soft_limit_violations']['max_overshoot_m'],0)

if __name__=='__main__':unittest.main()
