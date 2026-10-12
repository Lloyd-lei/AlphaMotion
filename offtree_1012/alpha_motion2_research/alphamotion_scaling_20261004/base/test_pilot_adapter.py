import hashlib, importlib.util, json, subprocess, sys, tempfile, unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import pilot_adapter as A

def q_wxyz(rotvec):
 rotvec=np.asarray(rotvec);q=Rotation.from_rotvec(rotvec.reshape(-1,3)).as_quat().reshape(rotvec.shape[:-1]+(4,))
 return np.concatenate([q[...,3:],q[...,:3]],axis=-1)

def oracle_fk(q,offsets,parents,root):
 F,J=q.shape[:2];out=np.zeros((F,J,3));out[:,0]=root;R=[]
 for j,p in enumerate(parents):
  own=Rotation.from_quat(q[:,j,[1,2,3,0]])
  R.append(own if p<0 else R[p]*own)
  if p>=0:out[:,j]=out[:,p]+R[p].apply(np.broadcast_to(offsets[j],(F,3)))
 return out,np.stack([r.as_matrix() for r in R],axis=1)

def fixture(seed=17):
 rng=np.random.default_rng(seed);F,J=13,4;parents=np.array([-1,0,1,1]);offsets=np.array([[0,0,0],[.3,0,0],[0,.5,0],[0,0,.2]])
 tq=q_wxyz(rng.normal(size=(1,J,3)));tpos,_=oracle_fk(tq,offsets,parents,np.array([[.3,.2,.1]]));q=q_wxyz(rng.normal(size=(F,J,3)));p,_=oracle_fk(q,offsets,parents,rng.normal(size=(F,3)))
 c={'parents':parents.tolist(),'joint_names':['root','middle','leaf_a','leaf_b'],'tpos_first_frame':tpos[0],'tpos_local_rotations':tq[0],'offsets':offsets}
 return c,{'global_positions':p,'local_rotations':q,'root_facing_quat':np.tile([1.,0,0,0],(F,1)),'fps':np.array(30)}

class PayloadTest(unittest.TestCase):
 def test_nonidentity_bind_matches_independent_scipy_oracle(self):
  c,z=fixture();static,s=A.prepare_spec(c)
  with tempfile.TemporaryDirectory() as t:
   f=Path(t)/'x.npz';np.savez(f,**z);out,qc,_=A.adapt(f,s)
  self.assertTrue(qc['passed']);self.assertLess(qc['canonical_fk_error']['max'],1e-12)
  self.assertLess(qc['official_rebase_formula_max_abs_error'],1e-12)
  np.testing.assert_allclose(out['pose9'][...,6:],out['position_canonical']/static['reach'],atol=2e-7)
  self.assertFalse(static['dof_valid'].any());self.assertFalse(static['metric_length_valid'])
  # The terminal joints retain their own motion orientation, despite not moving child positions.
  _,raw=oracle_fk(z['local_rotations'],c['offsets'],c['parents'],z['global_positions'][:,0])
  _,tpose=oracle_fk(c['tpos_local_rotations'][None],c['offsets'],c['parents'],np.zeros((1,3)))
  expected=raw@tpose.swapaxes(-1,-2)
  np.testing.assert_allclose(out['global_rot6'],np.concatenate([expected[...,0],expected[...,1]],axis=-1),atol=1e-7)
 def test_dynamic_offsets_are_quarantined(self):
  c,z=fixture();_,s=A.prepare_spec(c);z['global_positions'][:,2,0]+=np.linspace(0,.1,len(z['global_positions']))
  with tempfile.TemporaryDirectory() as t:
   f=Path(t)/'bad.npz';np.savez(f,**z);_,qc,_=A.adapt(f,s)
  self.assertFalse(qc['passed']);self.assertGreater(qc['max_native_local_offset_residual'],.09)
 def test_end_to_end_duplicate_source_group_and_holdout(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);bseen='test_seen';bhold='54e7e37365ac4063ba2ba65151fd9f5d';c,z=fixture();cond=root/'cond.npy';np.save(cond,{bseen:c,bhold:c},allow_pickle=True);condsha=A.sha(cond)
   bodies=[{'object_type':b,'upstream_category':'test','joint_names':c['joint_names'],'conditioning_path':str(cond),'conditioning_sha256':condsha} for b in [bseen,bhold]]
   records=[]
   for i,(b,source,seed) in enumerate([(bseen,'a',0),(bseen,'a_alias',0),(bseen,'b',1),(bseen,'c',2),(bseen,'d',3),(bseen,'e',4),(bhold,'x',5),(bhold,'x_alias',5)]):
    _,arrays=fixture(seed);p=root/f'{b}-{source}-000.npz';np.savez(p,**arrays)
    # Fixtures must share body bind pose. Recompute motion against fixed conditioning.
    arrays['global_positions'],_=oracle_fk(arrays['local_rotations'],c['offsets'],c['parents'],arrays['global_positions'][:,0]);np.savez(p,**arrays)
    records.append({'object_type':b,'dataset':'synthetic','rfilename':p.name,'path':str(p),'verification':{'hash':A.sha(p)}})
   (root/'selected_joint_names.json').write_text(json.dumps({'bodies':bodies}))
   (root/'subset_download_report.json').write_text(json.dumps({'status':'verified_complete','revision':A.REV,'files':records}))
   proc=subprocess.run([sys.executable,str(Path(A.__file__)),'--root',str(root)],capture_output=True,text=True)
   self.assertEqual(proc.returncode,0,proc.stderr)
   m=json.loads((root/'payload_manifest.json').read_text());self.assertEqual(m['duplicate_alias_files'],2);self.assertEqual(m['included_files'],6)
   seen=np.load(root/'payloads'/f'{bseen}.npz');held=np.load(root/'payloads'/f'{bhold}.npz')
   self.assertEqual(set(held['split']),{2});self.assertEqual(set(seen['split']),{0,1})
   self.assertFalse(set(seen['source_group_index'][seen['split']==0])&set(seen['source_group_index'][seen['split']==1]))
   self.assertEqual(m['bodies'][0]['conditioning_spec_error'],None)

if __name__=='__main__':unittest.main()
