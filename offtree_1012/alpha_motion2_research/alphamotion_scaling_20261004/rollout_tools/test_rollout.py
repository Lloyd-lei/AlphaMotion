"""Contract adversarial tests plus real-data and irregular-time encoding smoke."""
from pathlib import Path
import os,json,tempfile,shutil,unittest
import numpy as np
from PIL import Image
from contracts import load_bundle,cache_key,endpoint_metrics
from prepare import sha,write
from encode import encode

ROOT=Path(os.environ['ROLLOUT_ROOT'])
FONT=os.environ.get('ROLLOUT_FONT','/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')

class RolloutTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        src=ROOT/'bundles/adroit_pen';self.bundle=self.base/'adroit_pen/A_M_seed0';self.bundle.parent.mkdir();shutil.copytree(src/'A_M_seed0',self.bundle)
        for f in ['model.mjb','metadata.json']:shutil.copyfile(src/f,self.bundle.parent/f)
    def tearDown(self):self.tmp.cleanup()
    def change_manifest(self,**kw):
        p=self.bundle/'manifest.json';d=json.loads(p.read_text());d.update(kw);write(p,d)
    def change_arrays(self,fn):
        p=self.bundle/'arrays.npz';a=dict(np.load(p,allow_pickle=False));fn(a);np.savez_compressed(p,**a);self.change_manifest(arrays_sha256=sha(p))
    def test_real_shared_case_identity(self):
        for body in ['adroit_pen','relay_franka_full_v2']:
            first=None
            for p in sorted((ROOT/'bundles'/body).glob('*/manifest.json')):
                m,z,_=load_bundle(p.parent)
                current=(m,z)
                if first:
                    self.assertEqual(m['case_id'],first[0]['case_id'])
                    for k in ['frame_ids','timestamps_seconds','reference_position','reference_rotation','condition_position_cm','condition_global_rotation']:np.testing.assert_array_equal(z[k],first[1][k])
                first=current
                self.assertTrue(all(v['max_final_normalized_state_delta']==0 and v['max_curve_metric_delta']==0 for v in m['replay_validation'].values()))
                for t in [0,len(z['frame_ids'])-1]:
                    mm,deg=endpoint_metrics(z,'reference',t);self.assertLess(mm.max(),1e-4);self.assertLess(deg.max(),1e-4)
    def test_units_and_frames_rejected(self):
        for key,value in [('position_unit','m'),('frame','world_z_up'),('time_unit','milliseconds')]:
            original=json.loads((self.bundle/'manifest.json').read_text());self.change_manifest(**{key:value})
            with self.assertRaisesRegex(ValueError,'coordinate frame or units'):load_bundle(self.bundle)
            write(self.bundle/'manifest.json',original)
    def test_time_and_frame_identity_rejected(self):
        self.change_arrays(lambda a:a['timestamps_seconds'].__setitem__(1,a['timestamps_seconds'][0]))
        with self.assertRaisesRegex(ValueError,'frame/time manifest mismatch'):load_bundle(self.bundle)
    def test_bad_mesh_reference_rejected(self):
        with (self.bundle.parent/'model.mjb').open('ab') as f:f.write(b'bad')
        with self.assertRaisesRegex(ValueError,'bad input reference'):load_bundle(self.bundle)
    def test_invalid_rotation_rejected(self):
        self.change_arrays(lambda a:a['q20_rotation'].__setitem__((0,0,0,0),2.))
        with self.assertRaisesRegex(ValueError,'not SO3'):load_bundle(self.bundle)
    def test_cache_invalidates_on_inputs_and_view(self):
        base=cache_key(self.bundle,'full',FONT)
        self.assertNotEqual(base,cache_key(self.bundle,'detail',FONT))
        self.change_manifest(case_id='different-case')
        self.assertNotEqual(base,cache_key(self.bundle,'full',FONT))
    def test_nonuniform_video_preserves_pts(self):
        frames=self.base/'frames';frames.mkdir()
        for i,color in enumerate(['red','green','blue']):Image.new('RGB',(1920,1080),color).save(frames/f'{i:06d}.png')
        result=encode(frames,np.array([20.,20.010003,20.037019]),self.base/'smoke.mp4')
        self.assertEqual(result['encoded_frames'],3);self.assertLess(result['max_pts_error_seconds'],1e-6);self.assertEqual(result['full_decode'],'passed')

if __name__=='__main__':unittest.main(verbosity=2)
