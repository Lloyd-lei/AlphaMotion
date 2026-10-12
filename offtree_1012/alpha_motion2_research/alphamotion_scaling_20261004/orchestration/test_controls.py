"""Numerical controls for reproducible exposure and interpretable accounting."""
from pathlib import Path
import sys,unittest
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'base'));sys.path.insert(0,str(ROOT/'base/legacy'))
from test_recipe_training import bodies,config
from recipe_training import make_reference_inputs
from compute_accounting import ComputeCounter,effective_tokens
from scaling_data import static_reach

class Controls(unittest.TestCase):
    def test_masks_independent_of_model_rng_consumption(self):
        b=bodies()[0];rows=np.array([0,1,2,3]);c=config();device=torch.device('cpu')
        torch.manual_seed(3);first,_=make_reference_inputs(b,b,rows,np.random.default_rng(731),device,c)
        torch.manual_seed(99);torch.randn(70000)
        second,_=make_reference_inputs(b,b,rows,np.random.default_rng(731),device,c)
        for key in ['source_hidden','source_contact']:
            torch.testing.assert_close(first[key],second[key],equal_nan=True)
        for key in ['joint_indices','valid','values']:
            torch.testing.assert_close(first['condition'][key],second['condition'][key],equal_nan=True)
    def test_data_mask_does_not_advance_dropout_rng(self):
        b=bodies()[0];torch.manual_seed(31);before=torch.get_rng_state().clone()
        make_reference_inputs(b,b,np.arange(4),np.random.default_rng(731),torch.device('cpu'),config())
        self.assertTrue(torch.equal(torch.get_rng_state(),before))
    def test_compute_linear_and_mha_arithmetic_without_double_count(self):
        layer=torch.nn.Linear(3,5);counter=ComputeCounter(layer);layer(torch.zeros(2,4,3))
        self.assertEqual(counter.forward_flops,2*2*4*3*5);counter.close()
        attention=torch.nn.MultiheadAttention(8,2,batch_first=True);counter=ComputeCounter(attention)
        attention(torch.randn(2,3,8),torch.randn(2,5,8),torch.randn(2,5,8),need_weights=False)
        expected=2*2*(3*8*8+5*8*8+5*8*8+3*8*8)+4*2*3*5*8
        self.assertEqual(counter.forward_flops,expected);self.assertEqual(counter.report()['estimated_dense_training_flops'],expected*3);counter.close()
    def test_normalizer_depends_only_on_static_paths(self):
        parents=np.array([-1,0,1,0]);off=np.array([[0,0,0],[1,0,0],[0,2,0],[0,0,2]])
        self.assertEqual(static_reach(parents,off),3.)
    def test_padding_conditions_are_not_effective_input(self):
        row={'directions':[{'rows':4,'valid_conditions':3,'source_pose_tokens':20,'target_joint_tokens':20,'source_surface_tokens':0,'target_surface_tokens':0}], 'auxiliary':{}}
        r=effective_tokens(row,{})
        self.assertEqual(r['padding_tokens'],5);self.assertEqual(r['valid_conditions'],3)

if __name__=='__main__':unittest.main(verbosity=2)
