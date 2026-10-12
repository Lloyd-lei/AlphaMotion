"""Synthetic score fixtures only; verify paired sets and gate behavior."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from pilot_data import write_json
from analyze_pilot import comparison,matched,paired_summary

class AnalysisChecks(unittest.TestCase):
 def test_plot_artifacts_from_synthetic_pipeline(self):
  from test_train_pilot import fixtures,small_model,config
  from train_pilot import evaluate
  from analyze_pilot import plot_outputs
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);evaluate(small_model(),fixtures(),config(),p,'final')
   (p/'train.jsonl').write_text(json.dumps({'step':1,'domain':'xwbc','terms':{'total':1.,'rotation_deg':1.,'position_over_reach':.1}})+'\n')
   paths=plot_outputs(p)
   self.assertEqual(len(paths),4)
   for name in paths:
    image=(p/name).read_bytes();self.assertTrue(image.startswith(b'\x89PNG'));self.assertGreater(len(image),10000)
   self.assertEqual(json.loads((p/'plot_manifest.json').read_text())['visual_review_status'],'pending_visual_inspection')

 def test_same_selected_set(self):
  a={'position_over_reach':np.array([[4.,100.,8.]]),'rotation_deg':np.array([[10.,180.,10.]])}
  b={'position_over_reach':np.array([[2.,1.,4.]]),'rotation_deg':np.array([[5.,1.,5.]])}
  c=comparison(a,b,np.array([[True,False,True]]))
  self.assertEqual(c['position_over_reach']['baseline_mean'],6.)
  self.assertEqual(c['position_over_reach']['relative_improvement'],.5)

 def test_row_mismatch_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)
   for name,rows in [('a',[0,1]),('b',[1,0])]:np.savez(p/f'{name}.npz',rows=rows,position_over_reach=np.ones((2,3)),rotation_deg=np.ones((2,3)),selected=np.ones((2,3),bool))
   with self.assertRaises(ValueError):matched(p,{'error_file':'a.npz'},{'error_file':'b.npz'})

 def test_gate_honors_steps_and_body_holdout(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);cfg=json.loads((Path(__file__).parent/'candidate_config.json').read_text());write_json(p/'config.json',cfg)
   reports={k:{'cases':[]} for k in ['reference','initial','final']}
   for label in reports:
    for body,domain,exposure in [('old','xwbc','pilot_seen_body'),('novel','unimate','pilot_entire_body_holdout')]:
     if label=='reference' and domain!='xwbc':continue
     for mode in ['full','sparse_only']:
      for count in ([0] if label=='reference' else [0,1,2]):
       error=(.5 if label=='final' and count else 1.)
       selected=np.zeros((2,4),bool);selected[:,:count]=True;name=f'{label}_{body}_{mode}_{count}.npz'
       np.savez(p/name,rows=np.arange(2),position_over_reach=np.ones((2,4))*error,rotation_deg=np.ones((2,4))*error,selected=selected)
       reports[label]['cases'].append(dict(target=body,domain=domain,body_exposure=exposure,mode=mode,condition_count=count,error_file=name))
   for label,value in reports.items():write_json(p/f'evaluation_{label}.json',value)
   write_json(p/'real_batch_preflight.json',{'status':'passed'})
   write_json(p/'result.json',{'steps':3000,'checkpoint':{'fixture':'synthetic; nonexistent'}})
   r=paired_summary(p);self.assertTrue(all(r['checks'].values()));self.assertEqual(r['status'],'quantitative_gate_pass_visual_review_pending')
   write_json(p/'result.json',{'steps':3,'checkpoint':{'fixture':'synthetic; nonexistent'}})
   r=paired_summary(p);self.assertFalse(r['checks']['minimum_steps']);self.assertEqual(r['status'],'stop_and_inspect_before_scaleup')

if __name__=='__main__':unittest.main(verbosity=2)
