"""Paired native-FK evaluation and plots; no inference optimization, simulated success or invented metrics."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from pilot_data import read_json,write_json


def indexed(report):return {(c['target'],c['mode'],c['condition_count']):c for c in report['cases']}


def load_errors(run,case):
 with np.load(run/case['error_file'],allow_pickle=False) as z:return {k:z[k] for k in z.files}


def matched(run,a,b):
 x=load_errors(run,a);y=load_errors(run,b)
 if not np.array_equal(x['rows'],y['rows']):raise ValueError('paired comparison row mismatch')
 if x['position_over_reach'].shape!=y['position_over_reach'].shape:raise ValueError('comparison topology mismatch')
 return x,y


def comparison(x,y,mask=None):
 report={}
 for k in ['position_over_reach','rotation_deg']:
  a=x[k] if mask is None else x[k][mask];b=y[k] if mask is None else y[k][mask]
  if a.size==0:raise ValueError('empty comparison set')
  mean0=float(a.mean());mean1=float(b.mean())
  report[k]={'baseline_mean':mean0,'candidate_mean':mean1,'relative_improvement':(mean0-mean1)/max(mean0,1e-12),'count':int(a.size)}
 return report


def paired_summary(run):
 config=read_json(run/'config.json');result=read_json(run/'result.json')
 initial=indexed(read_json(run/'evaluation_initial.json'));final=indexed(read_json(run/'evaluation_final.json'));reference=indexed(read_json(run/'evaluation_reference.json'))
 conditions=[];retention=[];holdout=[]
 for key,c in final.items():
  body,mode,count=key
  if count>0:
   zero=final[(body,mode,0)];x,y=matched(run,zero,c);mask=y['selected']
   entry={'target':body,'mode':mode,'count':count,'body_exposure':c['body_exposure'],'selected_same_joint_set':comparison(x,y,mask),'remaining_same_joint_set':comparison(x,y,~mask)}
   conditions.append(entry)
  elif mode=='full':
   if c['domain']=='xwbc':
    x,y=matched(run,reference[key],c);retention.append({'target':body,**comparison(x,y)})
   elif c['body_exposure']=='pilot_entire_body_holdout':
    x,y=matched(run,initial[key],c);holdout.append({'target':body,**comparison(x,y)})
 def average(items,metric,nested=None):
  if not items:return None
  return float(np.mean([(x[nested] if nested else x)[metric]['relative_improvement'] for x in items]))
 # Primary test: source pose absent, lawful sparse constraints; report full-source/masked cases separately above.
 primary=[x for x in conditions if x['mode']=='sparse_only'];gate=config['expand_gate']
 translation=average(primary,'position_over_reach','selected_same_joint_set');rotation=average(primary,'rotation_deg','selected_same_joint_set')
 oldp=average(retention,'position_over_reach');oldr=average(retention,'rotation_deg')
 newp=average(holdout,'position_over_reach');newr=average(holdout,'rotation_deg')
 checks={'real_batch_checks_pass':read_json(run/'real_batch_preflight.json')['status']=='passed',
  'minimum_steps':result['steps']>=gate['minimum_completed_steps'],
  'selected_translation_improvement':translation is not None and translation>=gate['selected_endpoint_position_improvement_fraction_vs_no_se3'],
  'selected_rotation_improvement':rotation is not None and rotation>=gate['selected_endpoint_rotation_improvement_fraction_vs_no_se3'],
  'old_fullbody_retention':oldp is not None and oldr is not None and min(oldp,oldr)>=-gate['old_domain_fullbody_regression_fraction_max_vs_reference'],
  'new_body_nonregression':newp is not None and newr is not None and min(newp,newr)>=-gate['heldout_new_body_regression_fraction_max_vs_initial']}
 report={'status':'quantitative_gate_pass_visual_review_pending' if all(checks.values()) else 'stop_and_inspect_before_scaleup',
  'steps':result['steps'],'checks':checks,'aggregation':'equal weight per evaluated body/mode/condition-count, not frame-count weighted',
  'primary_sparse_only_selected_mean_relative_improvement':{'position':translation,'rotation':rotation},
  'condition_pairs':conditions,'old_domain_reference_retention':retention,'whole_new_body_initial_to_final':holdout,
  'visual_review':'required; automated plot creation is not human visual acceptance',
  'limitations':['Single seed bounded pilot','Several simultaneous changes; no isolated causal attribution','No world/object absolute anchors','Native output only; no refinement','No dynamics, simulation or physical robot validation','UniMate source ancestry across external datasets remains uncertified'],
  'checkpoint':result['checkpoint']}
 write_json(run/'paired_summary.json',report);return report


def plot_outputs(run):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 final=indexed(read_json(run/'evaluation_final.json'))
 candidates=[c for c in final.values() if c['mode']=='sparse_only' and c['condition_count']>0]
 selected=[]
 for want in ['unitree_g1_29dof','e457402d4b2e4681b5a5f88b747aea4a','79fed889082e4a25ad92be75b57f641c']:
  cases=[c for c in candidates if c['target']==want]
  if cases:selected.append(max(cases,key=lambda c:c['condition_count']))
 if len(selected)<3:
  for c in candidates:
   if c['target'] not in {x['target'] for x in selected}:selected.append(c)
   if len(selected)>=3:break
 plotpaths=[]
 for c in selected:
  name=c['target'];count=c['condition_count'];mode=c['mode']
  with np.load(run/f'final_{name}_{mode}_c0_examples.npz',allow_pickle=False) as z:z0={k:z[k] for k in z.files}
  with np.load(run/f'final_{name}_{mode}_c{count}_examples.npz',allow_pickle=False) as z:z1={k:z[k] for k in z.files}
  if not np.array_equal(z0['rows'],z1['rows']):raise ValueError('plot pairing')
  scale=10 if bool(z1['metric_length_valid']) else 1;unit='mm' if scale==10 else 'canonical units'
  gt=z1['reference_positions']*scale;p0=z0['predicted_fk_positions']*scale;p1=z1['predicted_fk_positions']*scale
  cloud=np.concatenate([gt[:2],p0[:2],p1[:2]],axis=0).reshape(-1,3);center=(cloud.max(0)+cloud.min(0))/2;radius=max(float(np.ptp(cloud,axis=0).max())*.55,1e-3)
  fig=plt.figure(figsize=(12,9.4));parents=z1['parents']
  for row in range(min(2,len(gt))):
   for col,(title,pose) in enumerate([('Reference',gt[row]),('Native: no SE3',p0[row]),(f'Native: {count} sparse SE3',p1[row])]):
    ax=fig.add_subplot(2,3,row*3+col+1,projection='3d')
    for j,pa in enumerate(parents):
     if pa<0:continue
     if col:
      a=gt[row,[pa,j]];ax.plot(a[:,0],a[:,2],a[:,1],color='#82949e',alpha=.3,lw=1)
     a=pose[[pa,j]];ax.plot(a[:,0],a[:,2],a[:,1],color='#186d9b' if col==0 else '#d46c44',lw=1.8)
    ix=z1['condition_indices'][row,z1['condition_valid'][row]]
    if len(ix):a=gt[row,ix];ax.scatter(a[:,0],a[:,2],a[:,1],c='#279566',s=35,marker='x',label='Requested joint')
    for setter,v in zip([ax.set_xlim,ax.set_ylim,ax.set_zlim],center[[0,2,1]]):setter(v-radius,v+radius)
    ax.set_box_aspect([1,1,1]);ax.view_init(18,-60)
    from matplotlib.ticker import MaxNLocator
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.set_major_locator(MaxNLocator(nbins=3))
    ax.tick_params(labelsize=7,pad=1);ax.set_xlabel('X',fontsize=8,labelpad=1);ax.set_ylabel('Z',fontsize=8,labelpad=1);ax.set_zlabel('Y',fontsize=8,labelpad=1)
    ax.set_title(f'{title}\nrow {z1["rows"][row]}',fontsize=10)
  display={'unitree_g1_29dof':'Unitree G1 (29 DOF)','e457402d4b2e4681b5a5f88b747aea4a':'UniMate quadruped (seen body)','79fed889082e4a25ad92be75b57f641c':'UniMate quadruped (held-out body)'}.get(name,name)
  fig.suptitle(display+' | root-centered FK | '+unit,fontsize=12)
  fig.text(.5,.015,'Green crosses are requested skeleton-joint constraints. No refinement, world anchor, simulation or hardware claim.',ha='center',fontsize=9)
  fig.subplots_adjust(left=.04,right=.96,bottom=.08,top=.90,hspace=.34,wspace=.14);path=run/f'paired_{name}.png';fig.savefig(path,dpi=160);plt.close(fig);plotpaths.append(path.name)
 history=[json.loads(s) for s in (run/'train.jsonl').read_text().splitlines() if s]
 if history:
  fig,axes=plt.subplots(1,3,figsize=(13,3.6))
  for ax,metric in zip(axes,['loss','rotation_deg','position_over_reach']):
   for domain in ['xwbc','unimate']:
    rows=[r for r in history if r['domain']==domain];x=np.array([r['step'] for r in rows]);y=np.array([(r['terms']['total' if metric=='loss' else metric] if 'terms' in r else r[metric]) for r in rows]);n=min(25,len(y))
    if n:ax.plot(x[n-1:],np.convolve(y,np.ones(n)/n,mode='valid'),label=domain)
   ax.set_xlabel('Optimizer step');ax.set_title(metric);ax.legend(fontsize=8);ax.grid(alpha=.2)
  fig.suptitle('Training losses, 25-observation moving means; heldout results are reported separately',fontsize=11);fig.tight_layout()
  path=run/'training_curves.png';fig.savefig(path,dpi=160);plt.close(fig);plotpaths.append(path.name)
 write_json(run/'plot_manifest.json',{'plots':plotpaths,'visual_review_status':'pending_visual_inspection'})
 return plotpaths

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args()
 print(json.dumps({'summary':paired_summary(args.run)['status'],'plots':plot_outputs(args.run)},indent=2))
