"""Read-only live experiment aggregation; never loads a model or starts training."""
from pathlib import Path
import argparse,collections,datetime,hashlib,json,math
import numpy as np

ORDER=['A_S_seed0','B_S_seed0','A_M_seed0','B_M_seed0','A_L_seed0','B_L_seed0']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()

def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def read_logs(path):
    if not path.exists():return
    with path.open() as f:
        for line in f:
            if not line.endswith('\n'):break
            if line.strip():yield json.loads(line)

def curves(rows):
    by=collections.defaultdict(list)
    for r in rows:by[(r['domain'],(r['step']-1)//100)].append(r)
    out=[]
    for (domain,block),rs in sorted(by.items()):
        terms={key:float(np.mean([r['terms'][key] for r in rs if key in r['terms']])) for key in ['rotation_deg','position_over_reach','condition_rotation_deg','condition_position_over_reach','slide_normalized_l1','surface_position','surface_normal'] if any(key in r['terms'] for r in rs)}
        out.append({'domain':domain,'step_block':block,'step_start':rs[0]['step'],'step_end':rs[-1]['step'],'updates':len(rs),'mean_step':float(np.mean([r['step'] for r in rs])),
            'mean_training_seconds':float(np.mean([r['cumulative_train_seconds'] for r in rs])),
            'mean_estimated_dense_flops':float(np.mean([r['cumulative_estimated_training_flops'] for r in rs])),'terms':terms})
    return out

def exact_eval_signature(folder,document):
    h=hashlib.sha256()
    for c in document['cases']:
        h.update(json.dumps({k:c[k] for k in ['source','target','mode','condition_count','body_exposure']},sort_keys=True).encode())
        with np.load(folder/c['error_file'],allow_pickle=False) as z:
            for key in ['rows','groups','frame_index','selected']:
                value=np.ascontiguousarray(z[key]);h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(value.tobytes())
    return h.hexdigest()

def collect(root):
    result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':{},'comparisons':{},'scope':'Real optimizer records and checkpoint evaluation only; no new inference; single seed, finite budgets, no scaling-law claim'}
    samples={};points=[]
    for name in ORDER:
        run=root/'runs'/name;rows=list(read_logs(run/'train.jsonl'));samples[name]=[r['sampling_sha256'] for r in rows]
        cfg=json.loads((run/'config.json').read_text()) if (run/'config.json').exists() else {}
        rec={'status':'not_started','steps':len(rows),'curve':curves(rows),'config_sha256':sha(run/'config.json') if cfg else None}
        if rows:
            last=rows[-1]
            if [r['step'] for r in rows]!=list(range(1,last['step']+1)):raise ValueError('nonconsecutive optimizer log '+name)
            rec.update(status='running',steps=last['step'],training_seconds=last['cumulative_train_seconds'],wall_seconds=last['cumulative_wall_seconds'],estimated_dense_flops=last['cumulative_estimated_training_flops'],tokens=last['cumulative_tokens'],sampling_sequence_sha256=hashlib.sha256('\n'.join(samples[name]).encode()).hexdigest(),domain_updates=dict(collections.Counter(r['domain'] for r in rows)))
            valid=collections.Counter();dataset_rows=collections.Counter();mode_rows=collections.Counter()
            for item in rows:
                for direction in item['directions']:
                    valid.update(direction['valid_loss_samples']);dataset_rows[direction['actual_target_dataset']]+=direction['rows'];mode_rows[direction['mode']]+=direction['rows']
            rec.update(valid_loss_sample_totals=dict(valid),target_dataset_direction_rows=dict(dataset_rows),mode_direction_rows=dict(mode_rows),exposure_count_scope='Supervised direction occurrences, including repeated frames across paired directions; not unique source frames; auxiliary tokens counted separately')
        for filename,key in [('result.json','result'),('failure.json','failure'),('exit_status.json','exit')]:
            if (run/filename).exists():rec[key]=json.loads((run/filename).read_text())
        if rec.get('result',{}).get('status')=='completed':rec['status']='completed'
        elif 'failure' in rec:rec['status']='failed'
        evals=[]
        for path in sorted((run/'eval').glob('evaluation_*.json')):
            d=json.loads(path.read_text());label=d['label'];step=0 if label=='initial' else cfg.get('steps',3000) if label=='final' else int(label.removeprefix('step'))
            signature=exact_eval_signature(path.parent,d);r=rows[step-1] if step and step<=len(rows) else None
            node={'run':name,'label':label,'step':step,'evaluation_set_sha256':signature,'training_seconds':r['cumulative_train_seconds'] if r else 0,'estimated_dense_flops':r['cumulative_estimated_training_flops'] if r else 0,'cases':d['cases'],'file':str(path),'sha256':sha(path)}
            evals.append(node);points.append(node)
        rec['evaluations']=evals
        rec['serialized_files_bytes']=sum(p.stat().st_size for p in run.rglob('*') if p.is_file()) if run.exists() else 0
        result['runs'][name]=rec
    started=[n for n in ORDER if samples[n]]
    min_steps=min((len(samples[n]) for n in started),default=0)
    mismatch=[{'run':n,'first_step':next((i+1 for i,(a,b) in enumerate(zip(samples[started[0]][:min_steps],samples[n][:min_steps])) if a!=b),None)} for n in started[1:]]
    result['comparisons']['actual_shared_sampling']={'runs':started,'shared_optimizer_steps':min_steps,'mismatches':[x for x in mismatch if x['first_step'] is not None],'all_six_available':len(started)==6,'meaning':'Exact source/target/rows/mask/condition validity/contact validity and auxiliary sampling choices; B extra patch/loss compute separately accounted'}
    common=[]
    labels=sorted(set(x['label'] for x in points))
    for label in labels:
        group=[x for x in points if x['label']==label]
        common.append({'label':label,'runs':[x['run'] for x in group],'all_six_available':len(group)==6,'same_exact_evaluation_rows_masks':len({x['evaluation_set_sha256'] for x in group})==1,'evaluation_set_sha256':group[0]['evaluation_set_sha256']})
    result['comparisons']['evaluation_nodes']=common
    # Use earlier measured snapshots only. No interpolation or nearest-future claims.
    for axis in ['training_seconds','estimated_dense_flops']:
        available={n:[x for x in points if x['run']==n and x['label'].startswith('step')] for n in ORDER}
        if all(available.values()):
            budget=min(max(p[axis] for p in v) for v in available.values());selected=[]
            for n,v in available.items():
                earlier=[p for p in v if p[axis]<=budget]
                if not earlier:selected.append({'run':n,'status':'no_saved_point_under_budget'});continue
                p=max(earlier,key=lambda x:x[axis]);gap=(budget-p[axis])/budget
                selected.append({'run':n,'label':p['label'],'actual_budget':p[axis],'relative_shortfall':gap,'status':'matched_within_1pct' if gap<=.01 else 'earlier_under_budget_not_equal','evaluation_set_sha256':p['evaluation_set_sha256']})
            result['comparisons'][axis]={'common_covered_budget':budget,'selected':selected,'estimated_not_measured_flops':axis=='estimated_dense_flops'}
    return result

def plot(summary,path):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    colors={'S':'#008f83','M':'#4169bd','L':'#b04475'}
    for name,r in summary['runs'].items():
        series=[p for p in r['curve'] if p['domain']=='xwbc']
        if not series:continue
        color=colors[name[2]];style='-' if name[0]=='A' else '--'
        for col,(axis,label,scale) in enumerate([('mean_step','Optimizer steps',1),('mean_training_seconds','Training GPU minutes',1/60),('mean_estimated_dense_flops','Estimated dense training TFLOPs',1e-12)]):
            for row,(key,yl) in enumerate([('position_over_reach','Shared joint position error / static reach'),('rotation_deg','Shared joint rotation error (degrees)')]):
                points=[p for p in series if key in p['terms']];axes[row,col].plot([p[axis]*scale for p in points],[p['terms'][key] for p in points],label=name,color=color,linestyle=style);axes[row,col].set(xlabel=label,ylabel=yl);axes[row,col].grid(alpha=.2)
    axes[0,0].legend(ncols=2,fontsize=8);fig.suptitle('Six from-scratch runs: actual XWBC training records\n100-step bins within domain; A solid / B dashed; shared terms only, not total-loss comparison',fontsize=13)
    fig.savefig(path,dpi=160);plt.close(fig)

def plot_native(summary,path,manifest):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scales={b['body']:b['normalization_reach']*10 for b in manifest['bodies'] if b['metric']}
    panels={
        'locomotion':[
            ('unitree_g1_29dof','full',0,'all_position_mm','G1: full source mapping, heldout clips',1),
            ('kuavo_s45','full',0,'all_position_mm','Kuavo: heldout body, mixed source clips',1)],
        'manipulation':[
            ('adroit_pen','sparse_only',2,'selected_position_over_reach','Adroit: only 2 sparse SE(3) conditions',scales['adroit_pen']),
            ('relay_franka_full_v2','sparse_only',2,'selected_position_over_reach','Franka: only 2 sparse SE(3) conditions',scales['relay_franka_full_v2'])]}
    colors={'S':'#008f83','M':'#4169bd','L':'#b04475'}
    for family,rows in panels.items():
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        for row,(body,mode,count,metric,title,multiplier) in enumerate(rows):
            for name,r in summary['runs'].items():
                nodes=[]
                for node in r['evaluations']:
                    if not node['label'].startswith('step'):continue
                    case=next((c for c in node['cases'] if c['target']==body and c['mode']==mode and c['condition_count']==count),None)
                    if case and case['metrics'].get(metric,{}).get('mean') is not None:nodes.append((node,case['metrics'][metric]['mean']*multiplier))
                nodes.sort(key=lambda x:x[0]['step'])
                for col,(axis,label,factor) in enumerate([('step','Optimizer steps',1),('training_seconds','Training GPU minutes',1/60),('estimated_dense_flops','Estimated dense training TFLOPs',1e-12)]):
                    if nodes:axes[row,col].plot([x[0][axis]*factor for x in nodes],[x[1] for x in nodes],marker='o',markersize=3,linestyle='-' if name[0]=='A' else '--',color=colors[name[2]],label=name)
                    axes[row,col].set(xlabel=label,ylabel=('Selected endpoint' if family=='manipulation' else 'All-joint position')+' error (mm)',title=title);axes[row,col].grid(alpha=.2)
            for ax in axes[row]:
                ax.relim();ax.autoscale(enable=True,axis='y');ax.set_ylim(bottom=0)
        axes[0,0].legend(ncols=2,fontsize=8)
        fig.suptitle('Unrefined checkpoint evaluation: '+family+'\nSame exact rows and conditions at each saved node; straight segments connect measured points only',fontsize=13)
        fig.savefig(path.with_name(path.stem+'_'+family+'.png'),dpi=160);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--plot',action='store_true');a=p.parse_args();s=collect(a.root);write(a.output,s)
    if a.plot:
        plot(s,a.output.with_suffix('.png'))
        plot_native(s,a.output,json.loads((a.root/'data_manifest.json').read_text()))
    print(json.dumps({'output':str(a.output),'runs':{n:{k:r.get(k) for k in ['status','steps','training_seconds','serialized_files_bytes']} for n,r in s['runs'].items()},'sampling':s['comparisons']['actual_shared_sampling']}))

if __name__=='__main__':main()
