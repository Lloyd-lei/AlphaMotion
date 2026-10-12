"""Page-width figures from completed quantitative reports, no model calls."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUNS=[f'{v}_{s}_seed0' for s in 'SML' for v in 'AB']
COLORS={'S':'#008f83','M':'#4169bd','L':'#b04475'}
NAMES={'fourier_gr3':'Fourier GR3','kuavo_s45':'Kuavo S45','unitree_h2':'XWBC H2','unitree_r1':'Unitree R1',
       '54e7e37365ac4063ba2ba65151fd9f5d':'UniMate serpentine',
       '669a750957bbceb71e6ec4a5_fbx':'UniMate animated hand',
       '669a76c557bbceb71e7106c6_fbx':'UniMate four-link chain',
       '79fed889082e4a25ad92be75b57f641c':'UniMate quadruped'}

def read(p):return json.loads(p.read_text())
def save(fig,path):
    fig.savefig(path,dpi=190,bbox_inches='tight',pad_inches=.09);plt.close(fig)
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
    base=root/'evaluation_phase2_v2/reports';out=base/'document_figures';out.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.titlesize':9,'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8})
    loco=read(base/'locomotion/locomotion_comparison.json');artifacts=[]
    for dataset in ['XWBC','UniMate']:
        for mode in ['full_source','pure_se3']:
            rows=[r for r in loco['rows'] if r['dataset']==dataset and r['input_mode']==mode and r['clip_exposure']=='heldout_clip' and r['body_exposure']=='heldout_body']
            bodies=sorted({r['body'] for r in rows})
            if len(bodies)!=4:raise ValueError('Expected all four heldout bodies in each dataset')
            metric='position' if dataset=='XWBC' else 'position_over_reach';mult=10 if dataset=='XWBC' else 1
            fig,axes=plt.subplots(2,2,figsize=(6.35,4.85),layout='constrained')
            for ax,body in zip(axes.flat,bodies):
                body_rows=[r for r in rows if r['body']==body]
                for run in RUNS:
                    for phase in [1,2]:
                        points=sorted([r for r in body_rows if r['run']==run and (r['step']<=3000)==(phase==1)],key=lambda r:r['step'])
                        ax.plot([r['step'] for r in points],[np.nan if r['metrics'][metric]['mean'] is None else r['metrics'][metric]['mean']*mult for r in points],
                            color=COLORS[run[2]],linestyle='-' if run[0]=='A' else '--',marker='o',markersize=2,
                            label=run.replace('_seed0','') if phase==1 else None,alpha=1 if phase==1 else .65)
                ax.axvline(3000,color='#555555',linewidth=.6);ax.set(title=NAMES[body]+f" | {body_rows[0]['physical_clips']} clips",xlabel='Optimizer steps',
                    ylabel='Joint position error (mm)' if dataset=='XWBC' else 'Position error / reach');ax.set_ylim(bottom=0);ax.set_xticks([0,3000,6000]);ax.grid(alpha=.2)
            handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,ncols=6,loc='outside lower center',frameon=False,handlelength=1.5,columnspacing=.8)
            title=('Paired source-to-target mapping' if dataset=='XWBC' else 'Same-body reconstruction') if mode=='full_source' else 'Sparse SE(3): full-body posture diagnostic'
            fig.suptitle(dataset+' heldout bodies | '+title+'\nPhase 2 includes LR restart + derivative correction',fontsize=9)
            artifacts.append(save(fig,out/f'heldout_{dataset}_{mode}.png'))
    fig,axes=plt.subplots(2,3,figsize=(6.35,5.05),layout='constrained')
    for row,(body,title) in enumerate([('unitree_g1_29dof','Seen G1'),('kuavo_s45','Heldout Kuavo')]):
        rows=[r for r in loco['rows'] if r['body']==body and r['input_mode']=='full_source' and r['clip_exposure']=='heldout_clip']
        for col,(field,label,scale) in enumerate([('step','Optimizer steps',1),('training_seconds','Training minutes',1/60),('estimated_dense_flops','Dense PFLOPs (est.)',1e-15)]):
            ax=axes[row,col]
            for run in RUNS:
                for phase in [1,2]:
                    points=sorted([r for r in rows if r['run']==run and (r['step']<=3000)==(phase==1)],key=lambda r:r['step'])
                    ax.plot([r[field]*scale for r in points],[np.nan if r['metrics']['position']['mean'] is None else r['metrics']['position']['mean']*10 for r in points],
                        color=COLORS[run[2]],linestyle='-' if run[0]=='A' else '--',marker='o',markersize=2,alpha=1 if phase==1 else .65,label=run.replace('_seed0','') if phase==1 else None)
            ax.set(title=title,xlabel=label);ax.set_ylim(bottom=0);ax.grid(alpha=.2)
            if col==0:ax.set_ylabel('Joint error (mm)');ax.axvline(3000,color='#555555',linewidth=.6);ax.set_xticks([0,3000,6000])
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,ncols=6,loc='outside lower center',frameon=False,handlelength=1.5,columnspacing=.8)
    fig.suptitle('Paired XWBC mapping | Fixed heldout motion cases\nObserved cost; phase2 includes LR restart + numerical repair',fontsize=10)
    artifacts.append(save(fig,out/'mapping_steps_time_compute.png'))
    for protocol in ['P0','P1']:
        report=read(base/f'manipulation/{protocol}_step6000/manipulation_comparison.json')
        fig,axes=plt.subplots(2,3,figsize=(6.35,5.15),layout='constrained')
        for row,body in enumerate(['adroit_pen','relay_franka_full_v2']):
            group=next(g for g in report['groups'] if g['body']==body and g['reference_cohort']=='strict_limit_valid_control')
            for col,(metric,mult,label) in enumerate([('selected_position_cm',10,'Endpoint error (mm)'),('selected_rotation_degrees',1,'Orientation error (deg)'),('target_frame_fraction',100,'All targets passed (%)')]):
                ax=axes[row,col]
                for run,stages in group['curves'].items():
                    data=stages['constrained_refinement'][metric]['mean']
                    if data is not None:ax.plot(report['steps'],np.array(data)*mult,color=COLORS[run[2]],linestyle='-' if run[0]=='A' else '--',label=run.replace('_seed0',''))
                default=group['curves'][RUNS[0]]['default_valid_q'][metric]['mean']
                if default is not None:ax.plot(report['steps'],np.array(default)*mult,color='#555555',linestyle=':',label='Default q')
                ax.set(title=('Adroit' if row==0 else 'Franka')+'\n'+label,xlabel='Refinement updates');ax.set_ylim(bottom=0);ax.grid(alpha=.2)
                if col==2:ax.set_ylim(0,102)
                if col==0:ax.set_ylabel('Mean over 3 source episodes')
        handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,ncols=4,loc='outside lower center',frameon=False)
        fig.suptitle(protocol+' | Six models at 6,000 training steps\nNative-q refinement, valid-reference controls; success = 1 cm AND 5 degrees',fontsize=10)
        artifacts.append(save(fig,out/f'native_q_{protocol}_6000.png'))
    (out/'render_receipt.json').write_text(json.dumps(dict(artifacts=artifacts,visual_review='pending',source_data='Unchanged full quantitative summaries; no filtered success cases'),indent=2)+'\n')
    print(json.dumps(dict(output=str(out),figures=len(artifacts))))

if __name__=='__main__':main()
