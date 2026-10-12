"""Optional file plots from measured records; no data/model generation."""
from pathlib import Path
import math
from curve_statistics import training_curve,refinement_curve


def plot_training(logs,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for label,path in logs.items():
        rows=training_curve(path)
        axes[0].plot([r['step'] for r in rows],[r['loss'] for r in rows],label=label)
        timed=[r for r in rows if r['training_elapsed_seconds'] is not None]
        if timed:axes[1].plot([r['training_elapsed_seconds'] for r in timed],[r['loss'] for r in timed],label=label)
    for ax,xlabel in zip(axes,['Optimizer step','Measured training elapsed seconds']):
        ax.set_xlabel(xlabel);ax.set_ylabel('Recorded total objective');ax.grid(alpha=.2)
        if ax.lines:ax.legend()
    fig.suptitle('Measured training curves — mixed objectives are not a universal quality score')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=160);plt.close(fig);return str(output)


def plot_refinement(results,output,metric='selected_position_mean'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=refinement_curve(results,metric);units={r['unit'] for r in rows}
    if len(units)!=1:raise ValueError('do not mix metric and nonmetric axes')
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for case_id in results:
        selected=[r for r in rows if r['case_id']==case_id]
        # None is an explicit curve gap; failed/difficult cases are never changed to0.
        y=[float('nan') if r['value'] is None else r['value'] for r in selected]
        axes[0].plot([r['step'] for r in selected],y,label=case_id)
        axes[1].plot([r['wall_seconds'] for r in selected],y,label=case_id)
    for ax,xlabel in zip(axes,['Refinement step (0 = no refinement)','Instrumented seconds incl. diagnostics/projection observers']):
        ax.set_xlabel(xlabel);ax.set_ylabel(metric+' ['+next(iter(units))+']');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Same trajectory, one optimizer process per protocol; failures remain gaps')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=160);plt.close(fig);return str(output)
