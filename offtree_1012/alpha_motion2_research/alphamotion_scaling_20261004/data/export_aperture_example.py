#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('/home/arenalabs/Desktop')/'"be water, robot"'/'alpha_motion2/research/alphamotion_scaling_20261004/data'
def main():
 m=json.loads((D/'payloads/relay_franka_full_v2.json').read_text());ep=next(e for e in m['episodes'] if e['split']==1);a,b=ep['row_range']
 with np.load(m['payload_path']) as z:
  rows=np.arange(a,b);rows=rows[z['frame_index'][rows]%40==0];t=z['timestamp_seconds'][rows];q=z['recorded_robot_qpos'][rows,7:9];opening=z['aperture_m'][rows]
 with (D/'aperture_eval_example.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['time_seconds','left_slide_m','right_slide_m','finger_origin_aperture_m']);w.writerows(zip(t,q[:,0],q[:,1],opening))
 fig,axs=plt.subplots(2,1,figsize=(8,5),sharex=True,layout='constrained')
 axs[0].plot(t,q[:,0]*1000,label='left recorded slide');axs[0].plot(t,q[:,1]*1000,label='right recorded slide');axs[0].axhline(40,color='gray',linestyle='--',label='MJCF upper limit');axs[0].set_ylabel('Slide travel (mm)');axs[0].legend(fontsize=8)
 axs[1].plot(t,opening*1000,color='#2764a0');axs[1].set_ylabel('Finger-origin opening (mm)');axs[1].set_xlabel('Recorded simulation time from episode start (s)')
 for ax in axs:ax.grid(alpha=.2)
 fig.suptitle('Recorded Franka simulation — held-out complete episode\nActual qpos / geometry; no model prediction',fontsize=11)
 fig.savefig(D/'aperture_eval_example.png',dpi=160);plt.close(fig)
 meta={'episode':ep['member'],'source_group':ep['source_group'],'split':'eval','stride':40,'frames':len(t),'geometry_definition':m['aperture_definition'],'source_payload_sha256':m['payload_sha256'],'csv_sha256':hashlib.sha256((D/'aperture_eval_example.csv').read_bytes()).hexdigest(),'prediction_included':False}
 (D/'aperture_eval_example.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))
if __name__=='__main__':main()
