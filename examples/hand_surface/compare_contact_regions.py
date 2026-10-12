from pathlib import Path
import os
os.environ['MPLBACKEND']='Agg'
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
import numpy as np,json
B=Path('/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/hand_surface_case_20261004/derived')
plt.rcParams.update({'font.size':10,'svg.fonttype':'none','axes.labelcolor':'#465360','text.color':'#253440'})
fig=plt.figure(figsize=(13,6.5));fig.suptitle('Same object frame; different surface coverage',fontsize=20,x=.07,ha='left',y=.97)
fig.text(.07,.895,'Constructed geometric illustration  ·  sphere radius 30 mm  ·  proximity threshold 2 mm',fontsize=11,color='#576876')
record={'object_radius_m':.03,'classification':'derived geometric comparison; no learned correspondence','region_grouping':'Illustrative object Y hemisphere partition; multiple Allegro fingertip patches may share one region. It is not a demonstrated map or equal contact coverage.','d_definition':'body_surface - sphere_nearest','bodies':{}}
for k,label in enumerate(['allegro','panda']):
 trace=json.loads((B/(label+'_trace.json')).read_text());z=np.load(B/(label+'_surface_samples.npz'));ax=fig.add_subplot(1,2,k+1,projection='3d');ax.set_position([.05+.48*k,.17,.44,.65]);u=np.linspace(0,2*np.pi,45);v=np.linspace(0,np.pi,25);ax.plot_wireframe(30*np.outer(np.cos(u),np.sin(v)),30*np.outer(np.sin(u),np.sin(v)),30*np.outer(np.ones_like(u),np.cos(v)),color='#b4cfde',alpha=.22,linewidth=.4)
 names=['ff_tip','mf_tip','rf_tip','th_tip'] if label=='allegro' else ['left_finger','right_finger'];colors=['#d67136','#267fad','#7c69b5','#3f9778'];counts={};regions={}
 for n,col in zip(names,colors):
  ids=[a['body_id'] for a in trace['sampling']['bindings'] if a['body_name']==n];mask=np.isin(z['body_id'],ids)&z['contact_geometry'];pts=z['object_nearest_point_m'][mask]*1000
  ax.scatter(*pts.T,s=10,color=col,alpha=.9,label=n);counts[n]=len(pts);regions[n]={'negative_object_y':int((pts[:,1]<0).sum()),'positive_object_y':int((pts[:,1]>=0).sum())}
 ax.set(xlim=(-35,35),ylim=(-35,35),zlim=(-35,35),xlabel='Object X (mm)',ylabel='Object Y (mm)',zlabel='Object Z (mm)');ax.set_box_aspect((1,1,1));ax.view_init(elev=23,azim=35);ax.set_title('Allegro V3 right · four fingers' if label=='allegro' else 'Panda · one symmetric opening',fontsize=13,pad=8);ax.legend(loc='upper left',bbox_to_anchor=(0,.97),frameon=False,fontsize=9);ax.grid(alpha=.15)
 fig.canvas.draw();pix=[]
 for p in trace['selected_surface_points']:
  v3=np.array(p['object_nearest_point_m'])*1000;a,b,_=proj3d.proj_transform(*v3,ax.get_proj());p2=ax.transData.transform([a,b]);pix.append({'body':p['body'],'object_point_m':p['object_nearest_point_m'],'pixel_png_180dpi':[float(p2[0]*1.8),float((fig.bbox.height-p2[1])*1.8)]})
 record['bodies'][label]={'patch_sample_counts':counts,'hemisphere_counts':regions,'projection_matrix':ax.get_proj().tolist(),'selected_point_pixels':pix}
fig.text(.07,.07,'Orange/blue/green/purple points are nearest points on the analytic sphere from sampled robot surfaces.',fontsize=10)
fig.text(.07,.037,'Shared object regions can combine several body patches; neither forces nor a one-to-one finger map are implied.',fontsize=10,color='#5e6e78')
fig.savefig(B/'object_frame_contact_comparison.png',dpi=180,facecolor='white');fig.savefig(B/'object_frame_contact_comparison.svg',facecolor='white');plt.close(fig)
(B/'object_frame_contact_comparison.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v['patch_sample_counts'] for k,v in record['bodies'].items()}))
