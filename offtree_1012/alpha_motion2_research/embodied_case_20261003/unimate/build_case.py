#!/usr/bin/env python3
"""CPU-only official preprocessing and measured diagrams; no neural inference."""
import ast, datetime as dt, hashlib, json, pickle, subprocess, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from mpl_toolkits.mplot3d import proj3d
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2]; CODE=BASE/'external/UniMate'
DR='053e8b4b06b6a8ba717a030e675dd5b7fd5be92e'; MR='971da7cfc1c8d99c2af6c00be9d2ed5700f99073'
DATA=Path('/home/arenalabs/.cache/huggingface/hub/datasets--Linzhan--UniML3D/snapshots')/DR
MODEL=Path('/home/arenalabs/.cache/huggingface/hub/models--Linzhan--UniMate/snapshots')/MR/'unimate_mixamo_f60'
CLIP=DATA/'features/mixamo/motions/Jumping_Up-000.npz'; COND=DATA/'features/mixamo/cond.npy'; CAP=DATA/'features/mixamo/captions.json'; STATS=MODEL/'dataset_stats.npy'; CONFIG=MODEL/'config.json'
F=[0,3,6]; JSEL=9

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(8*1024**2),b''):h.update(c)
 return h.hexdigest()

class RestrictedNumpy(pickle.Unpickler):
 def find_class(self,m,n):
  allowed={(x,y) for x in ['numpy._core.multiarray','numpy.core.multiarray'] for y in ['_reconstruct','scalar']}|{('numpy','ndarray'),('numpy','dtype')}
  if (m,n) not in allowed:raise ValueError((m,n))
  return super().find_class(m,n)

def ndict(p):
 with Path(p).open('rb') as f:
  np.lib.format._read_array_header(f,np.lib.format.read_magic(f));return RestrictedNumpy(f).load().item()

def funcs(p,names,env):
 nodes=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and (names is None or n.name in names)]
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),env)
 return {n.name:{'path':str(p),'line_start':n.lineno,'line_end':n.end_lineno} for n in nodes}

sys.path.insert(0,str(HERE/'reference_dependency'))
from Quaternions import Quaternions as Q
from Animation import offsets_from_positions
E={'np':np,'Quaternions':Q}; refs=funcs(CODE/'unimate/utils/rotation_conversions.py',['rotation_6d_to_matrix_np'],E)
refs.update(funcs(CODE/'unimate/utils/motion_utils.py',None,E));refs.update(funcs(CODE/'unimate/dataset/transforms.py',['apply_cropping','apply_normalization','apply_padding','build_parent_features'],E))
c=ndict(COND)['mixamo']; stats=ndict(STATS)['mixamo']; config=json.loads(CONFIG.read_text());caption=json.loads(CAP.read_text())['Jumping_Up-000']
with np.load(CLIP,allow_pickle=False) as d:original=d['global_positions'].copy();local=d['local_rotations'].copy();facing=d['root_facing_quat'].copy();fps=int(d['fps'])
parents=np.asarray(c['parents']);names=c['joint_names'];clean=c['clean_joint_names'];tpose=np.asarray(c['tpos_first_frame']).copy();tpose[:,1]-=tpose[:,1].min()
pos=original.copy();ground=float(pos[...,1].min());pos[...,1]-=ground
reference=np.broadcast_to(np.asarray(c['tpos_local_rotations'])[None],local.shape).copy();rebased=E['compute_rots_from_tpos'](Q(reference),Q(local),parents)
features=E['compute_unimate_motion_feats'](pos,rebased,parents,Q(facing));window,start=E['apply_cropping'](features,'tpos',60,start_idx=0)
J,D=features.shape[1:];mean=np.empty((J,D));std=np.empty((J,D));mean[0],std[0]=stats['mean_root'],stats['std_root'];mean[1:],std[1:]=stats['mean_local'],stats['std_local']
normalized=E['apply_normalization'](window,mean,std);padded,valid=E['apply_padding'](normalized,60);batch=padded.transpose(1,2,0)[None].astype(np.float32);mask=np.arange(60)<valid
identity6=Q.id(1).rotation_matrix(cont6d=True)[0];tpose12=np.concatenate([tpose,np.tile(identity6,(J,1)),np.zeros((J,3))],axis=-1);tnorm=E['apply_normalization'](tpose12,mean,std)
offsets=offsets_from_positions(tpose,parents);ric=E['recover_unimate_joint_pos_from_ric'](features);fk=E['recover_unimate_joint_pos_from_rot'](features,parents,offsets)
qc={'finite_features':bool(np.isfinite(features).all()),'ric_max_abs_error_body_units':float(np.abs(ric-pos[:-1]).max()),'fk_max_abs_error_body_units':float(np.abs(fk-pos[:-1]).max()),'normalization_roundtrip_max_abs_error':float(np.abs(normalized*std+mean-window).max()),'raw_feature_shape':list(features.shape),'model_motion_array_shape':list(batch.shape),'padded_frames_exact_zero':bool(np.all(padded[valid:]==0))}
assert features.shape==(7,22,12) and qc['finite_features']
assert qc['ric_max_abs_error_body_units']<1e-4 and qc['fk_max_abs_error_body_units']<1e-4
np.savez_compressed(HERE/'derived_preprocessing.npz',source_global_positions=original,grounded_global_positions=pos,parents=parents,tpose=tpose,raw_local_quaternions_wxyz=local,tpose_relative_quaternions_wxyz=rebased.qs,root_facing_quaternions_wxyz=facing,features_12d=features,normalized_12d=normalized,model_motion_array_bjdt=batch,time_mask=mask,normalization_mean=mean,normalization_std=std,tpose_12d=tpose12,tpose_normalized_12d=tnorm,recovered_ric=ric,recovered_fk=fk,fps=np.asarray(fps),selected_frames=np.asarray(F),selected_joint=np.asarray(JSEL))
# Shareable derived tensors only: original pose/quaternion source arrays stay remote.
np.savez_compressed(HERE/'model_inputs_derived.npz',features_12d=features,normalized_12d=normalized,model_motion_array_bjdt=batch,time_mask=mask,joint_mask=np.ones(J,dtype=bool),tpose_12d=tpose12,tpose_normalized_12d=tnorm,tpose_grounded_xyz=tpose,normalization_mean=mean,normalization_std=std,parents=parents,selected_frames=np.asarray(F),selected_joint=np.asarray(JSEL),fps=np.asarray(fps))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})
fig=plt.figure(figsize=(12,5.4),dpi=200,facecolor='#fcfcfd')
fig.text(.04,.953,'UNIMATE  /  A REAL PREPROCESSING TRACE',fontsize=17,weight='bold',color='#15233d')
fig.text(.04,.908,'Jumping_Up-000  |  22-joint Mixamo skeleton  |  30 fps  |  '+caption,fontsize=9,color='#58677b')
fig.text(.04,.056,'Measured data + official preprocessing. No neural denoising, robot retargeting, or simulation executed.',fontsize=9,color='#526176')
fig.text(.04,.028,'Axes: X / Y-up / Z-forward; lengths are canonical body units (skeleton diameter normalized to 2.0), not verified meters.',fontsize=8.2,color='#657387')
poses=[tpose]+[pos[f] for f in F];labels=['T-pose conditioning','Frame 0  /  0.000 s','Frame 3  /  0.100 s','Frame 6  /  0.200 s'];axes=[];origin=np.array([-.63,.08,.3])
for k,(pose,label) in enumerate(zip(poses,labels)):
 ax=fig.add_axes([.025+k*.244,.21,.235,.66],projection='3d',facecolor='#fcfcfd');axes.append(ax)
 for j,p in enumerate(parents):
  if p>=0:
   v=pose[[p,j]];ax.plot(v[:,0],v[:,2],v[:,1],color='#0c91a1' if j in [3,6,9,14] else '#647895',lw=2.1,alpha=.94)
 ax.scatter(pose[:,0],pose[:,2],pose[:,1],s=13,c='#344969',depthshade=False)
 pt=pose[JSEL];ax.scatter([pt[0]],[pt[2]],[pt[1]],s=74,c='#ef694e',edgecolor='white',linewidth=1,depthshade=False,zorder=10)
 for i,col in enumerate(['#d05a57','#398b66','#3f74b9']):
  end=origin+np.eye(3)[i]*.17;delta=end-origin;ax.quiver(origin[0],origin[2],origin[1],delta[0],delta[2],delta[1],color=col,arrow_length_ratio=.22,lw=1.1);ax.text(end[0],end[2],end[1],'XYZ'[i],color=col,fontsize=6.8)
 ax.set_xlim(-.85,.85);ax.set_ylim(-.55,.55);ax.set_zlim(-.03,1.55);ax.set_box_aspect((1.25,.6,1.35));ax.view_init(elev=12,azim=-67);ax.set_axis_off();ax.set_title(label,pad=-4,fontsize=10.6,weight='bold',color='#263a58')
 fig.text(.042+k*.244,.155,'Left Foot  |  j = 9  |  parent = 6',fontsize=8.6,color='#c65542');fig.text(.042+k*.244,.123,'['+', '.join(f'{x:+.4f}' for x in pt)+']',fontsize=8,fontfamily='DejaVu Sans Mono',color='#526176')
fig.canvas.draw();width,height=fig.canvas.get_width_height();pixels={'image':'skeleton_tpose_motion.png','width_px':width,'height_px':height,'pixel_origin':'top-left','plot_axis_order':['source_X','source_Z','source_Y'],'panels':[]}
for k,(ax,pose) in enumerate(zip(axes,poses)):
 def project(pt):
  x,y,z=proj3d.proj_transform(pt[0],pt[2],pt[1],ax.get_proj());xx,yy=ax.transData.transform((x,y));return [float(xx),float(height-yy)]
 xy=project(pose[JSEL]);x0=max(0,int(xy[0])-135);y0=max(0,int(xy[1])-135);x1=min(width,x0+270);y1=min(height,y0+270)
 panel={'label':labels[k],'source_frame_index':None if k==0 else F[k-1],'joint_index':JSEL,'joint_name':names[JSEL],'joint_source_xyz_body_units':pose[JSEL].tolist(),'joint_pixel_xy':xy,'projection_matrix_plot_coordinates':ax.get_proj().tolist(),'axes':{n:{'source_origin_xyz':origin.tolist(),'source_endpoint_xyz':(origin+np.eye(3)[i]*.17).tolist(),'origin_pixel_xy':project(origin),'endpoint_pixel_xy':project(origin+np.eye(3)[i]*.17)} for i,n in enumerate('XYZ')},'crop_bbox_xyxy':[x0,y0,x1,y1],'joint_pixel_xy_in_crop':[xy[0]-x0,xy[1]-y0],'crop_file':'left_foot_tpose_crop.png' if k==0 else f'left_foot_frame_{F[k-1]:02d}_crop.png'}
 pixels['panels'].append(panel)
fig.savefig(HERE/'skeleton_tpose_motion.png',dpi=200);fig.savefig(HERE/'skeleton_tpose_motion.svg')
for panel in pixels['panels']:
 x0,y0,x1,y1=panel['crop_bbox_xyxy'];fig.savefig(HERE/panel['crop_file'],dpi=200,bbox_inches=Bbox.from_bounds(x0/200,(height-y1)/200,(x1-x0)/200,(y1-y0)/200),pad_inches=0)
# Record the actual raster dimensions: floating bbox-to-pixel truncation may yield 269 rather than270 px.
import struct
for panel in pixels['panels']:
 panel['crop_rendered_width_px'],panel['crop_rendered_height_px']=struct.unpack('>II',(HERE/panel['crop_file']).read_bytes()[16:24])
 panel['crop_pixel_accuracy_note']='Pixel coordinates use the requested figure bbox; raster rounding is below one pixel.'
plt.close(fig);(HERE/'joint_pixels.json').write_text(json.dumps(pixels,indent=2)+'\n')
fig=plt.figure(figsize=(12,2.45),dpi=200,facecolor='#fcfcfd');fig.text(.04,.89,'LEFT FOOT  /  THE 12 VALUES ACTUALLY COMPUTED',fontsize=14,weight='bold',color='#15233d');fig.text(.04,.79,'Raw before statistics normalization. Rotation channels come from parent LeftLeg (j = 6) after T-pose rebase.',fontsize=9,color='#5b6d83')
ax=fig.add_axes([.04,.18,.92,.51]);ax.axis('off');cols=['frame','p.x','p.y','p.z','R.c0.x','R.c0.y','R.c0.z','R.c1.x','R.c1.y','R.c1.z','delta.x','delta.y','delta.z'];rows=[[str(i)]+[f'{v:+.5f}' for v in features[i,JSEL]] for i in F];table=ax.table(cellText=rows,colLabels=cols,cellLoc='center',loc='center',colWidths=[.05]+[.079]*12);table.auto_set_font_size(False);table.set_fontsize(7.3);table.scale(1,1.6)
for (rr,cc),cell in table.get_celld().items():
 cell.set_edgecolor('#e3e8ef');cell.set_linewidth(.55);cell.set_facecolor('#f0f5f9' if cc==0 else '#e8f4f3' if cc<4 else '#ecf0fb' if cc<10 else '#fcf0e8')
 if rr==0:cell.set_text_props(weight='bold',color='#34435a')
fig.text(.04,.06,'p: root-XZ-relative, facing-aligned, absolute Y height. delta: displacement to the next frame in its facing frame, body units/frame.',fontsize=8.1,color='#576c84');fig.savefig(HERE/'left_foot_12d_values.png',dpi=200);fig.savefig(HERE/'left_foot_12d_values.svg');plt.close(fig)
channels=['rifke_x','rifke_y_absolute_height','rifke_z','parent_tpose_rot_col0_x','parent_tpose_rot_col0_y','parent_tpose_rot_col0_z','parent_tpose_rot_col1_x','parent_tpose_rot_col1_y','parent_tpose_rot_col1_z','delta_x_destination_facing','delta_y_destination_facing','delta_z_destination_facing']
commit=subprocess.check_output(['git','-C',str(CODE),'rev-parse','HEAD'],text=True).strip()
trace={'schema_version':1,'created_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'evidence_label':'derived_from_real_reference_sample','neural_model_executed':False,'robot_retargeting_executed':False,'physics_or_real_robot_executed':False,'sample':{'dataset':'Linzhan/UniML3D','revision':DR,'object_type':'mixamo','clip':'Jumping_Up-000','caption':caption,'fps':fps,'source_frame_count':8,'source_frame_indices':list(range(8)),'source_last_frame_time_seconds':7/fps,'caption_source':str(CAP),'scope':'Released short feature clip, not the complete original Mixamo animation.'},'joint':{'index':JSEL,'raw_name':names[JSEL],'clean_name':clean[JSEL],'parent_index':int(parents[JSEL]),'parent_raw_name':names[parents[JSEL]],'rotation_feature_source_joint':int(parents[JSEL])},'coordinates':{'up':'+Y','canonical_forward':'+Z','quaternion_order':'WXYZ','position_units':'canonical body units; skeleton leaf-to-leaf diameter normalized to2.0; not asserted meters','local_delta_units':'canonical body units/frame, no fps multiplier','rotation_6d_units':'dimensionless','normalized_feature_units':'dimensionless standardized channels','scale_factor_from_cond':float(c['scale_factor']),'ground_height_mode':c['ground_height_mode'],'loader_global_y_subtraction':ground},'preprocessing':{'code_repo':'https://github.com/Friedrich-M/UniMate','code_commit':commit,'dependency':json.loads((HERE/'reference_dependency/revision.json').read_text()),'execution':'Unmodified official numpy function bodies compiled from AST; official Motion quaternion/FK files on local sys.path. No installation, torch or neural execution.','window_start_source_frame':start,'displayed_feature_rows':F,'source_to_feature_relation':'Feature row t uses pose t and delta to t+1. Eight source poses produce7 feature rows; source frame7 has no feature row.','realignment_applied':False,'realignment_reason':'Official loader realigns only if crop_start>0; this sample uses start0.','raw_feature_shape':[7,22,12],'normalized_shape':[7,22,12],'temporal_padded_shape':[60,22,12],'batch_motion_array_shape':[1,22,12,60],'batch_axes':['batch','joint','feature','time'],'time_valid_indices':list(range(7)),'time_padded_indices':list(range(7,60)),'joint_valid_indices':list(range(22)),'normalization_stats_source':str(STATS),'model_config_source':str(CONFIG),'normalization_mean_for_selected_joint':mean[JSEL].tolist(),'normalization_std_for_selected_joint':std[JSEL].tolist(),'channel_names':channels,'functions':refs},'tpose':{'joint_names':names,'clean_joint_names':clean,'parents':parents.tolist(),'selected_joint_xyz':tpose[JSEL].tolist(),'selected_joint_local_quaternion_wxyz':np.asarray(c['tpos_local_rotations'])[JSEL].tolist()},'frames':[],'qc':qc,'pixel_annotations':pixels,'schematic_not_executed':{'label':'schematic_not_executed','model_input_relation':'Real standardized/padded motion array computed. Caption/joint-name embeddings, noisy x_tau, denoiser v_theta and ODE rollout were not computed.','x_tau':None,'velocity_v_theta':None,'generated_motion':None,'contact_inference':None,'surface_encoder':None,'robot_adaptation':'No native G1/Go2/Franka-to-H2 inference established. This is a UniMate native Mixamo text-to-skeleton preprocessing case.'},'caveats':['Official rotation_conversions.py prose says first two rows; executable Motion encoding and numpy decoding use first two columns. This trace follows code.','Local displacement is not physical m/s and is distinct from neural flow-matching velocity v_theta.','No textured mesh loaded. Only actual canonical skeletons shown; no fabricated surface encoder.','Caption is a dataset reference, not a generated neural response.']}
for i in F:
 root_removed=pos[i,JSEL].copy();root_removed[[0,2]]-=pos[i,0,[0,2]];pq=rebased.qs[i,parents[JSEL]];rm=Q(pq[None]).rotation_matrix(cont6d=False)[0]
 trace['frames'].append({'feature_row_index':i,'source_pose_frame':i,'source_delta_destination_frame':i+1,'source_time_seconds':i/fps,'world_position_before_loader_grounding':original[i,JSEL].tolist(),'world_position_after_loader_grounding':pos[i,JSEL].tolist(),'root_xz_removed_position':root_removed.tolist(),'root_facing_quaternion_wxyz':facing[i].tolist(),'destination_root_facing_quaternion_wxyz':facing[i+1].tolist(),'world_delta_to_next_frame':(pos[i+1,JSEL]-pos[i,JSEL]).tolist(),'source_joint_local_quaternion_wxyz':local[i,JSEL].tolist(),'tpose_relative_parent_quaternion_wxyz':pq.tolist(),'tpose_relative_parent_rotation_matrix':rm.tolist(),'raw_feature_12d':features[i,JSEL].tolist(),'normalized_feature_12d':normalized[i,JSEL].tolist()})
(HERE/'trace.json').write_text(json.dumps(trace,ensure_ascii=False,indent=2)+'\n')
readme=['# UniMate real sample trace','',f'Actual clip: Jumping_Up-000. 8 released poses,22 Mixamo joints,30fps. Caption: {caption}',f'Selected joint: {names[JSEL]}, zero-based index9; parent LeftLeg,index6.','', 'build_case.py executes official numpy preprocessing/recovery bodies and official Motion quaternion/FK reference code on CPU. No installation, neural forward pass, ODE sampling, robot retargeting, simulation or hardware execution.', '', 'Frames0/3/6 map to feature rows0/3/6 and use frames1/4/7 for the delta channels. Eight poses yield7 feature rows; frame7 has no feature row.', '12D = RIFKE position3 + T-pose-relative,HML-reordered rotation6 + local displacement3. LeftFoot rotation channels store rebased parent LeftLeg rotation. Executable encoding uses matrix columns0 and1.', 'Y-up,Z-forward, canonical body units with skeleton diameter2.0; not asserted meters. Delta is body units/frame, not m/s; it is not neural flow velocity.', 'Cached official statistics normalize the actual values. Shape(7,22,12) becomes(60,22,12) then(1,22,12,60), with only time0-6 valid. This is preprocessing output, not a neural forward-pass result.', '', 'QC: '+json.dumps(qc), '', 'Figures: skeleton_tpose_motion.png/.svg, left_foot_12d_values.png/.svg and named foot crops. joint_pixels.json gives top-left PNG pixel positions and projected coordinate axes. trace.json and derived_preprocessing.npz retain numeric stages; asset_manifest.json records SHA256/sourcepaths/labels.', '', 'model_inputs_derived.npz is the shareable computed-tensor export: features_12d/normalized_12d [7,22,12], model_motion_array_bjdt [1,22,12,60], time_mask [60], joint_mask [22], T-pose arrays [22,12]/[22,3], statistics and topology. Original source pose/quaternion arrays stay remote. build_case.py is reproduction code targeting the pinned ArenaLabs cache and external source checkout; it is not standalone without the source dependencies listed in the manifest.', '', 'No textured mesh is loaded. This does not establish G1/Go2/Franka-to-H2 mapping. Raw references/dependency sources remain remote; only derived figures, selected trace, computed tensor export and reproduction script may be copied to local presentation assets.']
(HERE/'README.md').write_text('\n'.join(readme)+'\n')
references=[]
for p,role in [(CLIP,'motion'),(COND,'topology/T-pose'),(CAP,'caption'),(STATS,'normalization statistics'),(CONFIG,'official resolved config'),(CODE/'unimate/utils/motion_utils.py','executed preprocessing'),(CODE/'unimate/utils/rotation_conversions.py','executed numpy decoder'),(CODE/'unimate/dataset/transforms.py','executed transforms'),(CODE/'unimate/dataset/mixture/dataset.py','loader semantics'),(CODE/'unimate/dataset/mixture/collate.py','tensor axis contract')]+[(p,'official source dependency, not installed') for p in (HERE/'reference_dependency').glob('*.py')]:references.append({'label':'reference','role':role,'source_path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'copied_to_local_assets':False})
assets=[]
for p in sorted(HERE.iterdir()):
 if p.is_file() and p.name!='asset_manifest.json':assets.append({'label':'derived' if p.suffix in ['.png','.svg','.npz','.json'] else 'reproduction_code_or_notes','file':p.name,'absolute_path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'sourcepaths':[str(CLIP),str(COND),str(STATS)],'frame_indexes':F,'joint_index':JSEL,'units':'canonical body units; standardized arrays dimensionless','contains_schematic_numeric_output':False})
manifest={'schema_version':1,'created_at_utc':trace['created_at_utc'],'label_definitions':{'reference':'upstream data/code/statistics','derived':'actually computed/rendered from fixture','schematic_not_executed':'unexecuted model/robot stages; no fabricated numeric values'},'references':references,'assets':assets,'schematic_status':trace['schematic_not_executed'],'source_revisions':{'code':commit,'dataset':DR,'model_statistics':MR},'license_sources':['https://huggingface.co/datasets/Linzhan/UniML3D/blob/'+DR+'/README.md#licensing','https://huggingface.co/Linzhan/UniMate/blob/'+MR+'/LICENSE'],'qc':qc}
(HERE/'asset_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'out_dir':str(HERE),'qc':qc,'joint':trace['joint'],'frames':trace['frames'],'pixels':[{'panel':p['label'],'xy':p['joint_pixel_xy'],'crop':p['crop_file']} for p in pixels['panels']]},indent=2))
