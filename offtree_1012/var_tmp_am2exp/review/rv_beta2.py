import sys, importlib.util, numpy as np
R='/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004'
spec=importlib.util.spec_from_file_location('tr',R+'/data/trajectory_reader.py'); tr=importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
for b in sys.argv[1:]:
    d=tr.load_xwbc_trajectory(R+'/data/data_manifest_v2.json','lafan::fallAndGetUp1_subject4',b,with_contact=True)
    P=d['position_native']; rh=d['root_height_m']; hj=P[...,1]+100*rh[:,None]
    names=[str(x) for x in d['joint_names']]
    lo=hj.argmin(1); m=hj.min(1)
    foot=np.array(['ankle' in names[i].lower() or 'foot' in names[i].lower() or 'toe' in names[i].lower() or 'sole' in names[i].lower() for i in lo])
    print(b,'off-stance frames',(~foot).sum(),'lowest JOINT-centre h p10/50/90',np.round(np.percentile(m[~foot],[10,50,90]),1),'top lowest joints',np.unique(np.array(names)[lo[~foot]],return_counts=True))
