import sys, importlib.util, numpy as np
R='/home/arenalabs/Desktop/"be water, robot"/alpha_motion2/research/alphamotion_scaling_20261004'
spec=importlib.util.spec_from_file_location('tr',R+'/data/trajectory_reader.py'); tr=importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
d=tr.load_xwbc_trajectory(R+'/data/data_manifest_v2.json','lafan::fallAndGetUp1_subject4','human_smpl',with_contact=True)
for k,v in d.items(): print(k, getattr(v,'shape',v) if hasattr(v,'shape') else (v if not isinstance(v,(list,dict)) else type(v)))
print([str(x) for x in d['joint_names']])
