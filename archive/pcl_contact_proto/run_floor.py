import sys, json, time; sys.path.insert(0,'.')
from pcl_contact import floor as FL, sources as SR
m=json.load(open(SR.MANIFEST))['xwbc']['groups']
out={}
for sess in ['subject4','subject5']:
    gids=[g['group_id'] for g in m if g['group_id'].startswith('lafan::') and FL.session_of(g['group_id'])==sess]
    for body in ['unitree_g1_29dof','human_smpl']:
        t=time.time(); r=FL.session_floor(gids,body); r['seconds']=round(time.time()-t,1)
        out[f'{sess}/{body}']=r; print(sess,body,len(gids),'floor %.2f p10 %.2f p90 %.2f n %d'%(r['floor_cm'],r['p10'],r['p90'],r['n_static']),r['seconds'],'s',flush=True)
json.dump(out,open('out/session_floor.json','w'),indent=1)
