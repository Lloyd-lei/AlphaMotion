#!/usr/bin/env python3
"""Freeze six-run source identities and group/whole-body partition rules from audit."""
from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(audit,external):
 x= audit['xwbc'];held=set(x['heldout_bodies']);assert len(held)==4
 x['historical_pretrain_seen_flags']='Historical old checkpoint eligibility only; main scaling initializes from scratch.'
 x['train_eligible_sparse_rows_by_body']={b['body']:(0 if b['body'] in held else x['group_counts']['train_rows']) for b in x['bodies']}
 assert all(x['train_eligible_sparse_rows_by_body'][b]==0 for b in held)
 gids=[g['group_id'] for g in x['groups']];assert len(gids)==len(set(gids))
 ranges={}
 for g in x['groups']:
  assert g['length']>0 and g['split'] in [0,1];ranges.setdefault(g['cache_index'],[]).append((g['start'],g['start']+g['length'],g['group_id']))
 for rs in ranges.values():
  rs.sort();assert all(a[1]<=b[0] for a,b in zip(rs,rs[1:]))
 uni=audit['unimate'];seen_hash={};group_splits={}
 for b in uni['bodies']:
  for q in b['source_sequences']:
   if q['status']!='passed':continue
   sha=q['source_sha256'];group=q['source_group_id'];sp=q['split']
   if sha in seen_hash:assert seen_hash[sha]==group
   if group in group_splits:assert group_splits[group]==sp
   seen_hash[sha]=group;group_splits[group]=sp
 cases=[]
 # Whole heldout clips, sampled deterministically for an executable eval manifest.
 for ci in [0,1]:
  gs=sorted([g for g in x['groups'] if g['cache_index']==ci and g['split']==1],key=lambda g:hashlib.sha256(g['group_id'].encode()).hexdigest())[:2]
  for g in gs:
   for body in x['bodies']:
    cases.append({'case_id':g['group_id']+'->'+body['body'],'source_body':'human_smpl','target_body':body['body'],'group_id':g['group_id'],'clip_id':g['clip'],'body_exposure':'heldout_body' if body['body'] in held else 'seen_body','clip_exposure':'heldout_clip','source_frame_count':g['length'],'fps':x['fps'],'task_kind':'reconstruction' if body['body']=='human_smpl' else 'paired_mapping','reference_kind':'same_body' if body['body']=='human_smpl' else 'paired_target','target_morphology':'humanoid','reader':'trajectory_reader.load_xwbc_trajectory','reference_units':'cm','training_access':False})
 return {'schema':'six_run_data_manifest_v1','same_data_for':['A_S','A_M','A_L','B_S','B_M','B_L'],'initialization':'from_scratch_no_old_checkpoint','normalization':'geometry-only maximum root-to-joint path length; exclude root offset, no motion/eval GT fitting','xwbc':x,'unimate':uni,'external':external,'evaluation_examples':cases,'validation':{'xwbc_group_count':len(gids),'xwbc_train_eval_group_overlap':0,'xwbc_clip_row_overlap':0,'unimate_content_hash_split_conflicts':0,'whole_body_holdout_training_rows_under_manifest_policy':0,'validation_scope':'manifest eligibility policy; actual runtime sampler checked by training owner','partition_scope':'Unseen XWBC targets are heldout humanoids only. UniMate cross morphology supports own-body reconstruction/completion, no fake cross-body pair truth.'},'surface_contract':'stable_surface_v1, actual mesh+binding only; missing surface -> valid=false, never synthesize from skeleton points','exposure_semantics':{'seen_body_train_clip':'train pool (A/B,S/M/L same groups)','heldout_body_train_clip':'unseen target, seen motion; NEVER motion-unseen','seen_body_heldout_clip':'known target, unseen motion','heldout_body_heldout_clip':'unseen target and unseen motion'},'source_checksum_level':audit['checksum_level']}

if __name__=='__main__':
 audit=json.loads((HERE/'coverage_audit.json').read_text());external=json.loads((HERE/'external_manifest.json').read_text());m=build(audit,external)
 m['input_hashes']={n:digest(HERE/n) for n in ['coverage_audit.json','external_manifest.json']}
 (HERE/'data_manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'manifest':str(HERE/'data_manifest.json'),'sha256':digest(HERE/'data_manifest.json'),'eval_cases':len(m['evaluation_examples']),'validation':m['validation']}))
