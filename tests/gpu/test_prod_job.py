from surface_retargeting.config import CFG as SRCFG  # machine paths: configs/sr.toml
import sys, json, time, torch
from surface_retargeting.gpu import production as PR
jobs = [json.loads(l) for l in open(f'{SRCFG.data_root}/production/v1/_jobs/jobs.jsonl')]
name = sys.argv[1]; robot = sys.argv[2]; n = int(sys.argv[3])
job = [j for j in jobs if j['object']['name'] == name and j['robot'] == robot][0]
job = dict(job, clips=job['clips'][:n], job_id='TEST_' + job['job_id'])
summ, rows, qa = PR.run_job(job, f'{SRCFG.work}/results/prod_test', B_max=256)
print(json.dumps(summ, indent=1))
for q in qa: print(json.dumps({k: q[k] for k in ('clip_id', 'frames', 'flagged', 'flags_hist', 'human_contact_robot_solid_frac', 'ground_pen_max_mm', 'object_pen_max_mm', 'ground_slip_frac_of_contact') if k in q}))
