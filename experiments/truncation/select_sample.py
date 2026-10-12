"""Stratified 40-clip sample over the groups that exist in the converted set (deterministic, seed 0)."""
import json, os, sys, random, collections, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

QUOTA = {'carry': 5, 'push_pull': 5, 'lift_place': 4, 'whole_other': 3, 'sit': 5, 'ground': 3,
         'kick_hit': 4, 'upper_manip': 5, 'throw': 2, 'step': 2, 'locomotion': 2}


def main():
    rows = [json.loads(l) for l in open(OUT + '/cov_all.jsonl')]
    labs = load_labels()
    for r in rows:
        r['group'] = group_of(labs[r['motion_id']])
    rng = random.Random(0)
    by = collections.defaultdict(list)
    for r in rows:
        by[r['group']].append(r)
    pick = []
    for g, q in QUOTA.items():
        pool = sorted(by[g], key=lambda r: r['motion_id'])
        rng.shuffle(pool)
        seen_obj, seen_lu, chosen = set(), set(), []
        # greedy: prefer unseen object category, then unseen frame_lu
        for pass_ in range(3):
            for r in pool:
                if len(chosen) >= q:
                    break
                if r in chosen:
                    continue
                if pass_ == 0 and (r['obj'] in seen_obj or r['frame_lu'] in seen_lu):
                    continue
                if pass_ == 1 and r['frame_lu'] in seen_lu and r['obj'] in seen_obj:
                    continue
                chosen.append(r); seen_obj.add(r['obj']); seen_lu.add(r['frame_lu'])
        pick += chosen
    keys = ['motion_id', 'group', 'frame_lu', 'obj', 'actor', 'dur_s',
            'cov_f5_15', 'cov_f5_20', 'cov_f5_30', 'cov_f8_20', 'cov_w5_20', 'covhalf_f5_20',
            'nstar_f5_90', 'nstar_f5_95', 'nstar_w5_90', 'rescue_f5_20',
            'ctrl_self_last20', 'cov_f5_20_same_span', 'ctrl_samelu_otheractor', 'ctrl_sameactor_othergroup',
            'ctrl_othergroup_otheractor', 'first_cycle_end_s', 'n_contact_cycles', 'n_move_cycles',
            'cyc_in_15', 'cyc_in_20', 'cyc_in_30', 'text']
    with open(OUT + '/sample40.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(keys + ['nn_self_20_p50_p90_p99'])
        for r in pick:
            w.writerow([(round(r.get(k), 3) if isinstance(r.get(k), float) else r.get(k)) for k in keys] +
                       [r.get('nn_self_20', [None] * 3)[:3]])
    print(len(pick), collections.Counter(r['group'] for r in pick))
    print(collections.Counter(r['obj'] for r in pick))


if __name__ == '__main__':
    main()
