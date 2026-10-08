"""One bounded read-only audit of saved r2 runs; no model inference or training."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Run this audit through Slurm')
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean
import numpy as np

root = Path('/mnt/disk2/youyouyang/proposal2/foveated_v3')
project = Path('/home/youyouyang/proposal2')
def read(path):
    return json.loads(path.read_text())
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def same(a, b):
    assert np.allclose(a, b, atol=1e-10, rtol=0), (a, b)

runs = {'prepare': root/'prepare/22741', 'mean': root/'mean/runs/22742',
        'variance': root/'variance/runs/22754', 'gain': root/'gain/runs/22779',
        'paths': root/'evaluation/runs/22831', 'choice': root/'human_choice/runs/22832'}
for run in runs.values():
    for line in (run/'source.sha256').read_text().splitlines():
        expected, filename = line.split(maxsplit=1)
        assert digest(run/filename.lstrip('*')) == expected
protocol = read(runs['prepare']/'protocol.json')
assert protocol['maximum_budget'] == 16 and protocol['includes_initial_fixation']
assert np.ceil(protocol['training_mean_fixations']) == 16
assert protocol['geometry']['physical_calibration'] is False
visual = read(project/'foveated_v3/visual_review_22741.json')
assert visual['state'] == 'accepted'
assert visual['protocol_sha256'] == digest(runs['prepare']/'protocol.json')
mean = read(runs['mean']/'frozen_mean.json')
var_review = read(runs['variance']/'calibration_review.json')
assert var_review['state'] == 'accepted'
assert var_review['variance_checkpoint_sha256'] == digest(runs['variance']/'best.pt')
assert var_review['calibration_summary_sha256'] == digest(runs['variance']/'calibration_summary.json')
gain = read(runs['gain']/'completion.json')
assert gain['state'] == 'completed' and gain['best_checkpoint_sha256'] == digest(runs['gain']/'best.pt')
counts = read(runs['gain']/'cache_summary.json')['counts']
assert (counts['train_images'], counts['development_images']) == (3343, 371)
paths_config = read(runs['paths']/'config.json')
assert paths_config['state'] == 'completed'
summary = read(runs['paths']/'summary.json')
manifest_path = Path(protocol['manifest'])
assert digest(manifest_path) == protocol['manifest_sha256']
manifest = read(manifest_path)
raw_path = Path(manifest['raw_fixations'])
assert digest(raw_path) == protocol['raw_fixations_sha256']
raw = {(r['name'], int(r['subject'])): r for r in read(raw_path)
       if manifest['images'][r['name']]['split'] == 'development'}
policies = tuple(summary['matched_endpoints']['endpoint'])
collected = {s: {'end': {p: [] for p in policies}, 'curve': {p: [[] for _ in range(16)] for p in policies},
                 'trials': 0, 'denominators': [0]*16}
             for s in ('matched_endpoints', 'complete_budget_cohort')}
seen_trials = set()
record_paths = sorted(Path(paths_config['records']).glob('*.json'))
assert len(record_paths) == 371
for path in record_paths:
    record = read(path)
    assert record['identity_hash'] == summary['identity_hash']
    for trial in record['trials']:
        key = (record['image'], int(trial['subject']))
        assert key not in seen_trials
        seen_trials.add(key)
        original = raw[key]
        xy = [list(p) for p in zip(original['X'], original['Y'])]
        n = min(len(xy), 16)
        assert trial['original_length'] == len(xy) and trial['matched_budget'] == n
        assert trial['paths']['human'][0]['history_xy'] == xy[:n]
        first = trial['paths']['human'][0]['states'][0]['mse']
        for policy, paths in trial['paths'].items():
            assert len(paths) == (5 if policy in ('uniform', 'traincenter') else 1)
            for trajectory in paths:
                assert trajectory['history_xy'][0] == xy[0]
                assert len(trajectory['states']) == n
                assert trajectory['states'][0]['mse'] == first
                for step, state in enumerate(trajectory['states'], 1):
                    assert state['budget'] == step and np.isfinite(state['mse'])
                    assert not state['stopped']
    for scope, store in collected.items():
        trials = [t for t in record['trials'] if scope == 'matched_endpoints' or t['original_length'] >= 16]
        if not trials:
            continue
        store['trials'] += len(trials)
        for policy in policies:
            store['end'][policy].append(fmean(fmean(p['states'][-1]['mse'] for p in t['paths'][policy]) for t in trials))
        for step in range(16):
            available = [t for t in trials if t['matched_budget'] > step]
            store['denominators'][step] += len(available)
            if available:
                for policy in policies:
                    store['curve'][policy][step].append(fmean(fmean(p['states'][step]['mse'] for p in t['paths'][policy]) for t in available))
assert seen_trials == set(raw) and len(seen_trials) == 3698
for scope, store in collected.items():
    expected = summary[scope]
    assert store['trials'] == expected['trials']
    assert store['denominators'] == [d['trials'] for d in expected['denominators']]
    for policy in policies:
        same(fmean(store['end'][policy]), expected['endpoint'][policy]['mse'])
        same([fmean(v) for v in store['curve'][policy]], expected['curve_means'][policy]['mse'])
choice = read(runs['choice']/'config.json')
assert choice['state'] == 'completed' and choice['optimization_complete']
fits = read(runs['choice']/'fits.json')
assert all(f['status'] == 'optimized' and f['gradient_infinity_norm'] <= 1e-6 for f in fits.values())
alignment = read(runs['choice']/'reference_alignment.json')
assert alignment['requested_development_targets'] == alignment['matched_targets'] == 49690
assert not alignment['missing'] and not alignment['problems']
choice_summary = read(runs['choice']/'development_summary.json')
models = tuple(choice_summary['overall']['models'])
expected_keys = set()
for (name, subject), row in raw.items():
    left, top, right, bottom = manifest['images'][name]['display_content_box']
    xy = np.column_stack((row['X'], row['Y'])).astype(np.float32)
    for t in range(1, min(len(xy), 16)):
        if left <= xy[t, 0] < right and top <= xy[t, 1] < bottom:
            expected_keys.add((name, subject, t))
seen = set()
values = defaultdict(lambda: defaultdict(list))
with (runs['choice']/'development_predictions.jsonl').open() as stream:
    for line in stream:
        r = json.loads(line)
        key = (r['image'], int(r['subject']), r['target_index'])
        assert key not in seen and key in expected_keys and r['history_budget'] == r['target_index']
        seen.add(key)
        vector = [r[m] for m in models]
        assert np.isfinite(vector).all()
        values[r['image']][r['subject']].append(vector)
assert seen == expected_keys and len(seen) == 49690
means = np.mean([np.mean([np.mean(rows, axis=0) for rows in trials.values()], axis=0) for trials in values.values()], axis=0)
same(means, [choice_summary['overall']['models'][m]['macro_logp'] for m in models])
audit = {'state': 'passed', 'audit_job': os.environ['SLURM_JOB_ID'], 'date': '2026-10-07',
         'scope': 'Completed revised-observer development experiment; no inference, retraining, or new holdout claims',
         'verified': ['all six immutable run source manifests', 'training-only ceil budget16 and engineering geometry',
                      'accepted r2 visual review and variance review identities', 'complete new gain data and checkpoint',
                      'all3698 human histories match raw prefixes including revisits; same policy starts and budgets',
                      'all six policy endpoint means and1..16curves independently recomputed in both cohorts',
                      'fixed16cohort1915trials and denominator accounting', 'four human fits meet gradient tolerance',
                      'exact49690reference target alignment', 'all supported raw human targets present once',
                      'human-choice image/trial/state means independently recomputed'],
         'source_runs': {k: int(v.name) for k, v in runs.items()},
         'limitations': choice['limitations']}
(project/'results/foveated_completion_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
(project/'results/SHA256SUMS').write_text(''.join(
    f'{digest(path)}  {path.name}\n' for path in sorted((project/'results').glob('*.json'))))
print(json.dumps(audit))
