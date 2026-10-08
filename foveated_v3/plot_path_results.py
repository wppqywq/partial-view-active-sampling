"""Render completed path results and descriptive paired image intervals; no inference."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Run through Slurm')
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

source = Path(os.environ['FV3_EVALUATION_RUN'])
out = Path(os.environ['FV3_PLOT_OUT'])
out.mkdir(parents=True, exist_ok=True)
config = json.loads((source / 'config.json').read_text())
assert config['state'] == 'completed'
result = json.loads((source / 'summary.json').read_text())
fixed = result['complete_budget_cohort']
B = result['maximum_budget']
assert all(d['images'] == fixed['images'] and d['trials'] == fixed['trials']
           for d in fixed['denominators'])
styles = {
    'human': ('Human replay', '#31688e', '-'),
    'U': ('Uncertainty U', '#e69f00', '-'),
    'localG': ('Local G', '#7b4ba5', '-'),
    'globalG': ('Global G', '#d84a38', '-'),
    'traincenter': ('Center prior', '#777777', '--'),
    'uniform': ('Uniform', '#008d91', '--'),
}
fig, ax = plt.subplots(figsize=(8.2, 5.7))
for policy, (label, color, line) in styles.items():
    values = fixed['curve_means'][policy]['mse']
    assert len(values) == B and np.isfinite(values).all()
    assert np.isclose(values[-1], fixed['endpoint'][policy]['mse'])
    ax.plot(range(1, B+1), values, label=label, color=color, linestyle=line, lw=2.2)
initial = [fixed['curve_means'][p]['mse'][0] for p in styles]
assert np.ptp(initial) < 1e-7
ax.set(xlabel='Number of observations (includes initial fixation)',
       ylabel='Feature MSE (lower is better)', xticks=[1, 4, 8, 12, 16],
       title='Revised observer: unrestricted movement\n'
             f'Fixed cohort: {fixed["images"]} development images, {fixed["trials"]:,} human trials')
ax.grid(alpha=.18)
ax.spines[['top', 'right']].set_visible(False)
ax.legend(frameon=False)
fig.text(.5, .025, 'Same per-trial start and budget; average seeds, then trials, then images.\n'
         'Real human revisits retained. Movement and information exposure are not matched.',
         ha='center', fontsize=9)
fig.tight_layout(rect=(0, .09, 1, 1))
fig.savefig(out / 'fixed_cohort_mse.png', dpi=180)
fig.savefig(out / 'fixed_cohort_mse.pdf')
plt.close(fig)

# Pair policies within each human trial, then average trials within image.
# This preserves the evaluation's equal-image estimand and avoids treating
# participants or random seeds as independent image replicates.
pairs = [('globalG', 'U'), ('localG', 'U'), ('globalG', 'localG'),
         ('globalG', 'uniform'), ('localG', 'uniform'), ('globalG', 'traincenter')]
differences = {scope: {f'{a}_minus_{b}': [] for a, b in pairs}
               for scope in ('matched_endpoints', 'complete_budget_cohort')}
paths = sorted(Path(config['records']).glob('*.json'))
assert len(paths) == result['matched_endpoints']['images']
for path in paths:
    record = json.loads(path.read_text())
    assert record['identity_hash'] == result['identity_hash']
    for scope in differences:
        trials = [t for t in record['trials']
                  if scope == 'matched_endpoints' or t['original_length'] >= B]
        if not trials:
            continue
        endpoints = {p: np.mean([np.mean([v['states'][-1]['mse'] for v in t['paths'][p]])
                                 for t in trials]) for p in styles}
        for a, b in pairs:
            differences[scope][f'{a}_minus_{b}'].append(float(endpoints[a]-endpoints[b]))
rng = np.random.default_rng(20261007)
paired = {}
for scope, pair_values in differences.items():
    paired[scope] = {}
    for name, values in pair_values.items():
        a = np.asarray(values)
        assert len(a) == result[scope]['images']
        resamples = a[rng.integers(0, len(a), size=(2000, len(a)))].mean(1)
        paired[scope][name] = {'images': len(a), 'mean_mse_difference': float(a.mean()),
                             'image_bootstrap95': np.quantile(resamples, [.025, .975]).tolist()}
summary = {
    'source_run': str(source),
    'source_summary_sha256': hashlib.sha256((source / 'summary.json').read_bytes()).hexdigest(),
    'interpretation': 'Descriptive development comparisons; intervals not multiplicity corrected. '
                      'No movement or information-exposure matching; no human mechanism claim.',
    'paired_differences': paired,
    'fixed_cohort_selected_budgets': {
        p: {str(t): {m: fixed['curve_means'][p][m][t-1] for m in fixed['curve_means'][p]}
            for t in (4, 8, 16)} for p in styles}}
(out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
print(json.dumps({'state': 'completed', 'images': fixed['images'], 'trials': fixed['trials']}))
