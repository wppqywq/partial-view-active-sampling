"""Render saved numerical results only; execute inside Slurm, no model inference."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Use Slurm')
import json
import hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

run = Path('/mnt/disk2/youyouyang/proposal2/foveated_v3/mean/runs/22742')
out = Path(os.environ['FV3_PLOT_OUT'])
out.mkdir(parents=True, exist_ok=True)
config = json.loads((run/'config.json').read_text())
epoch, completed = int(config['best_epoch']), int(config['completed_epochs'])
terminal = config['state'] in ('early_stopped','budget_exhausted')
source = run / f'development_epoch_{epoch:03d}.json'
data = json.loads(source.read_text())
records = data['states']
lookup = {(r['image'], r['kind'], r['budget']): r for r in records}
assert len(lookup) == len(records)
names = sorted({r['image'] for r in records if r['kind']=='human' and r['budget']==16})
assert names
for name in names:
    assert all((name, kind, budget) in lookup for kind in ('human','random') for budget in range(1,17))
curves = {}
for kind in ('human','random'):
    matrix = np.array([[lookup[name,kind,b]['direct_mse'] for b in range(1,17)] for name in names])
    curves[kind] = matrix.mean(0).tolist()
history = []
for line in (run/'curves.jsonl').read_text().splitlines():
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        continue  # writer may currently be appending a newer epoch
    if row['epoch'] <= completed:
        history.append(row)
assert [r['epoch'] for r in history] == list(range(1,completed+1))
assert min(history,key=lambda r:r['development_mse'])['epoch'] == epoch
plt.rcParams.update({'font.size':11, 'axes.spines.top':False,'axes.spines.right':False})
fig, ax = plt.subplots(figsize=(7.4,4.8))
for kind, label, color, style in [('human','Recorded human prefixes','#2166ac','-'),
                                ('random','Random prefixes','#d47b00','--')]:
    ax.plot(range(1,17),curves[kind],label=label,color=color,linestyle=style,marker='.',linewidth=2)
ax.set(xlabel='Number of observations (includes initial fixation)',ylabel='Feature MSE (lower is better)',
       title=f'Revised observer: development replay\nEpoch {epoch}; fixed cohort of {len(names)} images',
       xticks=[1,4,8,12,16],xlim=(1,16))
ax.grid(alpha=.18)
ax.legend(frameon=False)
fig.text(.5,.025,'One human trajectory per image; starts are not matched to random.\nDescriptive predictor check, not a U/G policy comparison.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.105,1,1))
fig.savefig(out/'observation_curve.png',dpi=180)
fig.savefig(out/'observation_curve.pdf')
plt.close(fig)
fig, ax = plt.subplots(figsize=(7.4,4.6))
ax.plot([r['epoch'] for r in history],[r['train_mse'] for r in history],label='Training (resampled views)',color='#777777',linewidth=1.5)
ax.plot([r['epoch'] for r in history],[r['development_mse'] for r in history],label='Development (fixed views)',color='#2166ac',linewidth=2)
ax.scatter([epoch],[history[epoch-1]['development_mse']],color='#b2182b',s=40,zorder=3,label=f'Best so far: epoch {epoch}')
ax.set(xlabel='Training epoch',ylabel='Mean feature MSE',title=f'Mean predictor: {completed} / 80 epochs; '+('frozen' if terminal else 'training'))
ax.grid(alpha=.18);ax.legend(frameon=False,fontsize=9)
fig.text(.5,.02,'Training and development use different image/view distributions. '+('Mean is now frozen.' if terminal else 'Training remains in progress.'),ha='center',fontsize=8)
fig.tight_layout(rect=(0,.035,1,1))
fig.savefig(out/'training_curve.png',dpi=180)
fig.savefig(out/'training_curve.pdf')
plt.close(fig)
summary={'mean_job':22742,'plot_job':os.environ['SLURM_JOB_ID'],'snapshot_epochs_completed':completed,
         'checkpoint_epoch':epoch,'mean_frozen':terminal,'all_development_images':371,'fixed_cohort_images':len(names),
         'cohort':'same images at every budget, chosen human trajectory truly reaches16; no padding',
         'aggregation':'one state per image/kind/budget, images equally weighted',
         'mse_curves':curves,'best_overall_dev_mse':history[epoch-1]['development_mse'],
         'source_file':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
         'source_kind':'saved epoch evaluation; no model inference, no U or G labels',
         'limitations':['Only one preselected human trajectory per image, not all participants.',
                        'Random histories may start from a different participant initial point.',
                        'Checkpoint selected on development; mean training '+('is finished.' if terminal else 'remains unfinished.'),
                        'No cross-version controlled MSE comparison: observation/budget changed.']}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'training_snapshot.json').write_text(json.dumps(history,indent=2)+'\n')
print(json.dumps(summary),flush=True)
