"""Same-start, same-budget human and new-policy paths under frozen v3 models."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Run through Slurm')
import argparse
from collections import defaultdict
import json
from pathlib import Path
import signal
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import common as C
import signals as S
import train_gain as G

P, T = C.P, C.T
POLICIES = ('human', 'U', 'localG', 'globalG', 'uniform', 'traincenter')
METRICS = ('mse', 'core_coverage', 'mean_resolution', 'movement')
STOP = False


def request_stop(*_):
    global STOP
    STOP = True


signal.signal(signal.SIGUSR1, request_stop)
signal.signal(signal.SIGTERM, request_stop)


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def train_center(manifest, rows):
    counts = np.ones((16, 16), dtype=np.float64)
    included = excluded = trials = 0
    for row in rows:
        entry = manifest['images'][row['name']]
        if entry['split'] != 'train':
            continue
        trials += 1
        left, top, right, bottom = entry['display_content_box']
        xy = np.column_stack((row['X'][1:], row['Y'][1:]))
        valid = ((xy[:, 0] >= left) & (xy[:, 0] < right)
                 & (xy[:, 1] >= top) & (xy[:, 1] < bottom))
        normalized = (xy[valid] - [left, top]) / [right-left, bottom-top]
        counts += np.histogram2d(normalized[:, 1], normalized[:, 0], bins=16,
                                 range=((0, 1), (0, 1)))[0]
        included += int(valid.sum())
        excluded += int((~valid).sum())
    return counts/counts.sum(), {'training_trials': trials, 'included_free_fixations': included,
                                'excluded_border_fixations': excluded, 'bins': [16, 16],
                                'pseudocount_per_bin': 1, 'initial_fixations_excluded': True}


def cell_mass(grid, box, mass):
    left, top, right, bottom = box
    x = (grid['x_edges']-left)/(right-left)
    y = (grid['y_edges']-top)/(bottom-top)
    edges = np.linspace(0, 1, 17)
    dx = np.maximum(0, np.minimum(x[1:, None], edges[None, 1:])-np.maximum(x[:-1, None], edges[None, :-1]))
    dy = np.maximum(0, np.minimum(y[1:, None], edges[None, 1:])-np.maximum(y[:-1, None], edges[None, :-1]))
    result = (dy @ (mass*256) @ dx.T).ravel()
    assert (result > 0).all() and np.isclose(result.sum(), 1)
    return result/result.sum()


@torch.no_grad()
def encode(models, view, content):
    rgb = P.rgb_tensor(view['rgb'])[None].cuda()
    resolution = P.pool_mask(view['resolution_map'])[None].cuda()
    return models.current(rgb, resolution, content), resolution


def state_record(view, current, target, content, budget, history, movement, short_side, stopped=False):
    valid = view['valid_mask']
    mse = float(S.spatial_average((current[1]-target).square(), content).mean().item())
    assert np.isfinite(mse)
    return {'budget': budget, 'actual_budget': len(history), 'mse': mse,
            'core_coverage': float(view['core_mask'][valid].mean()),
            'mean_resolution': float(view['resolution_map'][valid].mean()),
            'movement': movement/short_side, 'movement_pixels': movement,
            'last_xy': list(history[-1]), 'stopped': stopped}


class CandidateGeometry:
    """Cache only candidate resolution maps, without candidate RGB or features."""
    def __init__(self, obs, xy):
        empty = obs.observe([])
        # Use exact observer geometry; reconstructing q by subtracting and
        # re-adding the floor can create tiny spurious changes at revisits.
        self.maps = torch.from_numpy(np.stack([
            obs._pad(obs._resolution_at(point * obs.scale_xy) * obs._content_mask)
            for point in xy])).cuda()[:, None]
        self.content_pixels = int(empty['valid_mask'].sum())

    @torch.no_grad()
    def current(self, resolution_map, pooled=True):
        current = torch.from_numpy(resolution_map).cuda()[None, None]
        increase = (self.maps-current).clamp_min_(0)
        # Maps are already zero outside content: do NOT multiply a pooled
        # fractional content weight a second time at image-content boundaries.
        changes = (increase.sum((1, 2, 3))/self.content_pixels).cpu().numpy()
        return changes, F.avg_pool2d(increase, 14, 14) if pooled else None


@torch.no_grad()
def scores_before(models, gain, obs, current, resolution, content, weights, xy, increases, budget):
    # Geometry-only increases never read an after-view RGB or target feature.
    vectors, uncertainty = S.online_vectors(*current, content, weights, xy, obs.content_box,
                                            resolution, increases, budget, models.maximum_budget)
    predicted = gain(vectors)
    assert predicted.shape == (len(xy), 2) and torch.isfinite(predicted).all()
    return {'U': uncertainty.cpu().numpy(), 'globalG': predicted[:, 0].cpu().numpy(),
            'localG': predicted[:, 1].cpu().numpy()}


@torch.no_grad()
def image_paths(name, entry, rows, models, gain, center, identity):
    assert entry['split'] == 'development' and P.digest(entry['path']) == entry['sha256']
    obs = T.observer(entry, models.options)
    grid = obs.choice_grid()
    xy = grid['xy']
    geometry = CandidateGeometry(obs, xy)
    weights = torch.stack([P.pool_mask(obs.local_weights(point)) for point in xy]).cuda()
    center_probability = cell_mass(grid, obs.content_box, center)
    target = models.target_for_labels(name)
    short_side = min(obs.content_box[2]-obs.content_box[0], obs.content_box[3]-obs.content_box[1])
    trials = []
    for row in sorted(rows, key=lambda r: int(r['subject'])):
        human_xy = np.column_stack((row['X'], row['Y'])).tolist()
        limit = min(len(human_xy), models.maximum_budget)
        assert limit >= 1
        start = [human_xy[0]]
        first_view = obs.observe(start)
        content = P.pool_mask(first_view['valid_mask'])[None].cuda()
        first_current, first_resolution = encode(models, first_view, content)
        first_state = state_record(first_view, first_current, target, content, 1, start, 0, short_side)
        first_changes, first_increases = geometry.current(first_view['resolution_map'])
        first_scores = (scores_before(models, gain, obs, first_current, first_resolution,
                                     content, weights, xy, first_increases, 1) if limit > 1 else None)
        human_states = [dict(first_state)]
        movement = 0.
        for budget in range(2, limit+1):
            history = human_xy[:budget]
            movement += float(np.linalg.norm(np.asarray(history[-1])-history[-2]))
            view = obs.observe(history)
            current, _ = encode(models, view, content)
            human_states.append(state_record(view, current, target, content, budget, history, movement, short_side))
        paths = {'human': [{'seed': None, 'history_xy': human_xy[:limit], 'states': human_states}]}
        for policy in POLICIES[1:]:
            paths[policy] = []
            for seed in range(5 if policy in ('uniform', 'traincenter') else 1):
                rng = T.rng_for('v3_closed_loop', name, row['subject'], policy, seed)
                history = [list(start[0])]
                view, current, resolution = first_view, first_current, first_resolution
                states, movement, stopped = [], 0., False
                for budget in range(1, limit+1):
                    state = state_record(view, current, target, content, budget, history, movement, short_side, stopped)
                    states.append(state)
                    if budget == limit:
                        continue
                    changes, increases = ((first_changes, first_increases) if budget == 1 else
                                          geometry.current(view['resolution_map'], policy in ('U', 'localG', 'globalG')))
                    eligible = np.flatnonzero(changes > 1e-8)
                    state['eligible_candidates'] = len(eligible)
                    if not len(eligible):
                        stopped = True
                        state['stopped'] = True
                        state['stop_reason'] = 'no_candidate_increases_resolution'
                        continue
                    if policy in ('U', 'localG', 'globalG'):
                        scores = (first_scores if budget == 1 else scores_before(
                            models, gain, obs, current, resolution, content, weights, xy, increases, len(history)))
                        values = scores[policy][eligible]
                        # Seeded exact-tie breaking avoids fixed row-major bias.
                        tied = eligible[values == values.max()]
                        chosen = int(rng.choice(tied))
                        state['chosen_score'] = float(scores[policy][chosen])
                        state['exact_ties'] = len(tied)
                    elif policy == 'uniform':
                        chosen = int(rng.choice(eligible))
                    else:
                        prob = center_probability[eligible]
                        chosen = int(rng.choice(eligible, p=prob/prob.sum()))
                    point = xy[chosen].tolist()
                    state['action'] = {'candidate': chosen, 'xy': point,
                                       'mean_resolution_increase': float(changes[chosen])}
                    movement += float(np.linalg.norm(np.asarray(point)-history[-1]))
                    history.append(point)
                    next_view = obs.observe(history)
                    assert np.all(next_view['resolution_map']+1e-7 >= view['resolution_map'])
                    view = next_view
                    current, resolution = encode(models, view, content)
                assert states[0]['mse'] == first_state['mse']
                paths[policy].append({'seed': seed, 'history_xy': history, 'states': states})
        trials.append({'subject': row['subject'], 'original_length': len(human_xy),
                       'matched_budget': limit, 'initial_xy': start[0], 'paths': paths})
    return {'image': name, 'identity_hash': identity, 'content_box': obs.content_box,
            'candidate_grid_shape': list(grid['shape']), 'trials': trials}


def boot(values, rng):
    array = np.asarray(values, dtype=float)
    samples = rng.integers(0, len(array), (2000, len(array)))
    return np.quantile(array[samples].mean(1), [.025, .975]).tolist()


def trial_metric(trial, policy, index, metric):
    return float(np.mean([path['states'][index][metric] for path in trial['paths'][policy]]))


def aggregate(records, budget, complete_only=False):
    """Average policy seeds, then subjects within image, then images equally."""
    endpoint = {p: [] for p in POLICIES}
    reductions = {p: [] for p in POLICIES}
    curves = {p: {m: [] for m in METRICS} for p in POLICIES}
    denominators = []
    selected = []
    for record in records:
        trials = [t for t in record['trials'] if not complete_only or t['original_length'] >= budget]
        if trials:
            selected.append((record['image'], trials))
            for policy in POLICIES:
                endpoint[policy].append(np.mean([trial_metric(t, policy, -1, 'mse') for t in trials]))
                reductions[policy].append(np.mean([trial_metric(t, policy, 0, 'mse')-
                                                  trial_metric(t, policy, -1, 'mse') for t in trials]))
    for step in range(1, budget+1):
        available = [(name, [t for t in trials if t['matched_budget'] >= step]) for name, trials in selected]
        available = [(name, trials) for name, trials in available if trials]
        denominators.append({'budget': step, 'images': len(available), 'trials': sum(len(t) for _, t in available)})
        for policy in POLICIES:
            for metric in METRICS:
                values = [np.mean([trial_metric(t, policy, step-1, metric) for t in trials]) for _, trials in available]
                curves[policy][metric].append(float(np.mean(values)) if values else None)
    rng = T.rng_for('v3_path_bootstrap', complete_only)
    summary = {'images': len(selected), 'trials': sum(len(t) for _, t in selected),
               'image_names': [n for n, _ in selected], 'denominators': denominators,
               'curve_means': curves, 'endpoint': {}, 'human_minus_policy': {}}
    for policy in POLICIES:
        summary['endpoint'][policy] = {'mse': float(np.mean(endpoint[policy])),
                                       'initial_to_endpoint_reduction': float(np.mean(reductions[policy])),
                                       'mse_image_bootstrap95': boot(endpoint[policy], rng),
                                       'reduction_image_bootstrap95': boot(reductions[policy], rng)}
        if policy != 'human':
            diff = np.asarray(endpoint['human'])-endpoint[policy]
            summary['human_minus_policy'][policy] = {'mean': float(diff.mean()),
                                                       'image_bootstrap95': boot(diff, rng)}
    return summary


def report(run, result):
    fig, axs = plt.subplots(1, 4, figsize=(18, 4))
    for ax, metric, label in zip(axs, METRICS, ('Feature MSE', 'Clear core coverage', 'Mean nominal resolution', 'Movement / content short side')):
        for policy in POLICIES:
            values = result['matched_endpoints']['curve_means'][policy][metric]
            ax.plot(range(1, len(values)+1), values, marker='.', label=policy)
        ax.set(xlabel='Observation budget (includes initial fixation)', ylabel=label)
        ax.grid(alpha=.2)
    axs[0].legend(fontsize=8)
    fig.suptitle('Human and new policy paths: same images, per-trial starts and available budgets')
    fig.tight_layout()
    fig.savefig(run/'comparison.png', dpi=160)
    fig.savefig(run/'comparison.pdf')
    plt.close(fig)
    primary, complete = result['matched_endpoints'], result['complete_budget_cohort']
    lines = ['# Continuous foveation: human and policy scene prediction', '',
             f"Development: {primary['images']} images, {primary['trials']} true human trials.",
             'Each policy starts at that participant\'s actual first fixation and receives T=min(actual length,B) budget.',
             'Average five baseline seeds within trial, trials within image, then images equally. No human padding or removed revisits.', '',
             '|Path|Matched endpoint MSE|Initial-to-endpoint reduction|', '|---|---:|---:|']
    for policy in POLICIES:
        value = primary['endpoint'][policy]
        lines.append(f"|{policy}|{value['mse']:.6f}|{value['initial_to_endpoint_reduction']:.6f}|")
    lines += ['', f"Fixed B={result['maximum_budget']} cohort: {complete['images']} images and {complete['trials']} trials; detailed results in summary.json.",
              'Curves use all participants reaching each displayed budget; all policies share the same denominator at that budget. The cohort changes across budgets; the separate complete-B curves use a fixed cohort.',
              'Endpoint differences and descriptive 95% image-bootstrap intervals are in summary.json; they are not multiplicity corrected.',
              'Candidate exhaustion holds the current input without inventing a fixation; actual budgets and stopped flags remain in every record.',
              'Human participants viewed normal images. This is offline replay through an engineering observation model, not a measurement of human internal prediction error.',
              'Human continuous coordinates and real revisits differ from policy grid/no-resolution-change exclusion. Movement and information exposure are reported, not perfectly matched.',
              'The prior holdout was not evaluated here. This is a development-version comparison, not a fresh confirmatory test.', '']
    (run/'REPORT.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mean-run', required=True, type=Path)
    parser.add_argument('--variance-run', required=True, type=Path)
    parser.add_argument('--gain-run', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--records-root', required=True, type=Path)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(T.SEED)
    models = C.FrozenObserver(args.mean_run, args.variance_run, require_calibration=True)
    gain = G.load_gain(args.gain_run, models)
    rows = json.loads(Path(models.manifest['raw_fixations']).read_text())
    center, center_info = train_center(models.manifest, rows)
    source_hashes = {p.name: P.digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    identity_data = {'observer_identity': models.identity, 'gain_checkpoint_sha256': gain.best_hash,
                     'gain_normalization_sha256': gain.norm_hash,
                     'sources': source_hashes, 'center_mass': center.tolist(), 'maximum_budget': models.maximum_budget,
                     'baseline_seeds': 5, 'seed': T.SEED, 'gpu': torch.cuda.get_device_name(),
                     'torch': torch.__version__, 'cuda': torch.version.cuda, 'aggregation': 'seeds_then_trials_then_images'}
    identity = C.identity_hash(identity_data)
    record_dir = args.records_root/identity
    record_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.run_dir/'config.json', {'state': 'running', 'identity_hash': identity, **identity_data,
                                          'records': str(record_dir), 'mean_run': str(args.mean_run),
                                          'variance_run': str(args.variance_run), 'gain_run': str(args.gain_run)})
    write_json(args.run_dir/'train_center.json', {'bin_mass_yx': center.tolist(), **center_info})
    by_image = defaultdict(list)
    for row in rows:
        if models.manifest['images'][row['name']]['split'] == 'development':
            by_image[row['name']].append(row)
    names = sorted(n for n, v in models.manifest['images'].items() if v['split'] == 'development')
    assert set(names) == set(by_image) and len(names) == 371
    started = time.monotonic()
    for index, name in enumerate(names):
        path = record_dir/(name+'.json')
        if path.exists():
            item = json.loads(path.read_text())
            assert item['identity_hash'] == identity and item['image'] == name
        else:
            item = image_paths(name, models.manifest['images'][name], by_image[name], models, gain, center, identity)
            write_json(path, item)
        progress = {'images': index+1, 'total': len(names), 'elapsed_seconds': time.monotonic()-started,
                    'last_image': name, 'state': 'running'}
        write_json(args.run_dir/'progress.json', progress)
        print(json.dumps(progress), flush=True)
        if STOP:
            models.assert_unchanged()
            gain.assert_unchanged()
            write_json(args.run_dir/'interrupted.json', {**progress, 'state': 'interrupted_after_saved_image'})
            config = json.loads((args.run_dir/'config.json').read_text())
            config['state'] = 'interrupted_resumable'
            write_json(args.run_dir/'config.json', config)
            return 75
    models.assert_unchanged()
    gain.assert_unchanged()
    records = [json.loads((record_dir/(name+'.json')).read_text()) for name in names]
    result = {'identity_hash': identity, 'maximum_budget': models.maximum_budget,
              'matched_endpoints': aggregate(records, models.maximum_budget),
              'complete_budget_cohort': aggregate(records, models.maximum_budget, True),
              'policy_paths_exhausted': {policy: sum(p['states'][-1]['stopped'] for r in records for t in r['trials']
                                                    for p in t['paths'][policy]) for policy in POLICIES[1:]}}
    write_json(args.run_dir/'summary.json', result)
    report(args.run_dir, result)
    config = json.loads((args.run_dir/'config.json').read_text())
    config['state'] = 'completed'
    config['elapsed_seconds'] = time.monotonic()-started
    write_json(args.run_dir/'config.json', config)
    print(json.dumps({'state': 'completed', 'images': len(records), 'elapsed_seconds': config['elapsed_seconds']}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
