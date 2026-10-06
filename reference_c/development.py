"""All fixed development images: frozen DGIII, native pixels and content cells.

Execute only inside Slurm. No fitting, scale selection, holdout images, or resubmit.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
import torch.nn.functional as F
from deepgaze_pytorch import DeepGazeIII
from deepgaze_pytorch.modules import encode_scanpath_features
from observer import PartialObserver


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def summarize(rows):
    pixel = [r for r in rows if r['status'] in ('scored', 'border_target')]
    cell = [r for r in rows if r['status'] == 'scored']
    result = dict(next_fixations=len(rows), pixel_denominator=len(pixel),
                  cell_denominator=len(cell), border_targets=sum(r['status'] == 'border_target' for r in rows),
                  invalid=sum(r['status'] == 'invalid' for r in rows))
    for kind, eligible in [('pixel', pixel), ('cell', cell)]:
        if eligible:
            result[f'mean_{kind}_log_likelihood_nats'] = float(np.mean([r[f'{kind}_logp'] for r in eligible]))
            result[f'mean_{kind}_bits_over_centerbias'] = float(np.mean([
                r[f'{kind}_logp'] - r[f'centerbias_{kind}_logp'] for r in eligible]) / np.log(2))
    if pixel:
        result['content_mass_range'] = [min(r['content_mass'] for r in pixel), max(r['content_mass'] for r in pixel)]
        result['mean_content_mass'] = float(np.mean([r['content_mass'] for r in pixel]))
    return result


def selected_history(history, offsets):
    selected = np.full((len(offsets), 2), np.nan, np.float32)
    for i, offset in enumerate(offsets):
        if len(history) >= -offset:
            selected[i] = history[offset]
    return selected


class CachedImage:
    """Exact operations from pinned DeepGazeIIIMixture; cache only image terms."""
    def __init__(self, model, image, cb):
        self.model = model
        self.h, self.w = image.shape[:2]
        self.image = torch.as_tensor(image.transpose(2, 0, 1)[None], device='cuda', dtype=torch.float32)
        self.cb = torch.as_tensor(cb[None], device='cuda')
        self.readout_shape = [math.ceil(v / model.downsample / model.readout_factor) for v in (self.h, self.w)]
        with torch.inference_mode():
            features = model.features(F.interpolate(self.image, scale_factor=1 / model.downsample, recompute_scale_factor=False))
            features = torch.cat([F.interpolate(item, self.readout_shape) for item in features], dim=1)
            self.saliency = [network(features) for network in model.saliency_networks]

    def predict(self, histories):
        selected = np.stack([selected_history(h, self.model.included_fixations) for h in histories])
        xh = torch.as_tensor(selected[:, :, 0], device='cuda')
        yh = torch.as_tensor(selected[:, :, 1], device='cuda')
        with torch.inference_mode():
            encoded = encode_scanpath_features(xh, yh, (self.h, self.w), device='cuda')
            encoded = F.interpolate(encoded, self.readout_shape)
            outputs = []
            for cached, scanpath, selection, finalizer in zip(self.saliency, self.model.scanpath_networks,
                    self.model.fixation_selection_networks, self.model.finalizers):
                y = scanpath(encoded) if scanpath is not None else None
                x = selection((cached.expand(len(histories), -1, -1, -1), y))
                outputs.append(finalizer(x, self.cb.expand(len(histories), -1, -1))[:, None])
            output = (torch.cat(outputs, dim=1) - np.log(len(outputs))).logsumexp(dim=1, keepdim=True)
        return output[:, 0].cpu().numpy(), selected

    def verify_original(self, history, cached):
        selected = selected_history(history, self.model.included_fixations)
        with torch.inference_mode():
            original = self.model(self.image, self.cb,
                torch.as_tensor(selected[:, 0][None], device='cuda'),
                torch.as_tensor(selected[:, 1][None], device='cuda'))[0, 0].cpu().numpy()
        error = float(np.max(np.abs(original - cached)))
        if not np.allclose(original, cached, atol=1e-4, rtol=0):
            raise AssertionError(f'Cached/batched vs official forward mismatch: {error}')
        return dict(max_abs_logp_difference=error, atol=1e-4, rtol=0,
                    scope='First scored state, batched cached path versus official full forward')


def cell_probabilities(logp, grid):
    probability = np.exp(logp.astype(np.float64))
    total = float(probability.sum())
    if not np.isfinite(probability).all() or abs(total - 1) > 2e-5:
        raise AssertionError(f'Pixel probability invalid: {total}')
    xe, ye = grid['x_edges'], grid['y_edges']
    content = probability[ye[0]:ye[-1], xe[0]:xe[-1]]
    masses = np.add.reduceat(np.add.reduceat(content, ye[:-1] - ye[0], axis=0), xe[:-1] - xe[0], axis=1).ravel()
    mass = float(masses.sum())
    if mass <= 0 or (masses <= 0).any():
        raise AssertionError('Nonpositive content/cell probability')
    conditional = masses / mass
    if abs(conditional.sum() - 1) > 1e-10:
        raise AssertionError('Cell probabilities not normalized')
    return conditional, mass, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--resume', type=Path, help='Explicit previous run; accept only identical experiment contract')
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Execution must be inside Slurm')
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    started = time.monotonic()
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    torch.manual_seed(20260928)
    np.random.seed(20260928)
    manifest = json.loads(args.manifest.read_text())
    names = sorted(n for n, e in manifest['images'].items() if e['split'] == 'development')
    assert len(names) == 371 and manifest['display_size'] == [1680, 1050]
    # The existing source JSON combines splits. Filter immediately and never open,
    # process, or evaluate a holdout image; retain source hash for auditability.
    raw_fixations = Path(manifest['raw_fixations'])
    development_names = set(names)
    trials = [r for r in json.loads(raw_fixations.read_text()) if r['name'] in development_names]
    if {r['name'] for r in trials} != set(names):
        raise ValueError('Filtered fixations must contain exactly the full development set')
    assert all(r['condition'] == 'freeview' and r['split'] == 'train' for r in trials)
    assert len(trials) == 3698 and sum(len(r['X']) - 1 for r in trials) == 53127
    upstream = subprocess.check_output(['git', '-C', str(args.root / 'vendor'), 'rev-parse', 'HEAD'], text=True).strip()
    assert upstream == '874f12e1ee519860f49860638cf7f6375956d45a'
    contract = dict(schema=1, split='development', images=names, batch_size=args.batch_size,
        source_sha256=digest(__file__), observer_sha256=digest(Path(__file__).with_name('observer.py')),
        manifest_sha256=digest(args.manifest), raw_fixation_sha256=digest(raw_fixations), upstream_commit=upstream,
        checkpoint_sha256=digest(args.root / 'cache/torch/hub/checkpoints/deepgaze3.pth'),
        centerbias_sha256=digest(args.root / 'centerbias.npy'),
        preprocessing='Native 1680x1050 display RGB 0..255; no rescale or crop; raw chronological X/Y',
        history='All raw prefixes retained; official model uses offsets [-1,-2,-3,-4], NaN for missing older points',
        initial_fixation_scored=False, grid='D choice_grid(16), row-major content cells with integer pixel edges',
        pixel_score='Unconditional full-display pixel probability, target floor(X),floor(Y)',
        cell_score='Pixel mass summed in each content cell, conditioned on next fixation being in the content box',
        border_targets='Pixel scored, cell excluded; preserved in all later raw histories',
        centerbias='External official MIT1003 template; identical cell aggregation/conditioning; no dev fitting',
        interpretation='Descriptive full-development reference; SALICON/COCO pretraining image overlap unresolved; not a pretraining-independent benchmark',
        source_filter='Read existing combined raw JSON, immediately retain only manifest development names; no holdout images/features/scores',
        fit_or_adapt=False, holdout_images_opened=0)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'contract.json', contract)
    previous = None
    if args.resume:
        previous = args.resume.resolve()
        if json.loads((previous / 'contract.json').read_text()) != contract:
            raise ValueError('Resume contract differs: refusing to mix experiments')
    model = DeepGazeIII(pretrained=True).eval().cuda()
    model.requires_grad_(False)
    template = np.load(args.root / 'centerbias.npy')
    cb = zoom(template, (1050 / template.shape[0], 1680 / template.shape[1]), order=0, mode='nearest')
    cb = (cb - logsumexp(cb)).astype(np.float32)
    rows_all, images_summary, equivalence = [], {}, None
    for image_index, name in enumerate(names):
        save = args.output / 'images' / Path(name).stem
        if previous and (previous / 'images' / Path(name).stem / 'complete.json').exists():
            import shutil
            previous_image = previous / 'images' / Path(name).stem
            completed = json.loads((previous_image / 'complete.json').read_text())
            assert digest(previous_image / 'scores.json') == completed['scores_sha256']
            assert digest(previous_image / 'cell_probabilities.npz') == completed['cells_sha256']
            shutil.copytree(previous / 'images' / Path(name).stem, save, dirs_exist_ok=True)
            rows = json.loads((save / 'scores.json').read_text())
            rows_all.extend(rows)
            images_summary[name] = json.loads((save / 'complete.json').read_text())
            print(f'Resumed {image_index + 1}/{len(names)} {name}', flush=True)
            continue
        save.mkdir(parents=True, exist_ok=True)
        entry = manifest['images'][name]
        assert entry['official_split'] == 'train' and entry['split'] == 'development'
        if digest(entry['path']) != entry['sha256']:
            raise ValueError(f'Image hash mismatch: {name}')
        with Image.open(entry['path']) as source:
            pil = source.convert('RGB')
            grid = PartialObserver(pil, content_box=entry['display_content_box']).choice_grid()
            image = np.array(pil)
        assert image.shape == (1050, 1680, 3)
        xe, ye = grid['x_edges'], grid['y_edges']
        left, top, right, bottom = entry['display_content_box']
        assert (xe[0], ye[0], xe[-1], ye[-1]) == (left, top, right, bottom)
        assert np.issubdtype(xe.dtype, np.integer) and np.issubdtype(ye.dtype, np.integer)
        assert (np.diff(xe) > 0).all() and (np.diff(ye) > 0).all()
        assert grid['xy'].shape == (grid['shape'][0] * grid['shape'][1], 2)
        assert int(np.outer(np.diff(ye), np.diff(xe)).sum()) == (right - left) * (bottom - top)
        cb_cells, cb_mass, _ = cell_probabilities(cb, grid)
        cached = CachedImage(model, image, cb)
        rows, states, cell_vectors, selected_vectors = [], [], [], []
        subjects = sorted((r for r in trials if r['name'] == name), key=lambda r: r['subject'])
        for trial in subjects:
            xy = np.column_stack([trial['X'], trial['Y']]).astype(np.float32)
            for t in range(1, len(xy)):
                row = dict(image=name, subject=trial['subject'], fixation_index=t,
                    x=float(xy[t, 0]), y=float(xy[t, 1]), history_xy=xy[:t].tolist(), history_length=t,
                    status='invalid', target_cell_index=None, cell_probability_row=None,
                    pixel_logp=None, centerbias_pixel_logp=None, cell_logp=None,
                    centerbias_cell_logp=None, content_mass=None, centerbias_content_mass=cb_mass)
                rows.append(row)
                if not np.isfinite(xy[:t + 1]).all() or not ((xy[:t + 1] >= 0).all() and (xy[:t + 1] < [1680, 1050]).all()):
                    row['reason'] = 'Nonfinite or outside-display raw prefix/target; history not repaired'
                    continue
                states.append((row, xy[:t]))
        for offset in range(0, len(states), args.batch_size):
            batch = states[offset:offset + args.batch_size]
            maps, selected = cached.predict([history for _, history in batch])
            if equivalence is None:
                equivalence = cached.verify_original(batch[0][1], maps[0])
                write_json(args.output / 'forward_equivalence.json', equivalence)
                print('Official-forward equivalence:', equivalence, flush=True)
            for (row, history), logp, hist in zip(batch, maps, selected):
                cells, mass, total = cell_probabilities(logp, grid)
                x, y = int(math.floor(row['x'])), int(math.floor(row['y']))
                xe, ye = grid['x_edges'], grid['y_edges']
                content_target = xe[0] <= x < xe[-1] and ye[0] <= y < ye[-1]
                row.update(status='scored' if content_target else 'border_target', pixel_logp=float(logp[y, x]),
                    centerbias_pixel_logp=float(cb[y, x]), content_mass=mass, pixel_probability_sum=total,
                    cell_probability_row=len(cell_vectors))
                if content_target:
                    cell = int((np.searchsorted(ye, y, side='right') - 1) * grid['shape'][1]
                        + np.searchsorted(xe, x, side='right') - 1)
                    row.update(target_cell_index=cell, cell_logp=float(np.log(cells[cell])),
                        centerbias_cell_logp=float(np.log(cb_cells[cell])))
                cell_vectors.append(cells.astype(np.float32))
                selected_vectors.append(hist)
        write_json(save / 'scores.json', rows)
        np.savez_compressed(save / 'cell_probabilities.npz',
            probabilities=np.asarray(cell_vectors, dtype=np.float32),
            centerbias_probabilities=cb_cells.astype(np.float32),
            selected_histories_xy=np.asarray(selected_vectors, dtype=np.float32),
            xy=grid['xy'], x_edges=grid['x_edges'], y_edges=grid['y_edges'], grid_shape=grid['shape'])
        summary = dict(**summarize(rows), subjects=len(subjects), image_sha256=entry['sha256'],
            content_box=entry['display_content_box'], grid_shape=list(grid['shape']),
            scores_sha256=digest(save / 'scores.json'), cells_sha256=digest(save / 'cell_probabilities.npz'))
        write_json(save / 'complete.json', summary)
        rows_all.extend(rows)
        images_summary[name] = summary
        elapsed = time.monotonic() - started
        write_json(args.output / 'progress.json', dict(completed_images=len(images_summary), expected_images=len(names),
            next_fixations=len(rows_all), elapsed_seconds=elapsed, current_image=name))
        print(f'Completed {image_index + 1}/{len(names)} {name}: {summary["pixel_denominator"]} pixel, '
              f'{summary["cell_denominator"]} cell; elapsed {elapsed:.1f}s', flush=True)
        del cached
    final = dict(**summarize(rows_all), completed=True, images=len(names),
        trials=len(trials), job=os.environ['SLURM_JOB_ID'], elapsed_seconds=time.monotonic() - started,
        peak_cuda_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
        device=torch.cuda.get_device_name(), torch=torch.__version__, numpy=np.__version__,
        forward_equivalence=equivalence, resumed_from=str(previous) if previous else None,
        interpretation=contract['interpretation'], per_image=images_summary)
    assert len(rows_all) == 53127
    if equivalence is None and previous:
        final['forward_equivalence'] = json.loads((previous / 'forward_equivalence.json').read_text())
        write_json(args.output / 'forward_equivalence.json', final['forward_equivalence'])
    for kind in ('pixel', 'cell'):
        key = f'mean_{kind}_log_likelihood_nats'
        final[f'equal_image_{key}'] = float(np.mean([r[key] for r in images_summary.values() if key in r]))
    with (args.output / 'scores.csv').open('w') as stream:
        keys = sorted({k for row in rows_all for k in row})
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows_all:
            writer.writerow({**row, 'history_xy': json.dumps(row['history_xy'], separators=(',', ':'))})
    write_json(args.output / 'summary.json', final)
    print(json.dumps({k: v for k, v in final.items() if k != 'per_image'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
