"""Audit official FreeView files without changing gaze records. Run inside Slurm."""
import collections
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zipfile

import numpy as np
from PIL import Image

SEED = 20260928
DISPLAY = (1680, 1050)


def geometry(width, height):
    """Official centered letterbox; integer resized dimensions, half-open bounds."""
    if width / height > DISPLAY[0] / DISPLAY[1]:
        rw, rh = DISPLAY[0], int(DISPLAY[0] * height / width)
    else:
        rw, rh = int(DISPLAY[1] * width / height), DISPLAY[1]
    left, top = (DISPLAY[0] - rw) // 2, (DISPLAY[1] - rh) // 2
    return [left, top, left + rw, top + rh]


def original_xy(row, entry):
    """Display -> original COCO coordinates (NOT the supplied padded JPEG)."""
    left, top, right, bottom = entry['display_content_box']
    w, h = entry['original_size']
    return ((np.asarray(row['X']) - left) * w / (right - left),
            (np.asarray(row['Y']) - top) * h / (bottom - top))


def digest(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def ranked(names, purpose):
    return sorted(names, key=lambda n: hashlib.sha256(f'{SEED}:{purpose}:{n}'.encode()).hexdigest())


def describe(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return dict(zip(['min', 'p25', 'median', 'p75', 'max'], np.percentile(a, [0, 25, 50, 75, 100]))) | {'mean': float(a.mean())}


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def overlays(rows, entries, chosen, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    subjects = sorted({r['subject'] for r in rows})
    by_image = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_image[r['name']].append(i)
    sheet, axes = plt.subplots(5, 4, figsize=(16, 14))
    selections = []

    def draw(ax, im, x, y, title):
        ax.imshow(im)
        ax.plot(x, y, color='#00cfff', linewidth=1, alpha=.85)
        ax.scatter(x, y, c=np.arange(len(x)), cmap='autumn', s=30, edgecolor='black', linewidth=.4)
        ax.scatter(x[0], y[0], marker='*', s=170, c='lime', edgecolor='black', zorder=5)
        for j, (xx, yy) in enumerate(zip(x, y)):
            ax.annotate(str(j), (xx, yy), fontsize=7, color='white', xytext=(3, 3), textcoords='offset points',
                        bbox={'facecolor': 'black', 'alpha': .6, 'pad': .3, 'edgecolor': 'none'})
        ax.set_xlim(-.5, im.width - .5)
        ax.set_ylim(im.height - .5, -.5)
        ax.set_title(title, fontsize=9)
        ax.axis('off')

    with PdfPages(output / 'overlays.pdf') as pdf:
        for k, name in enumerate(chosen):
            candidates = sorted(by_image[name], key=lambda i: rows[i]['subject'])
            idx = next((i for i in candidates if rows[i]['subject'] == subjects[k % len(subjects)]), candidates[0])
            r, entry = rows[idx], entries[name]
            with Image.open(entry['path']) as source:
                im = source.convert('RGB')
            x, y = original_xy(r, entry)
            left, top, right, bottom = entry['display_content_box']
            crop = im.crop((left, top, right, bottom)).resize(entry['original_size'])
            label = f'{name} | subject {r["subject"]} | n={len(x)}'
            draw(axes.flat[k], im, r['X'], r['Y'], label)
            fig, pair = plt.subplots(1, 2, figsize=(14, 5))
            draw(pair[0], im, r['X'], r['Y'], 'Downloaded display JPEG (1680 x 1050); raw gaze coordinates')
            draw(pair[1], crop, x, y, 'Content crop reconstructed at COCO native size; converted gaze')
            fig.suptitle(label + '\nGreen star = initial fixation 0 (condition only); numbers = original order')
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            selections.append({'name': name, 'row_index': idx, 'subject': r['subject']})
    sheet.suptitle('20 fixed training images | original display coordinates | green star = initial fixation 0', fontsize=13)
    sheet.tight_layout(rect=[0, 0, 1, .97])
    sheet.savefig(output / 'overlays.png', dpi=150)
    plt.close(sheet)
    return selections


def main(root, output):
    assert os.environ.get('SLURM_JOB_ID'), 'Use a Slurm allocation'
    started = time.time()
    raw = root / 'raw'
    readme = (raw / 'readme.txt').read_text()
    assert '1680x1050' in readme and 'including the initial center fixation' in readme
    rows = json.loads((raw / 'fixations.json').read_text())
    assert isinstance(rows, list) and rows
    assert {r['condition'] for r in rows} == {'freeview'}
    official = {s: {r['name'] for r in rows if r['split'] == s} for s in ('train', 'valid')}
    assert {r['split'] for r in rows} == set(official), 'Unexpected official split'
    assert not official['train'] & official['valid'], 'Image leakage across official splits'
    names = official['train'] | official['valid']
    assert all(Path(n).name == n and n.endswith('.jpg') for n in names)
    images = root / 'images'
    images.mkdir(exist_ok=True)
    native = {}
    with zipfile.ZipFile(raw / 'annotations_trainval2014.zip') as z:
        for split in ('train', 'val'):
            meta = json.loads(z.read(f'annotations/captions_{split}2014.json'))
            native.update({f'{v["id"]:012d}.jpg': [v['width'], v['height']] for v in meta['images']})
            del meta
    entries, duplicates, archive_counts = {}, [], {}
    for archive in sorted(raw.glob('COCOSearch18-images-*.zip')):
        with zipfile.ZipFile(archive) as z:
            members = [m for m in z.infolist() if Path(m.filename).suffix.lower() == '.jpg' and '__MACOSX' not in m.filename]
            archive_counts[archive.name] = len(members)
            for member in members:
                name = Path(member.filename).name
                if name not in names:
                    continue  # Do not extract images without released train/valid gaze.
                data = z.read(member)  # ZIP checks CRC; decode every selected image below.
                sha = hashlib.sha256(data).hexdigest()
                with Image.open(io.BytesIO(data)) as im:
                    im.load()
                    w, h = im.size
                    assert (w, h) == DISPLAY, f'Unexpected supplied JPEG geometry: {name}'
                    box = geometry(*native[name])
                    left, top, right, bottom = box
                    a = np.asarray(im.convert('RGB'))
                    strips = [a[::8, :max(0, left-8):8], a[::8, right+8::8],
                              a[:max(0, top-8):8, ::8], a[bottom+8::8, ::8]]
                    padding_error = max([float((s > 16).mean()) for s in strips if s.size] or [0.])
                    assert padding_error < .01, f'COCO geometry does not match black padding: {name}'
                if name in entries:
                    assert entries[name]['sha256'] == sha, f'Conflicting images named {name}'
                    duplicates.append([name, archive.name, member.filename])
                    continue
                path = images / name
                if not path.exists() or digest(path) != sha:
                    path.write_bytes(data)
                entries[name] = {'path': str(path), 'size': [w, h], 'original_size': native[name], 'sha256': sha,
                                 'archive': archive.name, 'member': member.filename,
                                 'display_content_box': box, 'padding_nonblack_fraction': padding_error}
    assert set(entries) == names, f'Missing {len(names - set(entries))} images'
    dev = set(ranked(official['train'], 'development')[:round(len(official['train']) * .1)])
    train = official['train'] - dev
    smoke = ranked(train, 'smoke')[:200]
    chosen = ranked(smoke, 'overlay')[:20]
    assert len(smoke) == 200 and len(chosen) == 20 and not set(smoke) & dev
    for name, entry in entries.items():
        entry.update(official_split='train' if name in official['train'] else 'valid',
                     split='holdout' if name in official['valid'] else 'development' if name in dev else 'train',
                     smoke=name in smoke)
    fields = sorted(set().union(*(r.keys() for r in rows)))
    errors, counts, starts, lengths, values = [], collections.Counter(), [], [], collections.defaultdict(list)
    per_split = collections.defaultdict(lambda: collections.Counter())
    for i, r in enumerate(rows):
        xyz = [np.asarray(r[k], dtype=float) for k in ('X', 'Y', 'T')]
        assert all(v.ndim == 1 for v in xyz) and len({len(v) for v in xyz}) == 1, f'Broken arrays at row {i}'
        x, y, t = xyz
        assert len(x) == r['length'] and len(x) > 0, f'Length mismatch at row {i}'
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(t)
        inside_display = (x >= 0) & (x < DISPLAY[0]) & (y >= 0) & (y < DISPLAY[1])
        entry = entries[r['name']]
        left, top, right, bottom = entry['display_content_box']
        inside_image = (x >= left) & (x < right) & (y >= top) & (y < bottom)
        flags = {'nonfinite': ~finite, 'outside_display': ~inside_display,
                 'outside_image': ~inside_image, 'nonpositive_duration': t <= 0}
        bad = {k: np.flatnonzero(v).tolist() for k, v in flags.items() if v.any()}
        if bad:
            errors.append({'row_index': i, 'name': r['name'], 'subject': r['subject'], **bad})
        for key, mask in flags.items():
            counts[key] += int(mask.sum())
            counts[key + '_after_initial'] += int(mask[1:].sum())
        ix, iy = original_xy(r, entry)
        assert np.allclose(ix[finite] * (right-left) / entry['original_size'][0] + left, x[finite])
        assert np.allclose(iy[finite] * (bottom-top) / entry['original_size'][1] + top, y[finite])
        starts.append(float(np.hypot(x[0]-840, y[0]-525)))
        lengths.append(len(x))
        for key, v in zip(('X', 'Y', 'T'), xyz):
            values[key].extend(v.tolist())
        per_split[entry['split']].update(scanpaths=1, fixations=len(x), next_fixations=len(x)-1)
    pairs = collections.Counter((r['name'], r['subject']) for r in rows)
    row_hashes = collections.Counter(hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest() for r in rows)
    for split, stats in per_split.items():
        stats['images'] = sum(e['split'] == split for e in entries.values())
        stats['observers'] = len({r['subject'] for r in rows if entries[r['name']]['split'] == split})
    summary = {'scanpaths': len(rows), 'images': len(names), 'fixations': sum(lengths),
               'observers': sorted({r['subject'] for r in rows}), 'fields': fields,
               'missing_fields': {k: sum(k not in r for r in rows) for k in fields},
               'null_fields': {k: sum(r.get(k) is None for r in rows) for k in fields},
               'scanpath_length': describe(lengths), 'coordinates_and_raw_duration': {k: describe(v) for k, v in values.items()},
               'official_splits': {s: {'images': len(ns), 'scanpaths': sum(r['split'] == s for r in rows)} for s, ns in official.items()},
               'splits': dict(per_split), 'issues': dict(counts), 'issue_rows': errors,
               'duplicate_image_subject_pairs': sum(v-1 for v in pairs.values()),
               'exact_duplicate_rows': sum(v-1 for v in row_hashes.values()),
               'archive_jpg_counts': archive_counts, 'identical_image_copies': len(duplicates),
               'image_sizes': dict(collections.Counter(f'{e["size"][0]}x{e["size"][1]}' for e in entries.values())),
               'images_with_padding': sum(e['display_content_box'] != [0, 0, *DISPLAY] for e in entries.values()),
               'padding_check_max_nonblack_fraction': max(e['padding_nonblack_fraction'] for e in entries.values()),
               'first_distance_from_center_px': describe(starts),
               'first_within_50px_of_center': sum(d <= 50 for d in starts)}
    manifest = {'seed': SEED, 'display_size': list(DISPLAY), 'raw_fixations': str(raw / 'fixations.json'),
                'initial_fixation': 'Index 0: retain as observed history, never score as a free choice; do not force to center.',
                'coordinate_mapping': 'JPEGs already have display padding: use raw X/Y directly. Content box comes from official COCO original dimensions; original_xy inverses integer resize and centered padding.',
                'split_rule': 'SHA256(seed:development:name), first round(10% official train) -> development; official valid -> internal holdout.',
                'smoke_images': smoke, 'images': dict(sorted(entries.items()))}
    manifest['overlays'] = overlays(rows, entries, chosen, output)
    save(root / 'manifest.json', manifest)
    save(root / 'audit.json', summary)
    commit = subprocess.run(['git', '-C', str(output), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    provenance = {'job_id': os.environ['SLURM_JOB_ID'], 'seed': SEED, 'python': platform.python_version(),
                  'numpy': np.__version__, 'pillow': Image.__version__, 'argv': sys.argv,
                  'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(started)),
                  'source_page': 'https://sites.google.com/view/cocosearch/coco-freeview',
                  'download_base': 'http://vision.cs.stonybrook.edu/~cvlab_download/',
                  'geometry_source': 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip (images metadata only)',
                  'readme_url': 'https://drive.google.com/uc?export=download&id=1Hj_jyK8Ml27Ge_5sogEtyI7XOjyad4aj',
                  'git_commit': commit.stdout.strip() if commit.returncode == 0 else None,
                  'git_note': 'No repository available; source snapshots and SHA256 identify this run.',
                  'sha256': {str(p): digest(p) for p in [*sorted(raw.iterdir()), Path(__file__), output / 'run.sh', root / 'manifest.json'] if p.is_file()},
                  'elapsed_seconds': round(time.time()-started, 2)}
    save(root / 'runs' / os.environ['SLURM_JOB_ID'] / 'provenance.json', provenance)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('issue_rows', 'image_sizes')}, indent=2))
    print(f'Complete: {root}; elapsed={provenance["elapsed_seconds"]}s')


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
