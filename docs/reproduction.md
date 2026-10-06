# Reproduction

## Dataset setup

Use the five official downloads linked in the main README (about 2.5 GB compressed). Python 3.12, the packages in `requirements.txt`, curl, ripgrep, and sha256sum are required. On a Slurm cluster:

```bash
export PROJECT_CODE="$PWD"
export DATA_ROOT=/path/to/new/external/coco_freeview
export PROJECT_PYTHON=/path/to/python
sbatch scripts/prepare_dataset.sh
```

Adapt the partition and resource directives to the cluster. Use a new external destination: the helper refuses to overwrite an existing manifest. It downloads missing files, checks `data_a/RAW_SHA256SUMS`, and runs the original audit. The helper was statically checked; its full download has not been repeated. If Google Drive returns a login page, obtain the actual `readme.txt` from the authors and place it in `DATA_ROOT/raw/`.

The audit creates `manifest.json`, `audit.json`, extracted display images, and coordinate overlays. Image coordinates refer to the supplied 1680 x 1050 displays. COCO caption metadata supplies original dimensions for recovering content bounds; no original COCO image archive is needed.

For the split, sort unique official training image names by hexadecimal SHA-256 of `20260928:development:IMAGE_FILENAME`. The first `round(0.1 * N)` form development; the remainder form training. Official validation images form the internal holdout. All participants for an image remain together. Expected sizes are 3343/371/603. Absolute paths make manifest byte hashes installation-specific.

## Exact observation settings

Resize bilinearly to 448 x 280 and pad to multiples of 14. Let `d` be distance from fixation, `r0 = 22.4` pixels, and `r50 = 39.2` pixels:

```text
q(d) = max(1/16, 1 / (1 + (max(d - r0, 0) / (r50 - r0))^2))
sigma(q) = 0.8 * sqrt(q^(-2) - 1)
```

Take the pointwise maximum q over the history; an empty history has q = 1/16. Blur the resized RGB at sigma 0, 0.25, 0.5, 1, 2, 4, 8, and 16, interpolating adjacent layers in Gaussian variance. There is no downsampling pyramid. Scales are relative to the full display, not its letterboxed content. Local U and local gain use content-masked Gaussian weights with sigma = r50. Model policies require a mean content-resolution increase above 1e-8; human-choice candidates retain revisits. A repeated identical observation changes no pixels.

## Training dependencies

These are the original experiment entry points, not a portable training package. The current mean, variance, gain, and evaluation sources are included. Older experiments are represented by aggregate results; their full pipelines are not included.

| Required artifact | Original source or generator |
| --- | --- |
| DINOv2 ViT-S/14 source and weights | `facebookresearch/dinov2`, revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8` |
| Dataset manifest | `data_a/audit.py` |
| Full-image targets and channel normalization | Original `predictor_d` cache; extraction utilities in `support/frozen_data.py` |
| Direct mean architecture and encoder | `support/frozen_mean.py` |
| Legacy preprocessing | `support/frozen_legacy_observer.py` |
| Preparation comparison against rejected r1 | Archived `foveated_v3/prepare/22738/observer.py` |
| Current preparation protocol and visual review | `foveated_v3/prepare.py`, followed by inspection |
| Mean checkpoint and source identity | `foveated_v3/train_mean.py` |
| Variance checkpoint and calibration review | `foveated_v3/train_variance.py`, followed by calibration review |
| Fresh gain labels and checkpoint | `foveated_v3/train_gain.py` |
| Matched DeepGaze scores | `reference_c/development.py` |

DINO weight SHA-256: `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.

Launchers assemble an isolated run directory: mean training copies the three helper files, and downstream stages copy the exact sources from the selected mean run. `support/` exposes those implementations but does not redirect the original launchers. The public legacy-observer copy has ASCII-only docstrings, so its hash differs from the immutable run copy.

On the original server, data and experiment artifacts live under `/mnt/disk2/youyouyang/proposal2`. The reused interpreter is `/mnt/disk2/youyouyang/deepgaze_video/envs/official/bin/python`, whose base interpreter also lives in that scratch tree. Keep both while jobs depend on them.

To rerun elsewhere, configure Python constants as well as launchers, regenerate the target cache and normalization, and create new manifest/protocol/source identities. Do not bypass hash checks or substitute edited files into an existing frozen run. The rejected-r1 preview dependency and archived DeepGaze score path also need replacement or regeneration. Public checkpoint bundles and automated bootstrap from raw data are not yet provided.

## Execution

All numerical work on the research server runs through Slurm. GPU launchers use one GPU, four CPUs, and 24 GB host memory; submit sequentially under its single-GPU quota.

| Stage | Launcher and required inputs |
| --- | --- |
| Observation preparation | `run_prepare.sh`; `FV3_SOURCE_DIR` |
| Mean | `run_mean.sh`; additionally `FV3_BUDGET_JSON`, `FV3_VISUAL_REVIEW_JSON` |
| Variance | `run_variance.sh --mean-run /absolute/mean/run` |
| Gain | `run_gain.sh`; `FV3_MEAN_RUN`, `FV3_VARIANCE_RUN` |
| Closed-loop and replay | `run_evaluate.sh`; additionally `FV3_GAIN_RUN` |
| Human choice | `run_human_choice.sh`; same frozen models and matching reference scores |

The original launchers preserve source, protocol, checkpoint, and cache identities. Resume only compatible runs. Development selects checkpoints; it is not an independent final evaluation. Exact floating-point reproduction across GPU configurations has not been established.

`bash scripts/check_release.sh` checks shell syntax, ASCII text, ignore rules, and aggregate checksums without starting training. `results/provenance.json` identifies the runs underlying the README tables.
