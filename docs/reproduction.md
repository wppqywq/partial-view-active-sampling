# Reproduction

## What this release contains

This is a source-and-results snapshot of an ongoing research experiment. It includes current training/evaluation code, the inherited helper sources, data/reference entry points, and small aggregate outputs. It excludes raw images, fixation records, pretrained weights, trained checkpoints, caches, personal notes, slide decks, and full execution logs.

Core package versions are in `requirements.txt`. DINOv2 was loaded from revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`. The pretrained ViT-S/14 checkpoint SHA-256 is `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`. External packages and data keep their own licenses and access conditions.

## Rebuild the public figures

| Scope | Current reproducibility |
| --- | --- |
| Aggregate figures | Rebuilt successfully from the bundled JSON files |
| Gaussian example | Original numerical code and results included |
| Dataset and image split | Public inputs, recorded checksums, deterministic rule, and preparation helper provided; new helper not redownload-tested |
| Full training on this server | Existing frozen inputs and environment are required |
| Full training on a fresh server | Not yet standalone: absolute paths, archived helper/checkpoint dependencies, manifest contracts, and reference score files require porting or regeneration |

Start with [dataset acquisition](dataset.md). Do not confuse figure reproduction with reproducing model training from raw data. The exact dataset archives and format description must be obtained from the authors.

No dataset or trained model is needed. With NumPy and Matplotlib installed:

```bash
python scripts/plot_results.py
```

On the research server, numerical work must run in Slurm:

```bash
sbatch scripts/build_public_figures.sh
```

The figure script reads only `results/*.json` and writes `results/figures/`. It does not touch active training outputs. `scripts/check_release.sh` checks ASCII text, repository contents, aggregate checksums, and shell syntax. The figure job also parses Python sources without importing or executing training modules.

## Existing research server

Training is site-specific. The original paths intentionally remain in scientific source snapshots, because source hashes are bound to checkpoints. This public cleanup did not migrate data or rewrite training code.

| Input or output | Existing location |
| --- | --- |
| Workspace | `/home/youyouyang/proposal2` |
| Data manifest | `/mnt/disk2/youyouyang/proposal2/coco_freeview/manifest.json` |
| DINO source and complete targets | `/mnt/disk2/youyouyang/proposal2/predictor_d` |
| Current experiment outputs | `/mnt/disk2/youyouyang/proposal2/foveated_v3` |
| Existing Python environment | `/mnt/disk2/youyouyang/deepgaze_video/envs/official/bin/python` |

The wrappers use one GPU at a time, four CPU cores, and 24 GB host memory. Do not run training or numerical diagnostics on the login node. Preparation and figure rendering use short CPU jobs. Use the site's partition, GPU, and time limits rather than copying these requests to an unrelated cluster.

The sequence is:

1. `run_prepare.sh`: lock the training-derived budget and render observation previews.
2. `run_mean.sh`: use `FV3_SOURCE_DIR`, `FV3_BUDGET_JSON`, and an actual `FV3_VISUAL_REVIEW_JSON` to train the mean.
3. `run_variance.sh --mean-run ABSOLUTE_MEAN_RUN`: freeze the completed mean and fit variance.
4. Inspect calibration results and record the checkpoint-bound review. A review must reflect actual results; it is not a bypass flag.
5. `run_gain.sh`: set `FV3_SOURCE_DIR`, `FV3_MEAN_RUN`, and `FV3_VARIANCE_RUN` to generate fresh labels and train G.
6. `run_evaluate.sh` and `run_human_choice.sh`: additionally set `FV3_GAIN_RUN`; submit sequentially under the single-GPU limit.

Resume only from the matching immutable source/protocol/cache identity. Do not submit a second copy of a live stage. Do not change an active job's code, checkpoints, source paths, or caches.

Mean and variance are already complete in this snapshot. The active gain continuation uses frozen sources under run 22779. No existing experiment needs to be restarted because of this cleanup.

## Another server

A full rerun requires obtaining the data and pretrained models, recreating the documented preprocessing and image split, and configuring site-specific paths and Slurm launchers. Several Python constants also refer to archived source/checkpoint paths. Editing a launcher alone is insufficient. The `support/` sources expose the inherited implementation, but public checkpoint bundles and a portable configuration layer have not yet been released.

Do not claim bitwise reproduction from this source-only release. Compare preprocessing, target normalization, split hashes, forward precision/batch conventions, and aggregate metrics. The historical project audit observed small forward differences across execution configurations; fixed hardware settings are part of the experiment provenance.
