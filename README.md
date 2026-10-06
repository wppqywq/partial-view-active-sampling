# Partial-view active sampling

Does gaze follow uncertainty, or the expected benefit of looking? We compare local uncertainty (U), expected local prediction gain (local G), and expected whole-image gain (global G), controlling for spatial preference, observation history, and newly available evidence.

## Method

**Observation.** Images are resized to 448 x 280. Each fixation provides a sharp core with gradually blurred surroundings; previously seen detail is retained. Core and half-sharpness diameters are 44.8 and 78.4 pixels, engineering settings rather than calibrated human acuity. Black borders are excluded from scoring.

**Prediction.** Frozen [DINOv2 ViT-S/14](https://github.com/facebookresearch/dinov2) encodes partial RGB. A head predicts the full-image 20 x 32 x 384 feature map: 1 x 1 projection to 128 channels, two 3 x 3 convolutions (dilation 1 and 2), and 1 x 1 output, with GELU. Targets use training-only normalization. Mean training uses MSE, AdamW, learning-rate decay, and early stopping, capped at 80 epochs. The mean is then frozen while an independent variance head learns Gaussian negative log-likelihood.

**Values.** U averages predicted variance around a candidate. Gain labels measure the decrease in squared feature error after revealing it, locally or across the image. U and local gain share fixed Gaussian spatial weights. A 2309-256-64-2 MLP predicts both gains from the current state and candidate geometry. Negative gains are retained. Observer changes require fresh labels and G training; online choices never access hidden targets.

**Experiments.** Mean/variance training covers 1-16 observations; gain labels use before-histories of 1-15. Sixteen is the ceiling of the training mean trajectory length (15.46). Closed-loop policies maximize U or G on a grid with 16 cells along its longer side; center and uniform sampling are baselines. Human replay retains actual trajectories and revisits. Policies share the human start and budget, min(trajectory length, 16). Average seeds within trial, trials within image, then images equally.

Human-choice conditional logit adds each signal to position, movement, resolution-change, and history controls. The first fixation is not scored. [DeepGaze III](https://github.com/matthias-k/DeepGaze) provides a full-image reference on matching targets. Reconstruction and human-choice prediction are separate outcomes.

## Results

October 6, 2026: current mean and variance training are complete. Fresh gain labels are being generated; policy and human-choice results are pending.

| Current observer: development set | Result |
| --- | ---: |
| Mean predictor feature MSE | 0.5618 |
| Partial-view DINO features without prediction: MSE | 0.8476 |
| Training-mean feature baseline: MSE | 0.9976 |
| Predicted variance / observed squared error | 0.9938 |
| Within-state variance/error rank correlation | 0.7642 |

Mean results cover 371 images and 11,271 fixed human/random states, averaging budgets within history type and images equally. Best epoch: 80; convergence is not established. Calibration shares checkpoint-selection data.

The **earlier hard-window observer** used eight observations. These completed results belong to that earlier setup.

| Trajectory | Feature MSE at observation 8 |
| --- | ---: |
| Human | 0.7185 |
| U | 0.7184 |
| Local G | 0.7114 |
| Global G | 0.7133 |
| Center prior | 0.7169 |
| Uniform | 0.7210 |

Across 364 matched images with one human trajectory per image and shared starts, local G has the lowest mean error, but differences are small.

| Earlier human-choice model | Mean target log probability (higher is better) |
| --- | ---: |
| Controls | -4.2018 |
| Controls + U | -4.1952 |
| Controls + local G | -4.1388 |
| Controls + global G | -4.1736 |
| DeepGaze III | -3.8543 |

Human-choice scores cover 17,837 targets on 603 internal holdout images after 1, 4, or 7 observations. Local G exceeds U by 0.0564 nats (paired-image 97.5% interval: 0.0468-0.0660; two-comparison adjustment). This supports prediction in that setup, not a human mechanism. The holdout has already been inspected. Data: [results/](results/).

Next: finish G training, then evaluate current policies and human choices on matched budgets.

## Data

[COCO-FreeView](https://sites.google.com/view/cocosearch/coco-freeview): 4,317 images, 10 participants, 43,048 trajectories, and 665,005 fixations in the public training/validation release. Image-disjoint splits contain 3,343 training, 371 development, and 603 internal holdout images; the holdout is the official validation split, not the withheld test set.

Required downloads: [FreeView fixations](https://vision.cs.stonybrook.edu/~cvlab_download/COCOFreeView_fixations_trainval.json), [target-present images](https://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-images-TP.zip), [target-absent images](https://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-images-TA.zip), [COCO 2014 annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip), and the [coordinate specification](https://drive.google.com/file/d/1Hj_jyK8Ml27Ge_5sogEtyI7XOjyad4aj/view). Use the supplied display images and FreeView fixations, not the search-task trajectories. Checksums are in [data_a/RAW_SHA256SUMS](data_a/RAW_SHA256SUMS).

## Code

| Directory | Purpose |
| --- | --- |
| `foveated_v3/` | Current observation model, training, closed-loop evaluation, human choice |
| `support/` | Frozen encoder, mean head, and data utilities inherited by current training |
| `data_a/` | Dataset audit and deterministic split |
| `reference_c/` | DeepGaze III reference scoring |
| `d_partial/` | Original hard-window preprocessing used by the reference |
| `results/` | Aggregate measurements and provenance |
| `scripts/` | Dataset setup and repository checks |

Python dependencies are in [requirements.txt](requirements.txt). Training order is `run_prepare.sh`, `run_mean.sh`, `run_variance.sh`, `run_gain.sh`, then `run_evaluate.sh` and `run_human_choice.sh`, all under `foveated_v3/`.

Launchers still require server-specific paths, cached targets, and frozen run artifacts. [Reproduction](docs/reproduction.md) covers these dependencies, dataset setup, and exact foveation settings. Foveation draws on [Image_Foveation_Python](https://github.com/ouyangzhibo/Image_Foveation_Python).
