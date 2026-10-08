# Partial-view active sampling

Does gaze follow uncertainty, or the expected benefit of looking? We compare local uncertainty (U), expected local prediction gain (local G), and expected whole-image gain (global G), controlling for spatial preference, observation history, and newly available evidence.

## Method

**Observation.** Images are resized to 448 x 280. Each fixation provides a sharp core with gradually blurred surroundings; previously seen detail is retained. Core and half-sharpness diameters are 44.8 and 78.4 pixels, engineering settings rather than calibrated human acuity. Black borders are excluded from scoring.

**Prediction.** Frozen [DINOv2 ViT-S/14](https://github.com/facebookresearch/dinov2) encodes partial RGB. A head predicts the full-image 20 x 32 x 384 feature map: 1 x 1 projection to 128 channels, two 3 x 3 convolutions (dilation 1 and 2), and 1 x 1 output, with GELU. Targets use training-only normalization. Mean training uses MSE, AdamW, learning-rate decay, and early stopping, capped at 80 epochs. The mean is then frozen while an independent variance head learns Gaussian negative log-likelihood.

**Values.** U averages predicted variance around a candidate. Gain labels measure the decrease in squared feature error after revealing it, locally or across the image. U and local gain share fixed Gaussian spatial weights. A 2309-256-64-2 MLP predicts both gains from the current state and candidate geometry. Negative gains are retained. Observer changes require fresh labels and G training; online choices never access hidden targets.

**Experiments.** Mean/variance training covers 1-16 observations; gain labels use before-histories of 1-15. Sixteen is the ceiling of the training mean trajectory length (15.46). Closed-loop policies maximize U or G on a grid with 16 cells along its longer side; center and uniform sampling are baselines. Human replay retains actual trajectories and revisits. Policies share the human start and budget, min(trajectory length, 16). Average seeds within trial, trials within image, then images equally.

Human-choice conditional logit adds each signal to position, movement, resolution-change, and history controls. The first fixation is not scored. [DeepGaze III](https://github.com/matthias-k/DeepGaze) provides a full-image reference on matching targets. Reconstruction and human-choice prediction are separate outcomes.

## Results

The revised continuous-foveation experiment finished on October 7, 2026. Global G gives the lowest scene-prediction error; local G best predicts human choice among the three value signals. DeepGaze III remains the stronger behavioral reference. These are development results, not an independent test.

**Observer and gain prediction.**

| Development measurement | Result |
| --- | ---: |
| Mean predictor feature MSE | 0.5618 |
| Partial-view DINO features without prediction: MSE | 0.8476 |
| Training-mean feature baseline: MSE | 0.9976 |
| Predicted variance / observed squared error | 0.9938 |
| Normalized variance calibration error | 0.0076 |
| Within-state variance/error rank correlation | 0.7642 |

Mean results cover 371 images and 11,271 fixed human/random states, averaging budgets within history type and images equally. The mean reached its 80-epoch cap with epoch 80 still best; convergence is not established. Variance selected epoch 39. Calibration shares checkpoint-selection data.

Fresh gain supervision contains 775,568 training and 171,056 development candidates across all 3,343 training and 371 development images. G stopped at epoch 11 and selected epoch 1.

| Gain prediction | Rank correlation with realized gain | Mean realized gain | Negative realized gains |
| --- | ---: | ---: | ---: |
| Local G | 0.6977 | 0.0639 | 4.46% |
| Global G | 0.5914 | 0.0118 | 9.95% |

**Scene prediction.** MSE is lower when reconstruction is better.

| Trajectory | MSE at 4 | MSE at 8 | MSE at 16 | Matched-endpoint MSE |
| --- | ---: | ---: | ---: | ---: |
| Human replay | 0.6015 | 0.5678 | 0.5286 | 0.5343 |
| U | 0.5789 | 0.5259 | 0.4813 | 0.4886 |
| Local G | 0.5660 | 0.5113 | 0.4567 | 0.4658 |
| Global G | 0.5613 | 0.5031 | 0.4511 | 0.4597 |
| Center prior | 0.5910 | 0.5481 | 0.5041 | 0.5117 |
| Uniform | 0.5906 | 0.5401 | 0.4865 | 0.4958 |

The 4/8/16 columns use the same 1,915 trials reaching 16 observations on 371 images. Matched endpoints include all 3,698 trials at T = min(actual length, 16). Starts and budgets match within trial; human revisits are retained without padding. Baseline seeds are averaged within trial, trials within image, then images equally. No model path exhausted its candidates.

At observation 16, local G minus U is -0.0247 MSE (paired-image 95% interval: -0.0271 to -0.0223); global G minus U is -0.0302 (-0.0329 to -0.0277). Global G minus local G is -0.0055 (-0.0061 to -0.0050).

| Trajectory at observation 16 | Mean sharpness q | Cumulative movement |
| --- | ---: | ---: |
| Human replay | 0.377 | 3.56 |
| U | 0.525 | 9.90 |
| Local G | 0.622 | 11.96 |
| Global G | 0.650 | 10.86 |
| Center prior | 0.476 | 6.49 |
| Uniform | 0.536 | 9.34 |

Movement is measured in content-short-side units; mean q is an engineering sharpness measure, not clear-pixel coverage. Neither movement nor acquired detail is matched. Humans saw normal images, so replay error measures this artificial observer, not human perceptual accuracy.

**Human choice.** Higher target log probability is better.

| Model | Mean target log probability |
| --- | ---: |
| Controls | -4.2762 |
| Controls + U | -4.2622 |
| Controls + local G | -4.2409 |
| Controls + global G | -4.2523 |
| DeepGaze III | -3.9723 |
| DeepGaze center prior | -4.7276 |

All models score the same 49,690 targets from 3,692 trials on 371 development images, after 1-15 observations. Real revisits remain; 40 border targets are excluded while their histories are retained. Scores average targets within trial, trials within image, then images equally. All four conditional-logit fits met their stopping criterion, and all three signal coefficients are positive.

Local G exceeds U by 0.0213 nats (95% interval: 0.0166-0.0261); global G exceeds U by 0.0100 (0.0046-0.0152). DeepGaze exceeds local G by 0.2687 (0.2503-0.2886). Current intervals use 2,000 paired-image bootstrap samples and are descriptive, without multiplicity correction. Shared controls do not establish a unique human mechanism; development also selected upstream checkpoints.

The saved-record audit passed for trajectories, starts, budgets, all policy curves/endpoints, matched reference targets, and human-choice aggregation. Complete curves, budget-specific scores, comparisons, and source hashes are retained in [results/](results/).

**Earlier hard-window results.** These use a different observer and an eight-observation budget.

| Trajectory | Feature MSE at observation 8 |
| --- | ---: |
| Human | 0.7185 |
| U | 0.7184 |
| Local G | 0.7114 |
| Global G | 0.7133 |
| Center prior | 0.7169 |
| Uniform | 0.7210 |

This comparison uses 364 matched images, one human trajectory per image, and shared starts. Reconstruction differences were small.

| Earlier human-choice model | Mean target log probability |
| --- | ---: |
| Controls | -4.2018 |
| Controls + U | -4.1952 |
| Controls + local G | -4.1388 |
| Controls + global G | -4.1736 |
| DeepGaze III | -3.8543 |

Earlier choice scores cover 17,837 targets on 603 internal holdout images after 1, 4, or 7 observations. Local G exceeded U by 0.0564 nats (paired-image 97.5% interval: 0.0468-0.0660; adjusted for two primary comparisons). This holdout has already been inspected; the old and current scores use different populations and budgets and are not directly comparable.

Future work may extend the mean predictor beyond 80 epochs to a development plateau; changing it would require refitting variance and regenerating and retraining G.

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
