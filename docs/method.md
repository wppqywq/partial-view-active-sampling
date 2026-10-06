# Implementation

## Observation model

The current version is continuous foveation r2. It retains the previous full-image target preprocessing, including bilinear resizing and black padding to multiples of 14. At the standard 448 x 280 input, the clear-core diameter is 44.8 pixels and the half-sharpness diameter is 78.4 pixels. Their scale is tied to the display, not the content box, so letterboxing does not redefine the observation size.

For distance d from a fixation, core radius r0, and half-sharpness radius r50:

```text
q(d) = max(1/16, 1 / (1 + (max(d - r0, 0) / (r50 - r0))^2))
sigma(q) = 0.8 * sqrt(q^(-2) - 1) pixels
```

Take the pointwise maximum q across actual historical fixations. The empty history uses q = 1/16. Gaussian layers at sigma = 0, 0.25, 0.5, 1, 2, 4, 8, 16 are computed from the resized source. Interpolation is in Gaussian variance. This is an approximation to spatially varying blur, not a physiological calibration or a strict frequency cutoff. No downsampling pyramid is used in r2.

The interface returns RGB, a continuous resolution map, a clear-core mask, and valid content/display masks. Known black borders remain sharp but contribute no evidence or loss. A repeated identical observation changes neither RGB nor the resolution map.

## Models

DINOv2 ViT-S/14 is frozen. Its full-image feature target has shape 384 x 20 x 32. Per-channel normalization is estimated using training images only and reused across observers.

The mean head uses a 1 x 1 projection to 128 channels, a 3 x 3 convolution, a dilation-2 3 x 3 convolution, and a 1 x 1 output projection, with GELU activations. Inputs include partial DINO features, pooled resolution, valid-content weights, and coordinates. It predicts the complete normalized feature map directly; it is not an RGB reconstruction network.

Training samples human or random histories and a supported prefix between 1 and 16. Development uses fixed human/random histories. MSE training is capped at 80 epochs with AdamW, learning-rate decay, and early stopping. Best checkpoint selection uses development MSE. A frozen identity records source, protocol, normalization, and checkpoint hashes.

The independent variance head takes the frozen partial features and mean, resolution, content weights, and coordinates. It is trained with Gaussian negative log-likelihood, with log variance constrained to [-6, 4]. No optimizer changes the mean or DINO parameters. Candidate calibration compares local predicted variance with local squared error using identical weights; the review is an engineering judgment on development data.

## Values and gains

For each candidate a, define fixed content-masked Gaussian weights w_a with sigma = r50. They do not depend on the history.

```text
U(a) = weighted mean of predicted variance under w_a
Delta_local(a) = weighted mean of error_before - error_after under w_a
Delta_global(a) = content-weighted mean of error_before - error_after
G(a) = learned prediction of Delta from the before state and candidate geometry
```

Negative realized gains remain negative. G uses 2,309 inputs: global/local pooling of partial features, predicted mean, and log variance, plus candidate coordinates, local current resolution, global potential resolution increase, and budget fraction. The MLP has widths 2309, 256, 64, 2 with GELU. Outputs are global and local gain.

Fresh labels cover all training/development images, human/random before-histories from 1 to 15, and 8 training or 16 development candidates per state. This is sampled candidate supervision, not enumeration of all candidates at every state. Labels are tied to the exact frozen observer. Old labels cannot be reused after changing it.

## Evaluations

Closed-loop policies choose among a fixed content grid whose longer side has 16 cells. A candidate must increase mean content resolution by more than 1e-8. U, local G, and global G select their respective largest scores; uniform and training-derived center priors are baselines. Baselines currently use five seeds. Fixed seeds resolve exact ties. Candidate exhaustion is recorded without inventing new fixations.

Final scene replay uses every available real development trial, the actual first fixation, and T = min(N, 16). Average baseline seeds within trial, then trials within image, then images equally. Report all budgets, the matched endpoint, and a separate fixed complete-16 cohort. Never pool participants' fixation coordinates into a synthetic path.

Human-choice evaluation uses actual before-histories from 1 to 15 and retains all content candidates, including revisits. Exact zero-resolution-change actions have G = 0. Border targets are counted and excluded from the content-conditional likelihood, while border history remains. The initial fixation is never a target. Coordinates follow the archived reference's float32 convention.

Conditional logit compares Controls, Controls + U, Controls + local G, and Controls + global G. Controls include position, movement distance, available resolution change, local resolution, visit count, and recency. Standardization and coefficient fitting use training images only. Scores use image/trial/state macro averaging and image bootstrap intervals. DeepGaze scores are joined by exact image, participant, target index, coordinates, history, and grid cell. Missing reference targets cause an explicit alignment failure, not silent sample removal.

The current gain and evaluation stages are not complete. The definitions above describe the implementation, not established empirical results.
