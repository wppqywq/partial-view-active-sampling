# Gaussian examples

`gaussian.py` contains the original executed notebook's numerical cells as an English Python script. It computes analytic uncertainty and local/global squared-error gains, then checks them against 200,000 Monte Carlo samples per case using seed 20260928. Output is `comparison.png` in the current working directory.

Run in a disposable output directory. On the research server, use a Slurm CPU allocation; do not run numerical work on the login node. The existing result is in `results/figures/gaussian_examples.png`.

The three cases isolate independent exact observations, observation noise, and correlations between regions. They are mathematical examples, not evidence about human behavior. Global gain here uses a sum of squared errors; converting to MSE divides by the fixed number of dimensions and preserves rankings within each case.
