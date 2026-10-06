# Data preparation

`audit.py` downloads and audits the official COCO-FreeView inputs, checks display coordinates, creates an image-disjoint split, and renders coordinate overlays. `run.sh` is the original site-specific Slurm entry point. It writes data and generated outputs to external experiment storage.

The audited release contains 4,317 images, 10 participants, 43,048 trials, and 665,005 fixations. Split sizes are 3,343 train, 371 development, and 603 internal holdout images. The initial fixation is experimental initialization, not a freely chosen prediction target. Border fixations are retained in histories and flagged for target eligibility.

Obtain data through the official source: https://sites.google.com/view/cocosearch/coco-freeview

See [complete acquisition instructions](../docs/dataset.md) for the five files, checksums, and split algorithm. For a new data destination, use `scripts/prepare_dataset.sh` with `DATA_ROOT`, `PROJECT_CODE`, and `PROJECT_PYTHON`; the original `run.sh` remains site-specific.

Raw data and overlays containing dataset images are excluded from the public source tree. Training uses the original saved split manifest; rerunning preparation is not necessary for the current experiment. Inspect paths and output behavior before running the script on another system.
