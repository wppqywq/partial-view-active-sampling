# Dataset acquisition and split reconstruction

Use the public COCO-FreeView training/validation release, not the COCO-Search18 search-task fixation files. FreeView reuses the Search18 image stimuli. Sources were checked on October 6, 2026:

- [COCO-FreeView official page](https://sites.google.com/view/cocosearch/coco-freeview)
- [COCO-Search18 image downloads and data terms](https://sites.google.com/view/cocosearch/home)

The full dataset description mentions 822,602 fixations. This project audits only the downloadable training/validation release: 665,005 fixations, 43,048 trials, 4,317 images, and 10 participants. The official test fixations are withheld. Our internal holdout comes from the official validation images; it is not the withheld benchmark test set.

## Required files

| Local name under DATA_ROOT/raw | Source | Purpose |
| --- | --- | --- |
| `fixations.json` | [FreeView train/validation JSON](https://vision.cs.stonybrook.edu/~cvlab_download/COCOFreeView_fixations_trainval.json) | Actual free-viewing histories |
| `COCOSearch18-images-TP.zip` | [Target-present images](https://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-images-TP.zip) | Display image stimuli |
| `COCOSearch18-images-TA.zip` | [Target-absent images](https://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-images-TA.zip) | Display image stimuli |
| `annotations_trainval2014.zip` | [COCO 2014 annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) | Original image dimensions, from caption metadata |
| `readme.txt` | [Official FreeView format description](https://drive.google.com/file/d/1Hj_jyK8Ml27Ge_5sogEtyI7XOjyad4aj/view) | Coordinate and initial-fixation convention |

The downloaded archives total about 2.5 GB; extracted images and outputs need additional space. No COCO original-image archive is required: use the supplied 1680 x 1050 display JPEGs. The annotations are used only to recover original dimensions and the letterboxed content box.

`data_a/RAW_SHA256SUMS` contains hashes from the completed original audit. Verify bytes before accepting a dataset revision. Google Drive may return a sign-in or confirmation page rather than text; manually download the actual format description if necessary. A fetch of the large fixation URL timed out during this documentation audit, so endpoint reachability is not claimed. The new helper has been statically checked; the 2.5 GB download was not repeated.

The official source page restricts redistribution of the dataset. Download it from the authors; do not push the archives, images, fixation records, or per-participant records into this repository.

## Prepare a new dataset copy

With Python 3.12, NumPy, Pillow, Matplotlib, curl, ripgrep, and sha256sum available, submit from the repository root on the research server:

```bash
export PROJECT_CODE="$PWD"
export DATA_ROOT=/path/to/new/external/coco_freeview
export PROJECT_PYTHON=/path/to/python
sbatch scripts/prepare_dataset.sh
```

Use a new destination. The helper refuses to overwrite an existing manifest or put raw data inside the source repository. It downloads missing files, verifies hashes, and executes the original audit in Slurm. On another cluster, adapt the partition and resource directives. `data_a/run.sh` remains the original site-bound launcher; use the new helper for an explicit destination.

Outputs are `manifest.json`, `audit.json`, extracted selected display images, and coordinate overlays in a job-specific audit folder. Overlays contain dataset images and remain outside the public repository.

## Exact image split rule

The seed is 20260928. Take the unique image names labeled `train` in the official fixation JSON. Sort them by the hexadecimal SHA-256 of:

```text
20260928:development:IMAGE_FILENAME
```

The first `round(0.1 * number_of_official_training_images)` names form development; the remaining official training images form training. All official `valid` images form the internal holdout. Expected sizes are 3,343 / 371 / 603. All participants for an image share its split. Smoke and overlay selections use the same deterministic ranking with separate purpose strings.

The manifest contains absolute data paths, so a fresh installation will have a different manifest byte hash even when image membership and source data are identical. Existing checkpoint contracts must not have their hard-coded manifest hashes bypassed. A portable training configuration and fresh provenance generation are still required for a full rerun on another server.
