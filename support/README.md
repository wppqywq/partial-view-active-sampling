# Reused implementation sources

These are source copies from completed mean run 22742:

- `frozen_mean.py`: direct mean head, frozen DINO encoder, and original training utilities.
- `frozen_data.py`: original data and tensor utilities.
- `frozen_legacy_observer.py`: original hard-window observer used for target compatibility.

They are included so the current pipeline's inherited implementation is inspectable. Current cluster launchers copy the immutable originals from experiment storage. Source hashes are part of checkpoint identities; do not replace files inside a frozen run with edited public copies.

The legacy observer has ASCII-only docstrings in this public copy. That comment-only change alters its file hash. Other helper files retain their archived contents. Full exact-byte snapshots remain in experiment storage and the pre-cleanup archive.
