# DeepGaze III reference

`run.py` provides the original small integration run. `development.py` scores complete real development trajectories with the official pretrained DeepGaze III model. The corresponding shell files are original cluster launchers.

The initial fixation is not scored as a free choice. True short histories are preserved, rather than filled with invented locations. Saved scores include full history, target coordinates, content-grid probability, and a center-bias reference.

Current human-choice analysis reuses these scores only after exact target/history/grid matching. It does not compare an unrestricted full-trajectory average with a shorter supported-budget average. DeepGaze has access to the full image and is a behavioral reference, not the partial observer's uncertainty model.

Model source and weights are external: https://github.com/matthias-k/DeepGaze

Scripts contain server-specific environment and data paths. No third-party model files or raw gaze records are included here.
