# Storage dependencies before deleting the old video project

Checked on October 6, 2026. Two distinct directories have similar names.

| Directory | Relationship to active sensing |
| --- | --- |
| `/home/youyouyang/deepgaze_video` | Older video-project source checkout. No direct reference was found in the current active-sensing source or frozen job source. No links from that checkout or environment editable imports into it were found. Its Git worktree was clean. |
| `/mnt/disk2/youyouyang/deepgaze_video` | Shared runtime storage. Required by the current project and active job; do not delete or move it during execution. |

Job 22779 runs the interpreter at `envs/official/bin/python` in the scratch directory. This is a symlink to `python/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12` in the same scratch tree. Both `envs/official` and `python` are required, and mapped libraries also come from that environment. Copying only the virtual environment directory does not make it independent.

The current project's datasets, DINO source/weights, DeepGaze III reference source/cache, learned checkpoints, and result caches are stored under `/mnt/disk2/youyouyang/proposal2`, not in the old home checkout. The dependency found in the video scratch tree is the reused Python runtime and installed packages; this does not mean every other directory in that tree has been audited as disposable.

Deleting only the home checkout has no identified impact on the current active-sensing job. That finding is scoped to this project, not permission to discard the older video's own source, reports, or other users' references. Neither directory was deleted or changed during this audit.

To remove the scratch dependency later: wait until all dependent jobs finish, create a new environment with its own base interpreter under this project's storage, install and verify the required packages, update future launchers, and test model loading in Slurm. Keep the old environment until replacement checks pass. Do not rename an active environment or change frozen job sources in place.
