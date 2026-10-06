# Legacy hard-window observer

`observer.py` implements the original uniformly blurred image with cumulative clear circular windows. It is retained for preprocessing equality checks and historical comparisons. It is not the current observation model; use `foveated_v3/observer.py` for continuous foveation.

Three docstrings have been converted to ASCII. This changes the source digest but not the computations. Existing frozen experiments continue using their original source snapshots outside this workspace.
