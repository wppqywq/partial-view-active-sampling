"""Before-only U/G inputs and offline gain labels for continuous foveation.

Local U and local gain use the SAME fixed, history-independent spatial weight.
Resolution improvement is a separate feature, never a multiplier on U.
Tensor operations must only be called inside Slurm jobs.
"""
import os

import numpy as np
import torch


def _require_job():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Numerical evaluation must run inside Slurm')


def spatial_average(value, weight):
    _require_job()
    return (value * weight).sum((-2, -1)) / weight.sum((-2, -1)).clamp_min(1e-12)


def local_uncertainty(log_variance, fixed_local_weights):
    """Candidate channel-mean predictive variance, with no after-view input."""
    return spatial_average(log_variance.exp(), fixed_local_weights).mean(1)


def realized_labels(before_mean, after_mean, target, content_weight, fixed_local_weights):
    """Offline supervision only. Columns: global gain, local gain."""
    _require_job()
    change = (before_mean - target).square() - (after_mean - target).square()
    global_gain = spatial_average(change, content_weight).mean(1)
    local_gain = spatial_average(change, fixed_local_weights).mean(1)
    result = torch.stack((global_gain, local_gain), dim=1)
    if not torch.isfinite(result).all():
        raise ValueError('Non-finite gain labels')
    return result


def online_vectors(partial, mean, log_variance, content_weight,
                   fixed_local_weights, coordinate_xy, content_box,
                   current_resolution, potential_resolution_increase,
                   budget, maximum_budget):
    """Build candidate features using BEFORE state and geometry only.

current_resolution: 1x1xHxW, pooled continuously (not a thresholded mask).
potential_resolution_increase: Nx1xHxW, pooled q_after-q_before from
geometry only; it must be computed without revealing RGB or encoding targets.
"""
    _require_job()
    if not 1 <= budget < maximum_budget:
        raise ValueError('G before-state budget must be in 1..maximum_budget-1')
    count = len(fixed_local_weights)
    if potential_resolution_increase.shape != fixed_local_weights.shape:
        raise ValueError('Candidate resolution increases and local weights must align')
    pooled = []
    for value in (partial, mean, log_variance):
        pooled.extend((spatial_average(value, content_weight).expand(count, -1),
                       spatial_average(value, fixed_local_weights)))
    left, top, right, bottom = content_box
    coordinates = torch.as_tensor(np.asarray(coordinate_xy), dtype=mean.dtype, device=mean.device)
    origin = coordinates.new_tensor([left, top])
    extent = coordinates.new_tensor([right-left, bottom-top])
    coordinates = 2 * (coordinates-origin) / extent - 1
    # q maps are zero outside content before pooling. Divide out boundary
    # occupancy for local conditional q; do not multiply global increments
    # by fractional occupancy a second time.
    conditional_resolution = current_resolution / content_weight.clamp_min(1e-12)
    local_resolution = spatial_average(conditional_resolution, fixed_local_weights)
    new_resolution = (potential_resolution_increase.sum((-2, -1)) /
                      content_weight.sum((-2, -1)).clamp_min(1e-12))
    scalars = torch.cat((coordinates, local_resolution, new_resolution,
                         mean.new_full((count, 1), budget / maximum_budget)), dim=1)
    result = torch.cat(pooled + [scalars], dim=1)
    if result.shape != (count, 2309) or not torch.isfinite(result).all():
        raise ValueError('Invalid before-only gain input')
    return result, local_uncertainty(log_variance, fixed_local_weights)
