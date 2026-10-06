"""Continuous multiscale observation. Numerical execution belongs in Slurm.

Engineering display-scale model, NOT calibrated human acuity. ``resolution``
is a nominal pyramid sampling ratio, not opacity or measured optical cutoff.
Design inspiration/provenance: vendor/PROVENANCE.md. No upstream code copied.
"""

import cv2
import numpy as np
from PIL import Image

# Each Slurm worker is assigned a bounded CPU share.
cv2.setNumThreads(1)


DEFAULT_CONFIG = {
    "long_side": 448,
    "patch_multiple": 14,
    "core_diameter_fraction": .16,
    "half_resolution_diameter_fraction": .28,
    "resolution_floor": 1 / 16,
    "blur_scale_pixels": .8,
}


class FoveatedObserver:
    """Render cumulative best resolution from raw display-image (x,y) points.

    Distances use resized display pixels, never content-box dimensions. Pixel
    centres are (column+.5,row+.5); content-box bounds are integer pixel edges.
    A repeated point leaves the input exactly unchanged. No duration/forgetting
    model is included. ``target`` is reserved for encoder/scoring supervision.
    """

    def __init__(self, image, long_side=448, patch_multiple=14,
                 core_diameter_fraction=.16,
                 half_resolution_diameter_fraction=.28,
                 resolution_floor=1 / 16, content_box=None,
                 blur_scale_pixels=.8):
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL image")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 1
               for v in (long_side, patch_multiple)):
            raise ValueError("long_side and patch_multiple must be positive integers")
        params = np.asarray([core_diameter_fraction,
                             half_resolution_diameter_fraction,
                             resolution_floor, blur_scale_pixels], dtype=float)
        if (not np.isfinite(params).all()
                or not 0 < core_diameter_fraction < half_resolution_diameter_fraction
                or not 0 < resolution_floor < .5 or blur_scale_pixels <= 0):
            raise ValueError("require finite 0 < core diameter < D50 and 0 < floor < .5")
        image = image.convert("RGB")
        self.original_size = image.size
        box = np.asarray(content_box if content_box is not None
                         else (0, 0, *image.size), dtype=float)
        if (box.shape != (4,) or not np.isfinite(box).all()
                or not np.equal(box, np.floor(box)).all()):
            raise ValueError("content_box must contain four finite integer pixel edges")
        left, top, right, bottom = self.content_box = tuple(int(v) for v in box)
        if not (0 <= left < right <= image.width
                and 0 <= top < bottom <= image.height):
            raise ValueError("content_box must be nonempty and inside the display image")
        ratio = long_side / max(image.size)
        w, h = [max(1, round(v * ratio)) for v in image.size]
        self.size = (w, h)
        self.scale_xy = np.array([w / image.width, h / image.height])
        self.core_radius = core_diameter_fraction * min(w, h) / 2
        self.half_resolution_radius = half_resolution_diameter_fraction * min(w, h) / 2
        self.radius = self.core_radius  # geometry-only compatibility; not local weights
        self.resolution_floor = float(resolution_floor)
        self.blur_scale_pixels = float(blur_scale_pixels)
        self.config = dict(long_side=long_side, patch_multiple=patch_multiple,
                           core_diameter_fraction=float(core_diameter_fraction),
                           half_resolution_diameter_fraction=float(half_resolution_diameter_fraction),
                           resolution_floor=self.resolution_floor,
                           blur_scale_pixels=self.blur_scale_pixels)
        # Exactly the old observer's target transform, including display borders.
        image = image.resize((w, h), Image.Resampling.BILINEAR)
        self._sharp = np.asarray(image, dtype=np.float32) / 255
        self._yy, self._xx = np.mgrid[:h, :w] + .5
        self._content_mask = ((self._xx >= left * self.scale_xy[0])
                              & (self._xx < right * self.scale_xy[0])
                              & (self._yy >= top * self.scale_xy[1])
                              & (self._yy < bottom * self.scale_xy[1]))
        self.shape = tuple(((v + patch_multiple - 1) // patch_multiple)
                           * patch_multiple for v in (h, w))
        # Revision 2: calibrated-in-pixels Gaussian scale space. The state q is
        # an engineering sharpness coordinate, NOT a measured optical cutoff.
        # sigma(q)=s*sqrt(q^-2-1). Adjacent Gaussian layers blend in variance.
        # The sharp source contributes only where sigma < .25 output pixels;
        # with fixed defaults this is a ~3.7px annulus outside the 22.4px core.
        maximum_sigma = self.blur_scale_pixels * np.sqrt(self.resolution_floor**-2 - 1)
        self._sigmas = np.r_[0., 2. ** np.arange(-2, max(-2, int(np.ceil(np.log2(maximum_sigma)))) + 1)]
        self._pyramid = [self._sharp]
        for sigma in self._sigmas[1:]:
            self._pyramid.append(cv2.GaussianBlur(self._sharp, (0, 0), float(sigma),
                                                borderType=cv2.BORDER_REFLECT_101))

    def _history(self, history_xy):
        points = np.asarray(history_xy, dtype=float)
        if points.shape == (0,):
            points = points.reshape(0, 2)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("history must have shape (N,2), ordered as x,y")
        if (not np.isfinite(points).all() or (points < 0).any()
                or (points >= self.original_size).any()):
            raise ValueError("history must contain finite points inside display image")
        return points * self.scale_xy

    def _pad(self, array):
        result = np.zeros(self.shape + array.shape[2:], dtype=array.dtype)
        w, h = self.size
        result[:h, :w] = array
        return result

    def _resolution_at(self, point):
        """q=1 in core, q=.5 at r50, floor in far periphery."""
        x, y = point
        distance = np.hypot(self._xx - x, self._yy - y)
        z = np.maximum(distance - self.core_radius, 0)
        z /= self.half_resolution_radius - self.core_radius
        return np.maximum(self.resolution_floor, 1 / (1 + z * z)).astype(np.float32)

    def observe(self, history_xy=()):
        """Fresh float32 RGB, resolution map, and boolean masks, encoder-padded.

        Resolution_map is zero outside content; known display borders stay sharp
        but never count as observation evidence. Core_mask denotes q=1, not all
        useful information. reveal_mask is a compatibility alias for core_mask;
        new state encoding and eligibility MUST use resolution_map instead.
        Adjacent low-pass Gaussian layers blend in variance. Untouched source
        mixing is restricted to sigma<.25 pixels, never the broad periphery.
        """
        w, h = self.size
        resolution = np.full((h, w), self.resolution_floor, dtype=np.float32)
        for point in self._history(history_xy):
            np.maximum(resolution, self._resolution_at(point), out=resolution)
        variance = self.blur_scale_pixels**2 * np.maximum(resolution**-2 - 1, 0)
        levels = self._sigmas**2
        lower = np.clip(np.searchsorted(levels, variance, side='right') - 1, 0, len(levels)-2)
        fraction = np.clip((variance-levels[lower]) / (levels[lower+1]-levels[lower]), 0, 1)
        rgb = np.zeros_like(self._sharp)
        for i, layer in enumerate(self._pyramid):
            weight = np.where(lower == i, 1 - fraction, 0)
            weight += np.where(lower + 1 == i, fraction, 0)
            rgb += layer * weight[..., None]
        rgb[~self._content_mask] = self._sharp[~self._content_mask]
        core = (resolution == 1) & self._content_mask
        core_padded = self._pad(core)
        return {"rgb": self._pad(np.clip(rgb, 0, 1)),
                "resolution_map": self._pad(resolution * self._content_mask),
                "sharp_mix_weight": self._pad((np.where(lower == 0, 1-fraction, 0)
                                                * self._content_mask).astype(np.float32)),
                "core_mask": core_padded,
                "reveal_mask": core_padded.copy(),
                "valid_mask": self._pad(self._content_mask),
                "display_mask": self._pad(np.ones((h, w), dtype=bool))}

    def choice_grid(self, grid_long_side=16):
        """Stable row-major content cells in ORIGINAL display coordinates."""
        if not isinstance(grid_long_side, int) or isinstance(grid_long_side, bool) or grid_long_side < 1:
            raise ValueError("grid_long_side must be a positive integer")
        left, top, right, bottom = self.content_box
        width, height = right - left, bottom - top
        nx, ny = [min(v, max(1, round(grid_long_side * v / max(width, height))))
                  for v in (width, height)]
        x_edges = np.rint(np.linspace(left, right, nx + 1)).astype(np.int64)
        y_edges = np.rint(np.linspace(top, bottom, ny + 1)).astype(np.int64)
        x, y = np.meshgrid((x_edges[:-1] + x_edges[1:]) / 2,
                           (y_edges[:-1] + y_edges[1:]) / 2)
        return {"xy": np.column_stack((x.ravel(), y.ravel())),
                "x_edges": x_edges, "y_edges": y_edges, "shape": (ny, nx)}

    def local_weights(self, xy):
        """Fixed Gaussian candidate footprint, independent of observation history.

        Sigma is r50. Returns unnormalized float32 content-only weights; aggregate
        U and local error using the SAME normalized weights. This kernel defines
        the local objective, not foveation opacity or newly available information.
        """
        points = self._history([xy])
        x, y = points[0]
        squared = (self._xx - x) ** 2 + (self._yy - y) ** 2
        weights = np.exp(-.5 * squared / self.half_resolution_radius ** 2)
        return self._pad((weights * self._content_mask).astype(np.float32))

    def _current_resolution(self, resolution_map):
        w, h = self.size
        if resolution_map is None:
            return np.full((h, w), self.resolution_floor, dtype=np.float32)
        q = np.asarray(resolution_map)
        if (q.shape != self.shape or not np.issubdtype(q.dtype, np.floating)
                or not np.isfinite(q).all() or (q < 0).any() or (q > 1).any()):
            raise ValueError("use finite padded floating resolution_map from observe()")
        current = q[:h, :w]
        if (current[self._content_mask] < self.resolution_floor - 1e-7).any():
            raise ValueError("content resolution is below this observer's floor")
        return current

    def candidate_improvements(self, resolution_map=None, grid_long_side=16):
        """Mean content resolution increase for each fixed-grid candidate.

        This is a separate information-availability covariate, never U itself.
        All history, including true revisits, remains valid for observe().
        """
        xy = self.choice_grid(grid_long_side)["xy"]
        current = self._current_resolution(resolution_map)
        improvement = np.asarray([
            np.maximum(self._resolution_at(p) - current, 0)[self._content_mask].mean()
            for p in xy * self.scale_xy], dtype=np.float32)
        return xy, improvement

    def resolution_increase(self, xy, resolution_map=None):
        """Padded content-only q_after-q_before for one candidate, without RGB.

        Pool this map to encoder cells for the G input's geometry covariate.
        Uses no target, after-view encoding, or observation of candidate content.
        """
        point = self._history([xy])[0]
        current = self._current_resolution(resolution_map)
        increase = np.maximum(self._resolution_at(point) - current, 0)
        return self._pad(increase * self._content_mask)

    def candidates(self, resolution_map=None, grid_long_side=16):
        """Return grid and candidates with positive mean resolution increase.

        The 1e-8 content-mean threshold only removes floating-point/no-change
        candidates; it is not a physiological inhibition-of-return assumption.
        """
        xy, improvement = self.candidate_improvements(resolution_map, grid_long_side)
        return xy, improvement > 1e-8

    def target(self):
        """Full RGB target; never supply to the policy or online observer head."""
        return self._pad(self._sharp)


PartialObserver = FoveatedObserver
