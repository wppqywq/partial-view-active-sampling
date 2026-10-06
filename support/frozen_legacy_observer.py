"""Pixel-space partial views; run inside a Slurm job. Requires numpy, Pillow."""

import numpy as np
from PIL import Image, ImageFilter


class PartialObserver:
    """Environment only: the policy receives observe() output, never target().

    Coordinates are (x, y), in the supplied display-image pixels, measured
    from the top-left edge. Pixel centres are (column + .5, row + .5).
    """

    def __init__(self, image, long_side=448, patch_multiple=14,
                 radius_fraction=.08, blur_fraction=.05, content_box=None):
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL image")
        if any(not isinstance(v, int) or v < 1
               for v in (long_side, patch_multiple)):
            raise ValueError("long_side and patch_multiple must be positive integers")
        if not (0 < radius_fraction <= 1 and 0 < blur_fraction <= 1):
            raise ValueError("radius/blur fractions must be in (0, 1]")
        image = image.convert("RGB")
        self.original_size = image.size  # width, height; no EXIF rotation
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
        self.size = (w, h)  # display canvas, before encoder padding
        self.scale_xy = np.array([w / image.width, h / image.height])
        content_short_side = min(np.array([right - left, bottom - top])
                                 * self.scale_xy)
        self.radius = radius_fraction * content_short_side
        image = image.resize((w, h), Image.Resampling.BILINEAR)
        self._sharp = np.asarray(image, dtype=np.float32) / 255
        blurred = image.filter(ImageFilter.GaussianBlur(blur_fraction * content_short_side))
        self._blurred = np.asarray(blurred, dtype=np.float32) / 255
        self._yy, self._xx = np.mgrid[:h, :w] + .5
        self._content_mask = ((self._xx >= left * self.scale_xy[0])
                              & (self._xx < right * self.scale_xy[0])
                              & (self._yy >= top * self.scale_xy[1])
                              & (self._yy < bottom * self.scale_xy[1]))
        self.shape = tuple(((v + patch_multiple - 1) // patch_multiple)
                           * patch_multiple for v in (h, w))

    def _history(self, history_xy):
        points = np.asarray(history_xy, dtype=float)
        if points.shape == (0,):
            points = points.reshape(0, 2)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("history must have shape (N, 2), ordered as x, y")
        if (not np.isfinite(points).all() or (points < 0).any()
                or (points >= self.original_size).any()):
            raise ValueError("history must contain finite points inside the display image")
        return points * self.scale_xy

    def _footprint(self, point):
        x, y = point
        return (((self._xx - x) ** 2 + (self._yy - y) ** 2 <= self.radius ** 2)
                & self._content_mask)

    def _pad(self, array):
        result = np.zeros(self.shape + array.shape[2:], dtype=array.dtype)
        w, h = self.size
        result[:h, :w] = array
        return result

    def observe(self, history_xy=()):
        """Replay the full supplied history; return fresh arrays, no hidden state.

        rgb: float32 HxWx3 in [0,1]; all masks: bool HxW.
        valid_mask is content; display_mask includes existing display borders.
        Display borders stay known/sharp and never count as revealed evidence.
        New encoder padding is black, at bottom/right, and false in every mask.
        """
        w, h = self.size
        mask = np.zeros((h, w), dtype=bool)
        for point in self._history(history_xy):
            mask |= self._footprint(point)
        rgb = np.where((mask | ~self._content_mask)[..., None],
                       self._sharp, self._blurred)
        return {"rgb": self._pad(rgb), "reveal_mask": self._pad(mask),
                "valid_mask": self._pad(self._content_mask),
                "display_mask": self._pad(np.ones((h, w), dtype=bool))}

    def choice_grid(self, grid_long_side=16):
        """Fixed row-major cells covering content, in display coordinates.

        Return xy (Nx2 cell centres), integer x_edges/y_edges (half-open
        cells), and shape (ny, nx). Edges partition the content box exactly.
        Integer rounding can make adjacent cell widths differ by one pixel.
        """
        if not isinstance(grid_long_side, int) or grid_long_side < 1:
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

    def candidates(self, reveal_mask=None, grid_long_side=16):
        """Return (xy, eligible), keeping grid cells stable across observations.

        xy: Nx2 in display coordinates, row-major, all inside content.
        eligible: N bools; false if the patch adds no new content pixels.
        Human replay may still contain repeats; observe() never removes them.
        """
        xy = self.choice_grid(grid_long_side)["xy"]
        w, h = self.size
        points = xy * self.scale_xy
        mask = np.zeros((h, w), dtype=bool)
        if reveal_mask is not None:
            reveal_mask = np.asarray(reveal_mask)
            if reveal_mask.shape != self.shape or reveal_mask.dtype != np.bool_:
                raise ValueError("use the padded boolean reveal_mask from observe()")
            mask = reveal_mask[:h, :w]
        eligible = np.array([np.any(self._footprint(p) & ~mask) for p in points])
        return xy, eligible

    def target(self):
        """Full RGB target for the frozen encoder/scoring branch ONLY."""
        return self._pad(self._sharp)
