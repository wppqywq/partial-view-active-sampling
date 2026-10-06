# Multiscale foveation provenance

On 2026-10-05 we inspected the public implementation
[Image_Foveation_Python](https://github.com/ouyangzhibo/Image_Foveation_Python)
and its [retina_transform.py](https://raw.githubusercontent.com/ouyangzhibo/Image_Foveation_Python/master/retina_transform.py).
Its repository advertises an MIT license and cites Perry & Geisler (2002) and
Jiang et al. (2015). Its source builds a Gaussian pyramid, computes resolution
from distance to the nearest historical fixation, and blends pyramid levels.

The exact upstream LICENSE could not be fetched in this environment. Therefore
no source or license text was copied or vendored. `../observer.py` is a new
implementation of the multiscale, cumulative-resolution design. This is
conceptual reuse, not an exact reproduction of the upstream retina model.
No upstream defaults, optical calibration, or psychophysical validation are
claimed. Dependencies are the existing NumPy, Pillow, and OpenCV environment.

## Revised engineering specification (r2, 2026-10-05)

The r1 design (D50=0.60 plus a downsampling pyramid) was rejected after the
user inspected its nearly clear cumulative eight-fixation view. Its original
source and rendered outputs remain immutable under preparation job22738.
The r2 change below is a single explicit engineering revision, not a search
for parameters that make a desired policy win. It is not a new acuity fit.

- Preserve the previous bilinear full-image target transform exactly.
- Measure scale against the full resized display's short side, not content
  bounding box. Thus image black borders do not redefine foveation size.
- Full-resolution core diameter: 0.16 display short side.
- Nominal half-resolution diameter D50: 0.28 display short side.
- At display size 448 x 280 these are 44.8 and 78.4 pixels, respectively.
- Thus D50 spans1.75 core diameters, compared with3.75 in rejected r1.
- Let r0 and r50 be half these diameters, and d the distance to a fixation.
  q(d)=max(1/16, 1/[1+(max(d-r0,0)/(r50-r0))^2]).
- Cumulative memory takes the pointwise maximum q over all past fixations;
  an empty history has q=1/16. No forgetting or duration is modeled.
- Gaussian width sigma(q)=0.8*sqrt(q^-2-1) in output pixels. Each Gaussian
  layer is filtered from the resized RGB source, without downsampling; adjacent
  layers blend in sigma-squared. Fixed sigma levels are0,.25,.5,1,2,4,8,16.
  There is no untouched source contribution at sigma>=.25 pixels: only the
  core and an approximately3.7pixel transition outside it can mix level0.
  Gaussian filters attenuate rather than strictly eliminate high frequencies;
  mixing adjacent Gaussian responses is an approximation to a variable blur.
- q and D50 are **engineering sharpness coordinates**, not transparency,
  recognition accuracy or measured optical spatial-frequency cutoff. The
  sigma relation and displayed pixels make the actual degradation explicit.
- Broadening/narrowing q is independent of perfect-memory accumulation.
  Previews show current-gaze-only input separately from cumulative memory.
- Local U and local realized Delta both use the same content-masked Gaussian
  footprint with sigma=r50, normalized when aggregating feature errors. The
  footprint is fixed around the candidate and independent of current history.
- Mean increase in q is a separate candidate covariate; policy eligibility
  requires an increase greater than 1e-8, solely a numerical no-change tolerance.
- Known display borders remain sharp, contribute no observation evidence, and
  are excluded from local kernels and scoring. Encoder padding is black/invalid.

The user-approved aim is to retain central detail **and useful medium-scale
context**, rather than sharply separate a small clear disk from uniform blur.
The human display geometry for this FreeView collection remains unverified;
these parameters must not be represented as a calibrated human acuity model.
