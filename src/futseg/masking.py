"""Alpha -> (inpaint_mask, composite_alpha), plus dilate/erode/feather helpers.

One segmentation yields *two* masks with opposite offsets. Conflating them is the
single easiest way to produce the classic cheap-cutout halo: the pixels just
outside a silhouette are camera-anti-aliased blends of subject and *original*
background, so unless the inpaint region extends underneath the composited edge,
they survive into the output as a stale rim hugging the subject.
"""

from collections.abc import Sequence
from typing import NamedTuple

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

Alpha = NDArray[np.float32]


class DerivedMasks(NamedTuple):
    """The two masks a single segmentation produces.

    `inpaint_mask` marks what the generative backend regenerates; `composite_alpha`
    is what the original person is pasted back with.
    """

    inpaint_mask: Alpha
    composite_alpha: Alpha


def _as_alpha(alpha: NDArray) -> Alpha:
    """Contiguous float32, which is what the OpenCV morphology calls expect."""
    return np.ascontiguousarray(alpha, dtype=np.float32)


def _disc(radius: int) -> NDArray[np.uint8]:
    """A round structuring element, so corners are not clipped square."""
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def erode(alpha: Alpha, radius: int) -> Alpha:
    """Shrink the shape by `radius` pixels."""
    if radius <= 0:
        return alpha
    return _as_alpha(cv2.erode(_as_alpha(alpha), _disc(radius)))


def dilate(alpha: Alpha, radius: int) -> Alpha:
    """Grow the shape by `radius` pixels."""
    if radius <= 0:
        return alpha
    return _as_alpha(cv2.dilate(_as_alpha(alpha), _disc(radius)))


def feather(alpha: Alpha, radius: int) -> Alpha:
    """Soften the edge with a Gaussian whose support is `radius` pixels."""
    if radius <= 0:
        return alpha
    size = 2 * radius + 1
    return _as_alpha(cv2.GaussianBlur(_as_alpha(alpha), (size, size), 0))


def union(alphas: Sequence[Alpha]) -> Alpha:
    """Merge instance masks with a per-pixel maximum.

    Maximum rather than sum: overlapping instances in a multi-person photo would
    otherwise exceed 1.0 and clip into a hard-edged blob.
    """
    if not len(alphas):
        raise ValueError("union requires at least one mask")
    return _as_alpha(np.maximum.reduce([_as_alpha(a) for a in alphas]))


def derive_masks(
    alpha: Alpha,
    *,
    inpaint_grow: int,
    composite_shrink: int,
    feather_radius: int,
) -> DerivedMasks:
    """Derive the inpaint and composite masks from one segmentation alpha.

        inpaint_mask    = 1 - erode(alpha, inpaint_grow)
        composite_alpha = feather(erode(alpha, composite_shrink), feather_radius)

    `inpaint_grow` must exceed `composite_shrink + feather_radius` so the
    generated background always extends underneath the composited edge. At
    equality the softened edge reaches exactly as far as the regenerated region,
    leaving original pixels visible through a partially transparent border.
    """
    if inpaint_grow <= composite_shrink + feather_radius:
        raise ValueError(
            "inpaint_grow must exceed composite_shrink + feather_radius "
            f"(got {inpaint_grow} <= {composite_shrink} + {feather_radius}); "
            "otherwise a rim of stale background survives at the silhouette"
        )

    alpha = _as_alpha(alpha)
    return DerivedMasks(
        inpaint_mask=_as_alpha(1.0 - erode(alpha, inpaint_grow)),
        composite_alpha=feather(erode(alpha, composite_shrink), feather_radius),
    )


def alpha_to_mask(alpha: Alpha) -> Image.Image:
    """Convert float alpha to an 8-bit greyscale image."""
    quantised = (np.clip(alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(quantised, mode="L")


def mask_to_alpha(image: Image.Image) -> Alpha:
    """Convert a greyscale image to float alpha in [0, 1]."""
    return _as_alpha(np.asarray(image.convert("L"), dtype=np.float32) / 255.0)
