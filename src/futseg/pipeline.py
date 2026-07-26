"""Orchestration: segment -> mask derivation -> inpaint -> composite.

The pipeline knows nothing about which backends it is given: it takes a
`Segmenter` and an `Inpainter` and never inspects them, so swapping YOLO for SAM2
or a flat fill for a diffusion model changes nothing here.

The final step is not optional. Whatever the backend produced, the original
subject is composited back over it through `composite_alpha`, so the person is
never altered by the generative model — not even at the mask boundary, which is
exactly where a diffusion model is most tempted to redraw an ear.
"""

import numpy as np
from PIL import Image

from futseg.inpaint.base import Inpainter
from futseg.masking import alpha_to_mask, derive_masks
from futseg.segmentation.base import Segmenter

#: Grow the inpaint region this far *into* the subject.
DEFAULT_INPAINT_GROW = 12
#: Pull the composited subject this far in from its own silhouette.
DEFAULT_COMPOSITE_SHRINK = 3
#: Soften the composited edge by this radius.
DEFAULT_FEATHER_RADIUS = 5


def run(
    image: Image.Image,
    *,
    segmenter: Segmenter,
    inpainter: Inpainter,
    inpaint_grow: int = DEFAULT_INPAINT_GROW,
    composite_shrink: int = DEFAULT_COMPOSITE_SHRINK,
    feather_radius: int = DEFAULT_FEATHER_RADIUS,
) -> Image.Image:
    """Replace the background of `image`, keeping the subject pixel-identical.

    Raises `ValueError` through `derive_masks` if the mask offsets would leave a
    stale rim; that guard is deliberately not swallowed here.
    """
    alpha = segmenter.segment(image)
    masks = derive_masks(
        alpha,
        inpaint_grow=inpaint_grow,
        composite_shrink=composite_shrink,
        feather_radius=feather_radius,
    )

    background = inpainter.inpaint(image, masks.inpaint_mask)

    # Image.composite takes the first image where the mask is white, so the
    # original wins wherever the subject is opaque and the generated background
    # wins elsewhere, with a feathered band between them.
    return Image.composite(
        image.convert("RGB"),
        background.convert("RGB"),
        alpha_to_mask(np.asarray(masks.composite_alpha)),
    )
