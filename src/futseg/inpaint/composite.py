"""Non-generative Pillow-only backend: a fast dev loop, not a quality fallback.

These exist so segmentation and pipeline plumbing can be validated on real
photographs without downloading multi-GB diffusion weights. No model, no GPU, no
prompt — which is precisely why the `Inpainter` protocol takes no prompt argument.

Three backends rather than one with a mode flag: each is a handful of lines with
no branching, and the protocol already makes them interchangeable.
"""

import numpy as np
from PIL import Image, ImageFilter

from futseg.masking import Alpha, _as_alpha


def _blend(original: Image.Image, replacement: Image.Image, mask: Alpha) -> Image.Image:
    """Composite `replacement` into `original` wherever `mask` is non-zero.

    Alpha-weighted rather than a hard switch: the protocol carries float masks,
    and a soft mask must produce a real blend.
    """
    weights = _as_alpha(mask)[..., None]
    base = np.asarray(original.convert("RGB"), dtype=np.float32)
    new = np.asarray(replacement.convert("RGB").resize(original.size), dtype=np.float32)
    blended = base * (1.0 - weights) + new * weights
    return Image.fromarray(blended.round().clip(0, 255).astype(np.uint8), mode="RGB")


class SolidColorInpainter:
    """Replace the masked region with a flat colour."""

    def __init__(self, color: tuple[int, int, int] = (0, 0, 0)) -> None:
        self.color = color

    def inpaint(self, image: Image.Image, mask: Alpha) -> Image.Image:
        return _blend(image, Image.new("RGB", image.size, color=self.color), mask)


class BlurInpainter:
    """Replace the masked region with a blurred copy of the original.

    Useful for eyeballing a seam: the background stays plausible, so a stale rim
    at the silhouette stands out rather than hiding against a flat fill.
    """

    def __init__(self, radius: int = 24) -> None:
        self.radius = radius

    def inpaint(self, image: Image.Image, mask: Alpha) -> Image.Image:
        blurred = image.convert("RGB").filter(ImageFilter.GaussianBlur(self.radius))
        return _blend(image, blurred, mask)


class ImageInpainter:
    """Replace the masked region with a static background image.

    The background is resized to the canvas rather than tiled or cropped, so the
    frame is always covered whatever its aspect ratio.
    """

    def __init__(self, background: Image.Image) -> None:
        self.background = background

    def inpaint(self, image: Image.Image, mask: Alpha) -> Image.Image:
        return _blend(image, self.background, mask)
