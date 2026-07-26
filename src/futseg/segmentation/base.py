"""Segmenter protocol: segment(image) -> HxW float32 alpha in [0, 1]."""

from typing import Protocol, runtime_checkable

from PIL import Image

from futseg.masking import Alpha


@runtime_checkable
class Segmenter(Protocol):
    """Produces a person alpha for an image.

    The return type is **float32 alpha, not a boolean mask**, even though both
    current backends only emit hard 0.0/1.0. Alpha matting is a scheduled upgrade
    and `bool -> float` is the expensive migration; `masking.py` needs float alpha
    for the composite regardless. Do not narrow this to `bool`.
    """

    def segment(self, image: Image.Image) -> Alpha:
        """Return an HxW float32 alpha in [0, 1] covering every person found."""
        ...
