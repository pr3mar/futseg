"""Inpainter protocol: inpaint(image, mask) -> full-canvas image."""

from typing import Protocol, runtime_checkable

from PIL import Image

from futseg.masking import Alpha


@runtime_checkable
class Inpainter(Protocol):
    """Regenerates the masked region of an image.

    There is deliberately **no prompt parameter**. Prompt and sampler settings are
    constructor-injected into the diffusion backend, so the non-generative
    composite backend is never handed an argument it would ignore.
    """

    def inpaint(self, image: Image.Image, mask: Alpha) -> Image.Image:
        """Return a full-canvas image with the masked region regenerated."""
        ...
