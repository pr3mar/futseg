"""The Segmenter and Inpainter protocol boundaries.

These protocols are what keep backends swappable without touching `pipeline.py`,
so the tests pin the two properties that are easy to erode: what counts as a
conforming implementation, and the fact that `inpaint` takes no prompt.
"""

import inspect

import numpy as np
from PIL import Image

from futseg.inpaint.base import Inpainter
from futseg.segmentation.base import Segmenter


class StubSegmenter:
    def segment(self, image: Image.Image) -> np.ndarray:
        return np.zeros((image.height, image.width), dtype=np.float32)


class StubInpainter:
    def inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        return image


class NotASegmenter:
    def predict(self, image: Image.Image) -> None: ...


def test_a_class_with_segment_satisfies_segmenter() -> None:
    assert isinstance(StubSegmenter(), Segmenter)


def test_a_class_without_segment_does_not_satisfy_segmenter() -> None:
    assert not isinstance(NotASegmenter(), Segmenter)


def test_a_class_with_inpaint_satisfies_inpainter() -> None:
    assert isinstance(StubInpainter(), Inpainter)


def test_a_class_without_inpaint_does_not_satisfy_inpainter() -> None:
    assert not isinstance(NotASegmenter(), Inpainter)


def test_inpaint_takes_no_prompt_argument() -> None:
    """Prompt and sampler settings are constructor-injected into the diffusion
    backend, so `composite.py` is never handed a parameter it would ignore.

    Fails the moment someone adds `prompt` to the call signature.
    """
    parameters = list(inspect.signature(Inpainter.inpaint).parameters)

    assert parameters == ["self", "image", "mask"]


def test_segment_takes_only_an_image() -> None:
    parameters = list(inspect.signature(Segmenter.segment).parameters)

    assert parameters == ["self", "image"]
