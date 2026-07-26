"""Non-generative inpainting backends.

Pillow only, no model, no GPU. These exist so segmentation and pipeline plumbing
can be validated on real photographs without downloading diffusion weights — a
fast dev loop, not a quality fallback.
"""

import numpy as np
import pytest
from PIL import Image

from futseg.inpaint.composite import BlurInpainter, ImageInpainter, SolidColorInpainter

WIDTH, HEIGHT = 32, 24
FILL = (255, 0, 255)


def photo() -> Image.Image:
    """A textured canvas.

    Deliberately not a flat colour: blurring a uniform image returns the same
    uniform image, so "the masked region was replaced" would be unassertable for
    `BlurInpainter` and the test would pass for the wrong reason.
    """
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 200, size=(HEIGHT, WIDTH, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def right_half_mask() -> np.ndarray:
    """1.0 where the backend should regenerate."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    mask[:, WIDTH // 2 :] = 1.0
    return mask


@pytest.fixture(params=["solid", "blur", "image"])
def inpainter(request):
    return {
        "solid": SolidColorInpainter(color=FILL),
        "blur": BlurInpainter(radius=4),
        "image": ImageInpainter(background=Image.new("RGB", (8, 8), color=FILL)),
    }[request.param]


def test_returns_a_full_canvas_image(inpainter) -> None:
    """The protocol promises a full canvas, not a crop of the masked region."""
    result = inpainter.inpaint(photo(), right_half_mask())

    assert isinstance(result, Image.Image)
    assert result.size == (WIDTH, HEIGHT)
    assert result.mode == "RGB"


def test_unmasked_pixels_are_left_exactly_alone(inpainter) -> None:
    """Anything outside the mask must survive byte-identical.

    Fails if a backend regenerates the whole canvas and relies on the pipeline's
    later composite to hide it.
    """
    original = photo()
    result = inpainter.inpaint(original, right_half_mask())

    np.testing.assert_array_equal(
        np.asarray(result)[:, : WIDTH // 2], np.asarray(original)[:, : WIDTH // 2]
    )


def test_masked_pixels_are_replaced(inpainter) -> None:
    original = photo()
    result = inpainter.inpaint(original, right_half_mask())

    right_before = np.asarray(original)[:, WIDTH // 2 :]
    right_after = np.asarray(result)[:, WIDTH // 2 :]
    assert not np.array_equal(right_after, right_before)


def test_an_empty_mask_changes_nothing(inpainter) -> None:
    empty = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

    result = inpainter.inpaint(photo(), empty)

    np.testing.assert_array_equal(np.asarray(result), np.asarray(photo()))


def test_solid_colour_fills_with_exactly_that_colour() -> None:
    result = SolidColorInpainter(color=FILL).inpaint(photo(), right_half_mask())

    right = np.asarray(result)[:, WIDTH // 2 :]
    assert (right == np.array(FILL, dtype=np.uint8)).all()


def test_blur_leaves_a_flat_image_flat_but_smears_detail() -> None:
    """A blurred uniform image is still uniform, so use a real edge to prove it."""
    stripes = np.array([[0, 255]] * HEIGHT, dtype=np.uint8).repeat(WIDTH // 2, axis=1)
    striped = Image.fromarray(np.tile(stripes[..., None], 3))
    full = np.ones((HEIGHT, WIDTH), dtype=np.float32)

    result = np.asarray(BlurInpainter(radius=5).inpaint(striped, full))

    assert len(np.unique(result)) > 2, "blur should introduce intermediate values"


def test_static_background_is_resized_to_the_canvas() -> None:
    """A background of any size must cover the frame, not tile or crop."""
    small = Image.new("RGB", (4, 3), color=FILL)
    full = np.ones((HEIGHT, WIDTH), dtype=np.float32)

    result = ImageInpainter(background=small).inpaint(photo(), full)

    assert result.size == (WIDTH, HEIGHT)
    assert (np.asarray(result) == np.array(FILL, dtype=np.uint8)).all()


def test_a_soft_mask_blends_rather_than_hard_switching() -> None:
    """The protocol takes float alpha; a 0.5 mask must produce a real blend."""
    original = photo()
    half = np.full((HEIGHT, WIDTH), 0.5, dtype=np.float32)

    result = np.asarray(SolidColorInpainter(color=FILL).inpaint(original, half))

    assert not np.array_equal(result, np.asarray(original)), "did not move toward the fill"
    assert not (result == np.array(FILL, dtype=np.uint8)).all(), "hard-switched to the fill"


def test_backends_satisfy_the_inpainter_protocol(inpainter) -> None:
    from futseg.inpaint.base import Inpainter

    assert isinstance(inpainter, Inpainter)
