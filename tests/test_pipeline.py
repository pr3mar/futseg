"""End-to-end orchestration: segment -> derive masks -> inpaint -> composite.

Backends are stubs, so nothing here downloads weights. The load-bearing test is
`test_no_original_background_survives_outside_the_person`: it fills the generated
background with a colour that appears nowhere in the input, so a single surviving
original pixel outside the subject is detectable. That is the cheap-cutout halo,
caught arithmetically instead of by squinting at a photograph.
"""

import numpy as np
import pytest
from PIL import Image

from futseg.masking import derive_masks
from futseg.pipeline import (
    DEFAULT_COMPOSITE_SHRINK,
    DEFAULT_FEATHER_RADIUS,
    DEFAULT_INPAINT_GROW,
    run,
)

SIZE = 128
FILL = (255, 0, 255)  # appears nowhere in the noise below


def noisy_photo(seed: int = 0) -> Image.Image:
    """Random pixels, so "the original leaked through" is detectable."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 200, size=(SIZE, SIZE, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def disc_alpha(radius: int = 40) -> np.ndarray:
    yy, xx = np.ogrid[:SIZE, :SIZE]
    centre = SIZE // 2
    return (((yy - centre) ** 2 + (xx - centre) ** 2) <= radius**2).astype(np.float32)


class StubSegmenter:
    def __init__(self, alpha: np.ndarray) -> None:
        self._alpha = alpha
        self.seen: list[Image.Image] = []

    def segment(self, image: Image.Image) -> np.ndarray:
        self.seen.append(image)
        return self._alpha


class RecordingInpainter:
    """Paints the masked region a flat colour and remembers the mask it got."""

    def __init__(self, color=FILL) -> None:
        self.color = color
        self.masks: list[np.ndarray] = []

    def inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        self.masks.append(mask.copy())
        pixels = np.asarray(image, dtype=np.float32)
        fill = np.broadcast_to(np.array(self.color, dtype=np.float32), pixels.shape)
        blended = pixels * (1.0 - mask[..., None]) + fill * mask[..., None]
        return Image.fromarray(blended.round().astype(np.uint8), mode="RGB")


def test_output_matches_the_input_canvas() -> None:
    result = run(
        noisy_photo(), segmenter=StubSegmenter(disc_alpha()), inpainter=RecordingInpainter()
    )

    assert result.size == (SIZE, SIZE)
    assert result.mode == "RGB"


def test_the_person_is_returned_untouched() -> None:
    """The original subject is pasted back over the backend's output, so where
    the composite alpha is fully opaque the pixels must be byte-identical.

    Fails if the composite step is dropped, or if the person is taken from the
    backend's canvas rather than the original.
    """
    photo = noisy_photo()
    alpha = disc_alpha()

    result = run(photo, segmenter=StubSegmenter(alpha), inpainter=RecordingInpainter())

    masks = derive_masks(
        alpha,
        inpaint_grow=DEFAULT_INPAINT_GROW,
        composite_shrink=DEFAULT_COMPOSITE_SHRINK,
        feather_radius=DEFAULT_FEATHER_RADIUS,
    )
    fully_person = masks.composite_alpha >= 1.0 - 1e-6
    np.testing.assert_array_equal(
        np.asarray(result)[fully_person], np.asarray(photo)[fully_person]
    )


def test_no_original_background_survives_outside_the_person() -> None:
    """The anti-halo guarantee.

    Every pixel the subject does not cover at all must come from the generated
    background. Because the background is a flat colour absent from the input, a
    surviving original pixel is unambiguous — this is the stale rim that hugs a
    cheap cutout, made arithmetic.
    """
    photo = noisy_photo()
    alpha = disc_alpha()

    result = run(photo, segmenter=StubSegmenter(alpha), inpainter=RecordingInpainter())

    masks = derive_masks(
        alpha,
        inpaint_grow=DEFAULT_INPAINT_GROW,
        composite_shrink=DEFAULT_COMPOSITE_SHRINK,
        feather_radius=DEFAULT_FEATHER_RADIUS,
    )
    no_person = masks.composite_alpha <= 0.0
    background = np.asarray(result)[no_person]

    assert (background == np.array(FILL, dtype=np.uint8)).all(), (
        "original background leaked outside the subject — this is the halo"
    )


def test_the_seam_blends_rather_than_cutting() -> None:
    """Between fully-person and fully-background there must be a soft band, or
    the composite is a hard cutout with visible stair-stepping."""
    result = np.asarray(
        run(noisy_photo(), segmenter=StubSegmenter(disc_alpha()), inpainter=RecordingInpainter())
    )

    is_fill = (result == np.array(FILL, dtype=np.uint8)).all(axis=-1)
    partially_fill = (result[..., 0] > 200) & (result[..., 1] > 0) & ~is_fill
    assert partially_fill.any(), "expected a feathered band between subject and background"


def test_the_backend_receives_the_derived_mask_not_the_raw_alpha() -> None:
    """The backend regenerates `1 - erode(alpha, k)`, which is larger than the
    plain background. Handing it `1 - alpha` would reintroduce the halo.
    """
    alpha = disc_alpha()
    inpainter = RecordingInpainter()

    run(noisy_photo(), segmenter=StubSegmenter(alpha), inpainter=inpainter)

    given = inpainter.masks[0]
    assert given.sum() > (1.0 - alpha).sum()


def test_an_empty_segmentation_replaces_the_whole_canvas() -> None:
    empty = np.zeros((SIZE, SIZE), dtype=np.float32)

    result = run(noisy_photo(), segmenter=StubSegmenter(empty), inpainter=RecordingInpainter())

    assert (np.asarray(result) == np.array(FILL, dtype=np.uint8)).all()


def test_a_full_frame_person_is_returned_unchanged() -> None:
    photo = noisy_photo()
    everything = np.ones((SIZE, SIZE), dtype=np.float32)

    result = run(photo, segmenter=StubSegmenter(everything), inpainter=RecordingInpainter())

    np.testing.assert_array_equal(np.asarray(result), np.asarray(photo))


def test_the_segmenter_sees_the_input_image() -> None:
    photo = noisy_photo()
    segmenter = StubSegmenter(disc_alpha())

    run(photo, segmenter=segmenter, inpainter=RecordingInpainter())

    assert segmenter.seen == [photo]


def test_an_unsafe_mask_configuration_is_rejected() -> None:
    """`derive_masks` guards `k > j + feather`; the pipeline must not swallow it."""
    with pytest.raises(ValueError, match="inpaint_grow"):
        run(
            noisy_photo(),
            segmenter=StubSegmenter(disc_alpha()),
            inpainter=RecordingInpainter(),
            inpaint_grow=2,
            composite_shrink=2,
            feather_radius=4,
        )


def test_defaults_satisfy_the_halo_constraint() -> None:
    """The shipped defaults must themselves be a legal configuration."""
    assert DEFAULT_INPAINT_GROW > DEFAULT_COMPOSITE_SHRINK + DEFAULT_FEATHER_RADIUS
