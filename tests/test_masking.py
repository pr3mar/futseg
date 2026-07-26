"""The two-mask derivation and its morphology helpers.

The load-bearing test here is `test_no_gap_between_inpaint_and_composite`. If the
two masks are derived with the wrong offsets — or someone "simplifies" them into
one — a rim of stale original background survives at the silhouette, which is the
classic cheap-cutout halo. Synthetic masks only; no model required.
"""

import numpy as np
import pytest
from PIL import Image

from futseg.masking import (
    alpha_to_mask,
    derive_masks,
    dilate,
    erode,
    feather,
    mask_to_alpha,
    union,
)

SIZE = 128
CENTRE = SIZE // 2
RADIUS = 40


def disc(radius: int = RADIUS) -> np.ndarray:
    """A filled circle as float32 alpha — a silhouette with curvature."""
    yy, xx = np.ogrid[:SIZE, :SIZE]
    inside = (yy - CENTRE) ** 2 + (xx - CENTRE) ** 2 <= radius**2
    return inside.astype(np.float32)


def distance_inside(alpha: np.ndarray) -> np.ndarray:
    """Distance from each pixel to the outside of the shape, in pixels."""
    yy, xx = np.ogrid[:SIZE, :SIZE]
    return RADIUS - np.sqrt((yy - CENTRE) ** 2 + (xx - CENTRE) ** 2)


# --------------------------------------------------------------------------- #
# morphology helpers
# --------------------------------------------------------------------------- #


def test_erode_shrinks_the_shape() -> None:
    alpha = disc()
    assert erode(alpha, 5).sum() < alpha.sum()


def test_dilate_grows_the_shape() -> None:
    alpha = disc()
    assert dilate(alpha, 5).sum() > alpha.sum()


@pytest.mark.parametrize("op", [erode, dilate])
def test_zero_radius_is_identity(op) -> None:
    alpha = disc()
    np.testing.assert_array_equal(op(alpha, 0), alpha)


def test_erode_and_dilate_move_the_boundary_by_the_radius() -> None:
    """A 5px erosion of a disc of radius R leaves a disc of about radius R-5."""
    eroded_area = erode(disc(), 5).sum()
    expected = np.pi * (RADIUS - 5) ** 2
    assert eroded_area == pytest.approx(expected, rel=0.05)


def test_feather_creates_intermediate_values() -> None:
    """A hard mask has only 0 and 1; feathering must produce a soft edge."""
    softened = feather(disc(), 4)
    interior = (softened > 0.01) & (softened < 0.99)
    assert interior.any()


def test_feather_zero_radius_is_identity() -> None:
    alpha = disc()
    np.testing.assert_array_equal(feather(alpha, 0), alpha)


def test_feather_keeps_values_in_range() -> None:
    softened = feather(disc(), 6)
    assert softened.min() >= 0.0
    assert softened.max() <= 1.0


def test_union_takes_the_per_pixel_maximum() -> None:
    """Multi-person images union their instance masks; overlap must not sum."""
    left = np.zeros((4, 4), dtype=np.float32)
    left[:, :2] = 1.0
    right = np.zeros((4, 4), dtype=np.float32)
    right[:, 1:] = 0.5

    merged = union([left, right])

    expected = np.array([[1.0, 1.0, 0.5, 0.5]] * 4, dtype=np.float32)
    np.testing.assert_allclose(merged, expected)


def test_union_of_one_mask_is_that_mask() -> None:
    alpha = disc()
    np.testing.assert_array_equal(union([alpha]), alpha)


def test_union_rejects_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="at least one"):
        union([])


# --------------------------------------------------------------------------- #
# the two-mask derivation
# --------------------------------------------------------------------------- #


def test_inpaint_mask_grows_into_the_person() -> None:
    """It must cover more than the raw background, or the seam keeps stale pixels."""
    alpha = disc()
    masks = derive_masks(alpha, inpaint_grow=8, composite_shrink=2, feather_radius=3)

    assert masks.inpaint_mask.sum() > (1.0 - alpha).sum()


def test_composite_alpha_pulls_the_person_in() -> None:
    alpha = disc()
    masks = derive_masks(alpha, inpaint_grow=8, composite_shrink=2, feather_radius=3)

    assert masks.composite_alpha.sum() < alpha.sum()


def test_no_gap_between_inpaint_and_composite() -> None:
    """The invariant: wherever the person is not fully opaque, the background
    underneath it was regenerated.

    Fails if the offsets are swapped, if either mask is derived with the wrong
    sign, or if `inpaint_grow` stops exceeding `composite_shrink + feather_radius`.
    """
    alpha = disc()
    masks = derive_masks(alpha, inpaint_grow=8, composite_shrink=2, feather_radius=3)

    not_fully_person = masks.composite_alpha < 1.0 - 1e-6
    assert np.all(masks.inpaint_mask[not_fully_person] == 1.0)


def test_rejects_a_configuration_that_would_leave_a_halo() -> None:
    """`inpaint_grow > composite_shrink + feather_radius` is the whole contract;
    a caller that violates it gets an error, not a subtly haloed image."""
    with pytest.raises(ValueError, match="inpaint_grow"):
        derive_masks(disc(), inpaint_grow=4, composite_shrink=2, feather_radius=2)


def test_derived_masks_are_float32_in_range_and_same_shape() -> None:
    alpha = disc()
    masks = derive_masks(alpha, inpaint_grow=8, composite_shrink=2, feather_radius=3)

    for mask in (masks.inpaint_mask, masks.composite_alpha):
        assert mask.dtype == np.float32
        assert mask.shape == alpha.shape
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0


def test_empty_segmentation_inpaints_everything() -> None:
    """No person found: the whole canvas is background, nothing is composited."""
    alpha = np.zeros((16, 16), dtype=np.float32)
    masks = derive_masks(alpha, inpaint_grow=8, composite_shrink=2, feather_radius=3)

    assert masks.inpaint_mask.min() == 1.0
    assert masks.composite_alpha.max() == 0.0


# --------------------------------------------------------------------------- #
# PIL interop
# --------------------------------------------------------------------------- #


def test_alpha_survives_a_round_trip_through_pil() -> None:
    alpha = disc()
    restored = mask_to_alpha(alpha_to_mask(alpha))

    np.testing.assert_allclose(restored, alpha, atol=1.0 / 255)


def test_alpha_to_mask_produces_an_8_bit_greyscale_image() -> None:
    image = alpha_to_mask(disc())

    assert isinstance(image, Image.Image)
    assert image.mode == "L"
    assert image.size == (SIZE, SIZE)


def test_mask_to_alpha_normalises_to_unit_range() -> None:
    image = Image.new("L", (4, 4), color=255)

    alpha = mask_to_alpha(image)

    assert alpha.dtype == np.float32
    np.testing.assert_allclose(alpha, np.ones((4, 4), dtype=np.float32))
