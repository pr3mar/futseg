"""Filling pinholes in a segmentation mask, without erasing real gaps.

Observed on a real photo (#28): a subject's raised hand came back peppered with
single-pixel holes and glasses lenses came back as background, so the generated
backdrop painted straight through them.

The obvious fix — flood-fill from the border, call everything unreached subject —
is wrong. A hand on a hip encloses a triangle of *genuine* background that the
border cannot reach either, and filling it would weld the arm to the torso. So
the fill is bounded by area: small enclosed regions are noise, large ones are
real, and the threshold is what separates them.
"""

import numpy as np
import pytest

from futseg.masking import fill_holes

SIZE = 200


def solid_block() -> np.ndarray:
    """A subject occupying the middle of the frame."""
    alpha = np.zeros((SIZE, SIZE), dtype=np.float32)
    alpha[40:160, 40:160] = 1.0
    return alpha


def test_a_mask_without_holes_is_unchanged() -> None:
    alpha = solid_block()
    np.testing.assert_array_equal(fill_holes(alpha), alpha)


def test_a_single_pixel_hole_is_filled() -> None:
    alpha = solid_block()
    alpha[100, 100] = 0.0

    assert fill_holes(alpha)[100, 100] == 1.0


def test_scattered_speckle_is_filled() -> None:
    """The observed defect: a shoulder peppered with pinholes."""
    rng = np.random.default_rng(0)
    alpha = solid_block()
    ys = rng.integers(50, 150, 200)
    xs = rng.integers(50, 150, 200)
    alpha[ys, xs] = 0.0

    filled = fill_holes(alpha)

    assert filled[50:150, 50:150].all()


def test_a_large_enclosed_gap_is_preserved() -> None:
    """A hand on a hip encloses real background.

    This is the test that stops the naive flood-fill implementation: it is
    unreachable from the border, exactly like a pinhole, and must survive anyway.
    """
    alpha = solid_block()
    alpha[70:130, 70:130] = 0.0  # 3600 px, 25% of the subject

    filled = fill_holes(alpha)

    assert not filled[70:130, 70:130].any(), "a real enclosed gap was welded shut"


def test_the_threshold_is_a_fraction_of_the_subject_not_absolute_pixels() -> None:
    """The same hole is noise on a 40 MP photo and a real gap on a thumbnail, so
    an absolute pixel count cannot be right for both."""
    small = np.zeros((60, 60), dtype=np.float32)
    small[10:50, 10:50] = 1.0
    small[28:32, 28:32] = 0.0  # 16 px in a 1600 px subject == 1%

    large = solid_block()  # 14400 px subject
    large[100:104, 100:104] = 0.0  # 16 px == 0.11%

    # At a 0.5% threshold the same 16px hole is above it in one and below in the
    # other, and the outcomes must differ accordingly.
    assert not fill_holes(small, max_hole_ratio=0.005)[28:32, 28:32].any()
    assert fill_holes(large, max_hole_ratio=0.005)[100:104, 100:104].all()


def test_background_touching_the_border_is_never_filled() -> None:
    """Everything outside the subject reaches the frame edge; filling that would
    turn the whole canvas into subject."""
    alpha = solid_block()

    filled = fill_holes(alpha)

    assert not filled[0, :].any()
    assert not filled[:, 0].any()
    assert filled.mean() < 0.5


def test_a_hole_touching_the_border_is_not_a_hole() -> None:
    """A notch cut in from the frame edge is background, however small."""
    alpha = np.ones((SIZE, SIZE), dtype=np.float32)
    alpha[0:3, 0:3] = 0.0

    assert not fill_holes(alpha)[0:3, 0:3].any()


def test_an_empty_mask_stays_empty() -> None:
    """No person found must not become a full-frame subject."""
    empty = np.zeros((SIZE, SIZE), dtype=np.float32)

    assert not fill_holes(empty).any()


def test_a_full_mask_stays_full() -> None:
    full = np.ones((SIZE, SIZE), dtype=np.float32)

    assert fill_holes(full).all()


def test_output_is_float32_in_range_with_the_same_shape() -> None:
    alpha = solid_block()
    alpha[100, 100] = 0.0

    filled = fill_holes(alpha)

    assert filled.dtype == np.float32
    assert filled.shape == alpha.shape
    assert filled.min() >= 0.0
    assert filled.max() <= 1.0


def test_disabling_the_threshold_fills_nothing() -> None:
    """`max_hole_ratio=0` is the escape hatch for a caller that wants the raw
    segmentation."""
    alpha = solid_block()
    alpha[100, 100] = 0.0

    assert fill_holes(alpha, max_hole_ratio=0.0)[100, 100] == 0.0


def test_soft_alpha_values_are_not_hardened() -> None:
    """The protocol carries float alpha; filling holes must not quantise the
    feathered edges the matting upgrade will eventually produce."""
    alpha = solid_block()
    alpha[40:160, 40] = 0.5

    filled = fill_holes(alpha)

    assert filled[100, 40] == pytest.approx(0.5)
