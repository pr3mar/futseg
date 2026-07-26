"""Boundary IoU.

The project's quality bar is about edges, and plain IoU is dominated by the
torso: a mask can be badly wrong at the silhouette and still score well. These
tests pin the property that makes boundary IoU worth having — that it *falls*
where plain IoU barely moves.
"""

import numpy as np
import pytest

from futseg.metrics import boundary_iou, iou

SIZE = 200


def disc(radius: int, centre: tuple[int, int] = (SIZE // 2, SIZE // 2)) -> np.ndarray:
    yy, xx = np.ogrid[:SIZE, :SIZE]
    cy, cx = centre
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2).astype(np.float32)


def ragged_edge(radius: int, teeth: int = 24, depth: int = 6) -> np.ndarray:
    """A disc whose boundary is chewed by radial notches.

    Area is nearly unchanged; the silhouette is visibly wrong. That is exactly
    the failure the project cares about and plain IoU shrugs at.
    """
    yy, xx = np.ogrid[:SIZE, :SIZE]
    cy = cx = SIZE // 2
    dy, dx = yy - cy, xx - cx
    distance = np.sqrt(dy**2 + dx**2)
    angle = np.arctan2(dy, dx)
    wobble = radius - depth * (np.cos(angle * teeth) > 0)
    return (distance <= wobble).astype(np.float32)


def test_identical_masks_score_one() -> None:
    mask = disc(60)
    assert boundary_iou(mask, mask) == pytest.approx(1.0)


def test_disjoint_masks_score_zero() -> None:
    left = disc(30, centre=(60, 40))
    right = disc(30, centre=(140, 160))
    assert boundary_iou(left, right) == pytest.approx(0.0)


def test_empty_masks_score_one() -> None:
    """Two correct 'no person here' answers agree perfectly."""
    empty = np.zeros((SIZE, SIZE), dtype=np.float32)
    assert boundary_iou(empty, empty) == pytest.approx(1.0)


def test_prediction_empty_against_non_empty_scores_zero() -> None:
    assert boundary_iou(np.zeros((SIZE, SIZE), dtype=np.float32), disc(60)) == pytest.approx(0.0)


def test_it_is_symmetric() -> None:
    a, b = disc(60), ragged_edge(60)
    assert boundary_iou(a, b) == pytest.approx(boundary_iou(b, a))


def test_a_ragged_edge_hurts_boundary_iou_far_more_than_plain_iou() -> None:
    """The whole reason this metric exists.

    Fails if boundary_iou degenerates into plain IoU — e.g. if the boundary band
    is computed over the whole mask rather than its rim.
    """
    truth = disc(60)
    ragged = ragged_edge(60)

    plain = iou(ragged, truth)
    boundary = boundary_iou(ragged, truth)

    assert plain > 0.85, "the shapes should still overlap heavily by area"
    assert boundary < plain - 0.2, "boundary IoU must punish the chewed silhouette"


def test_a_shrunken_mask_scores_worse_at_the_boundary() -> None:
    truth = disc(60)
    shrunk = disc(54)

    assert boundary_iou(shrunk, truth) < iou(shrunk, truth)


def test_mismatched_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="same shape"):
        boundary_iou(np.zeros((4, 4), dtype=np.float32), np.zeros((5, 5), dtype=np.float32))


def test_plain_iou_behaves() -> None:
    mask = disc(60)
    assert iou(mask, mask) == pytest.approx(1.0)
    assert iou(mask, np.zeros_like(mask)) == pytest.approx(0.0)
