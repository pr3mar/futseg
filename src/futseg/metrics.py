"""Mask quality metrics.

Plain IoU is dominated by the torso and barely moves when the silhouette is
wrong, which makes it useless as a signal for the thing this project actually
cares about. Boundary IoU compares only a thin band along each mask's rim, so
edge errors are not averaged away by a large correct interior.

Reference: Cheng et al., "Boundary IoU: Improving Object-Centric Image
Segmentation Evaluation" (CVPR 2021).
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from futseg.masking import Alpha, _as_alpha


def _binarise(mask: Alpha, threshold: float = 0.5) -> NDArray[np.uint8]:
    return (_as_alpha(mask) >= threshold).astype(np.uint8)


def _boundary_band(mask: NDArray[np.uint8], radius: int) -> NDArray[np.uint8]:
    """The rim of a mask: everything it loses when eroded by `radius`."""
    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    # Pad before eroding so a shape touching the image border keeps that edge as
    # interior rather than gaining a false boundary along the frame.
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    eroded = cv2.erode(padded, kernel)[1:-1, 1:-1]
    return mask - eroded


def _band_radius(shape: tuple[int, int], ratio: float) -> int:
    """Band width as a fraction of the image diagonal, per the paper."""
    diagonal = float(np.sqrt(shape[0] ** 2 + shape[1] ** 2))
    return max(1, int(round(ratio * diagonal)))


def iou(prediction: Alpha, truth: Alpha, threshold: float = 0.5) -> float:
    """Plain intersection over union. Kept for comparison, not for judging edges."""
    pred, gt = _binarise(prediction, threshold), _binarise(truth, threshold)
    if pred.shape != gt.shape:
        raise ValueError(f"masks must have the same shape, got {pred.shape} and {gt.shape}")
    union = np.count_nonzero(pred | gt)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(pred & gt) / union)


def boundary_iou(
    prediction: Alpha,
    truth: Alpha,
    dilation_ratio: float = 0.02,
    threshold: float = 0.5,
) -> float:
    """IoU restricted to a band along each mask's boundary.

    `dilation_ratio` sets the band width as a fraction of the image diagonal.
    Two empty masks score 1.0: agreeing that there is nothing here is a correct
    answer, not an undefined one.
    """
    pred, gt = _binarise(prediction, threshold), _binarise(truth, threshold)
    if pred.shape != gt.shape:
        raise ValueError(f"masks must have the same shape, got {pred.shape} and {gt.shape}")

    radius = _band_radius(pred.shape, dilation_ratio)
    pred_band = _boundary_band(pred, radius)
    gt_band = _boundary_band(gt, radius)

    union = np.count_nonzero(pred_band | gt_band)
    if union == 0:
        return 1.0
    return float(np.count_nonzero(pred_band & gt_band) / union)
