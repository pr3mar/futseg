"""YOLO11-seg person segmentation over the instance union (--quality fast).

This is the *fast* tier, not the default. Ultralytics emits 32 mask prototypes at
1/4 stride, and each instance mask is a linear combination of those, upsampled and
bbox-cropped. On a 4000px-wide photo one mask pixel covers roughly 25 original
pixels, so hair, fingers and gaps between limbs are gone before any
post-processing runs. `retina_masks=True` improves the upsampling but not the
underlying information limit, which is why the default tier refines with SAM2.
"""

import cv2
import numpy as np
from PIL import Image

from futseg.masking import Alpha, fill_holes, union
from futseg.paths import weights_dir

#: COCO class index for "person".
PERSON_CLASS = 0

DEFAULT_WEIGHTS = "yolo11n-seg.pt"


class YoloSegmenter:
    """Segment every person in an image with YOLO11-seg.

    Implements the `Segmenter` protocol. Instances are unioned, so multi-person
    images work without any extra flag.
    """

    def __init__(
        self,
        *,
        device: str,
        weights: str = DEFAULT_WEIGHTS,
        imgsz: int = 1280,
        confidence: float = 0.25,
        fill_mask_holes: bool = True,
        model: object | None = None,
    ) -> None:
        """
        `device` is already resolved by `device.py`; this backend never asks torch
        whether CUDA is available. `imgsz` defaults above the ultralytics default
        of 640 because the prototype stride is what limits this tier.

        `model` is for tests, which inject a stub rather than download weights.
        """
        self.device = device
        self.imgsz = imgsz
        self.confidence = confidence
        self.fill_mask_holes = fill_mask_holes
        self.weights_path = weights_dir() / weights
        self._model = model

    def _load(self) -> object:
        """Load the model on first use, from an absolute path inside the cache.

        The path matters: given a bare filename, ultralytics downloads the
        checkpoint into the current working directory.
        """
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.weights_path))
        return self._model

    def segment(self, image: Image.Image) -> Alpha:
        """Return an HxW float32 alpha covering every person found."""
        results = self._load()(
            image,
            imgsz=self.imgsz,
            conf=self.confidence,
            device=self.device,
            retina_masks=True,
            verbose=False,
        )
        result = results[0]
        height, width = image.height, image.width

        if result.masks is None:
            return np.zeros((height, width), dtype=np.float32)

        instances = np.asarray(result.masks.data.cpu().numpy(), dtype=np.float32)
        classes = np.asarray(result.boxes.cls.cpu().numpy())

        # Filtering here rather than via the `classes=` predict argument keeps the
        # guarantee in code the tests can exercise, instead of depending on a
        # library keyword whose behaviour could change unnoticed.
        people = [
            mask
            for mask, cls in zip(instances, classes, strict=True)
            if int(cls) == PERSON_CLASS
        ]
        if not people:
            return np.zeros((height, width), dtype=np.float32)

        alpha = union(people)
        if alpha.shape != (height, width):
            alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)
        # After resizing, so the area threshold is measured at output scale.
        if self.fill_mask_holes:
            alpha = fill_holes(alpha)
        return np.ascontiguousarray(alpha, dtype=np.float32)
