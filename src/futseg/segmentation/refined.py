"""YOLO11 boxes -> SAM2 box-prompted refinement (default, --quality best).

The default tier, and the one carrying the project's edge-quality bar. YOLO11-seg
alone quantizes its masks to roughly 25 source pixels on a large photo, because
its 32 mask prototypes live at 1/4 stride; SAM2 recovers the silhouette at full
resolution. The old objection to SAM — that it needs external box or point
prompts — was self-defeating: the detector is the prompt source.

Still a *binary* silhouette. Semi-transparent hair needs alpha matting, which is
deferred, so this is high quality but not literally pixel-perfect.
"""

import cv2
import numpy as np
from PIL import Image

from futseg.masking import Alpha, union
from futseg.paths import weights_dir

#: COCO class index for "person", as emitted by the *detector*.
PERSON_CLASS = 0

DEFAULT_DETECTOR_WEIGHTS = "yolo11n.pt"
DEFAULT_SAM_WEIGHTS = "sam2_b.pt"


class RefinedSegmenter:
    """Detect people, then refine each detection into a mask with SAM2.

    Implements the `Segmenter` protocol.
    """

    def __init__(
        self,
        *,
        device: str,
        detector_weights: str = DEFAULT_DETECTOR_WEIGHTS,
        sam_weights: str = DEFAULT_SAM_WEIGHTS,
        imgsz: int = 1280,
        confidence: float = 0.25,
        detector: object | None = None,
        sam: object | None = None,
    ) -> None:
        """
        `device` is already resolved by `device.py`; neither model is asked whether
        CUDA is available. `detector` and `sam` are for tests, which inject stubs
        rather than download weights.
        """
        self.device = device
        self.imgsz = imgsz
        self.confidence = confidence
        self.detector_weights_path = weights_dir() / detector_weights
        self.sam_weights_path = weights_dir() / sam_weights
        self._detector = detector
        self._sam = sam

    def _load_detector(self) -> object:
        if self._detector is None:
            from ultralytics import YOLO

            self._detector = YOLO(str(self.detector_weights_path))
        return self._detector

    def _load_sam(self) -> object:
        if self._sam is None:
            from ultralytics import SAM

            # Absolute path, or ultralytics downloads the checkpoint into the
            # current working directory — observed, not assumed.
            self._sam = SAM(str(self.sam_weights_path))
        return self._sam

    def _person_boxes(self, image: Image.Image) -> list[list[float]]:
        """Detect people and return their boxes as SAM prompts."""
        result = self._load_detector()(
            image,
            imgsz=self.imgsz,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )[0]
        boxes = np.asarray(result.boxes.xyxy.cpu().numpy(), dtype=np.float32)
        classes = np.asarray(result.boxes.cls.cpu().numpy())
        return [
            box.tolist()
            for box, cls in zip(boxes, classes, strict=True)
            if int(cls) == PERSON_CLASS
        ]

    def segment(self, image: Image.Image) -> Alpha:
        """Return an HxW float32 alpha covering every person found."""
        height, width = image.height, image.width
        empty = np.zeros((height, width), dtype=np.float32)

        prompts = self._person_boxes(image)
        if not prompts:
            # No prompts means no work for SAM; calling it anyway would load and
            # run a model to produce nothing.
            return empty

        result = self._load_sam()(
            image,
            bboxes=prompts,
            device=self.device,
            verbose=False,
        )[0]
        if result.masks is None:
            return empty

        # Every returned mask corresponds to a prompt we chose, so all of them are
        # people. Do NOT filter on `result.boxes.cls` here: SAM2 numbers its masks
        # by prompt ordinal (0, 1, 2, ...), not by COCO class, so filtering
        # against PERSON_CLASS would silently keep only the first person.
        instances = np.asarray(result.masks.data.cpu().numpy(), dtype=np.float32)

        alpha = union(list(instances))
        if alpha.shape != (height, width):
            alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(alpha, dtype=np.float32)
