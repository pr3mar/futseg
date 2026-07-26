"""YOLO11 detection -> SAM2 box-prompted refinement: the default tier.

Both models are injected, so the fast suite exercises the wiring futseg owns —
person filtering before prompting, prompt construction, instance union, dtype
conversion — without downloading weights. A `@pytest.mark.slow` test runs the
real pair.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from futseg.segmentation.refined import PERSON_CLASS, RefinedSegmenter

WIDTH, HEIGHT = 64, 48


class FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._array

    def tolist(self) -> list:
        return self._array.tolist()


class FakeDetector:
    """Returns canned person/non-person boxes and records its call."""

    def __init__(self, boxes: np.ndarray, classes: list[int]) -> None:
        self._boxes = boxes
        self._classes = classes
        self.calls: list[dict] = []

    def __call__(self, image, **kwargs):
        self.calls.append(kwargs)
        return [
            SimpleNamespace(
                boxes=SimpleNamespace(
                    xyxy=FakeTensor(self._boxes),
                    cls=FakeTensor(np.array(self._classes, dtype=np.float32)),
                )
            )
        ]


class FakeSam:
    """Returns canned boolean masks, as SAM2 really does, and records prompts."""

    def __init__(self, masks: np.ndarray | None, classes: list[int] | None = None) -> None:
        self._masks = masks
        self._classes = classes
        self.calls: list[dict] = []

    def __call__(self, image, **kwargs):
        self.calls.append(kwargs)
        if self._masks is None:
            return [SimpleNamespace(masks=None, boxes=None)]
        # SAM2 numbers its results by prompt ordinal, not by semantic class.
        ordinals = self._classes or list(range(len(self._masks)))
        return [
            SimpleNamespace(
                masks=SimpleNamespace(data=FakeTensor(self._masks)),
                boxes=SimpleNamespace(cls=FakeTensor(np.array(ordinals, dtype=np.float32))),
            )
        ]


def image() -> Image.Image:
    return Image.new("RGB", (WIDTH, HEIGHT), color=(20, 30, 40))


def band(x0: int, x1: int) -> np.ndarray:
    """One boolean instance mask, as SAM2 emits them."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
    mask[:, x0:x1] = True
    return mask


def boxes(count: int) -> np.ndarray:
    return np.array([[i * 10.0, 0.0, i * 10.0 + 8, 40.0] for i in range(count)], dtype=np.float32)


def segmenter(detector, sam, **kwargs) -> RefinedSegmenter:
    return RefinedSegmenter(device="cpu", detector=detector, sam=sam, **kwargs)


def test_detected_boxes_are_passed_to_sam_as_prompts() -> None:
    detector = FakeDetector(boxes(2), [PERSON_CLASS, PERSON_CLASS])
    sam = FakeSam(np.stack([band(0, 8), band(10, 18)]))

    segmenter(detector, sam).segment(image())

    assert sam.calls[0]["bboxes"] == boxes(2).tolist()


def test_non_person_detections_are_never_prompted() -> None:
    """A dog's box must not become a SAM prompt, or it lands in the foreground.

    Fails if the class filter moves after the SAM call, or is dropped.
    """
    detector = FakeDetector(boxes(2), [PERSON_CLASS, 16])  # 16 = COCO dog
    sam = FakeSam(np.stack([band(0, 8)]))

    segmenter(detector, sam).segment(image())

    assert sam.calls[0]["bboxes"] == [boxes(2)[0].tolist()]


def test_sam_result_classes_are_prompt_ordinals_and_must_not_be_filtered() -> None:
    """SAM2 numbers its masks 0..N-1 by prompt order, not by COCO class.

    Filtering those against PERSON_CLASS — the pattern `yolo.py` correctly uses
    on *detection* output — would silently keep only the first person. This test
    fails the moment someone copies that pattern across.
    """
    detector = FakeDetector(boxes(3), [PERSON_CLASS] * 3)
    sam = FakeSam(np.stack([band(0, 8), band(20, 28), band(40, 48)]), classes=[0, 1, 2])

    alpha = segmenter(detector, sam).segment(image())

    assert alpha[:, 0:8].all()
    assert alpha[:, 20:28].all(), "second prompt's mask was dropped"
    assert alpha[:, 40:48].all(), "third prompt's mask was dropped"


def test_instance_masks_are_unioned() -> None:
    detector = FakeDetector(boxes(2), [PERSON_CLASS, PERSON_CLASS])
    sam = FakeSam(np.stack([band(0, 8), band(20, 28)]))

    alpha = segmenter(detector, sam).segment(image())

    assert alpha[:, 0:8].all()
    assert alpha[:, 20:28].all()
    assert not alpha[:, 10:18].any()


def test_boolean_masks_become_float32_alpha() -> None:
    """SAM2 returns dtype=bool; the protocol requires float32 in [0,1]."""
    detector = FakeDetector(boxes(1), [PERSON_CLASS])
    sam = FakeSam(np.stack([band(0, 8)]))

    alpha = segmenter(detector, sam).segment(image())

    assert alpha.dtype == np.float32
    assert set(np.unique(alpha).tolist()) <= {0.0, 1.0}
    assert alpha.shape == (HEIGHT, WIDTH)


def test_no_person_detected_skips_sam_entirely() -> None:
    """Prompting SAM with an empty box list would be a wasted model call."""
    detector = FakeDetector(boxes(1), [16])
    sam = FakeSam(np.stack([band(0, 8)]))

    alpha = segmenter(detector, sam).segment(image())

    assert not alpha.any()
    assert sam.calls == []


def test_no_detections_at_all_yields_empty_alpha() -> None:
    detector = FakeDetector(np.zeros((0, 4), dtype=np.float32), [])
    sam = FakeSam(None)

    alpha = segmenter(detector, sam).segment(image())

    assert alpha.shape == (HEIGHT, WIDTH)
    assert not alpha.any()


def test_sam_returning_no_masks_yields_empty_alpha() -> None:
    detector = FakeDetector(boxes(1), [PERSON_CLASS])
    sam = FakeSam(None)

    assert not segmenter(detector, sam).segment(image()).any()


def test_masks_at_another_resolution_are_resized() -> None:
    detector = FakeDetector(boxes(1), [PERSON_CLASS])
    small = np.ones((1, HEIGHT // 2, WIDTH // 2), dtype=bool)
    sam = FakeSam(small)

    assert segmenter(detector, sam).segment(image()).shape == (HEIGHT, WIDTH)


def test_resolved_device_reaches_both_models() -> None:
    """`device.py` owns the cuda/cpu decision for the detector *and* SAM."""
    detector = FakeDetector(boxes(1), [PERSON_CLASS])
    sam = FakeSam(np.stack([band(0, 8)]))

    RefinedSegmenter(device="cuda", detector=detector, sam=sam).segment(image())

    assert detector.calls[0]["device"] == "cuda"
    assert sam.calls[0]["device"] == "cuda"


def test_both_checkpoints_resolve_into_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Confirmed the hard way: SAM downloads into the CWD given a bare filename."""
    monkeypatch.setenv("FUTSEG_CACHE_DIR", str(tmp_path))

    seg = RefinedSegmenter(device="cpu", detector=FakeDetector(boxes(0), []), sam=FakeSam(None))

    for path in (seg.detector_weights_path, seg.sam_weights_path):
        assert path.is_absolute()
        assert tmp_path in path.parents


@pytest.mark.slow
def test_real_models_refine_a_real_photo() -> None:
    """Runs YOLO11 detection and SAM2 for real, proving the fakes match the API.

    Also the only place the two-stage wiring is exercised against ultralytics'
    actual return types — notably that SAM2 masks arrive as bool.
    """
    from ultralytics.utils import ASSETS

    from futseg.io import load_image

    photo = load_image(Path(ASSETS) / "bus.jpg")
    alpha = RefinedSegmenter(device="cpu").segment(photo)

    assert alpha.dtype == np.float32
    assert alpha.shape == (photo.height, photo.width)
    assert alpha.max() <= 1.0
    assert alpha.any(), "expected at least one person in bus.jpg"
    assert 0.01 < alpha.mean() < 0.6
