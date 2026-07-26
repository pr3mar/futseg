"""YOLO11-seg fast tier.

The model is injected, so the fast suite exercises the parts futseg owns — class
filtering, instance union, resizing, dtype — without downloading weights. A
`@pytest.mark.slow` test at the bottom runs the real model, because a fake can
only prove the code is self-consistent, not that it matches ultralytics.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from futseg.segmentation.yolo import PERSON_CLASS, YoloSegmenter

WIDTH, HEIGHT = 64, 48


class FakeTensor:
    """Mimics the torch tensor ultralytics returns, without importing torch."""

    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class FakeModel:
    """Records how it was called and replays canned instance masks."""

    def __init__(self, masks: np.ndarray | None, classes: list[int] | None = None) -> None:
        self._masks = masks
        self._classes = classes if classes is not None else []
        self.calls: list[dict] = []

    def __call__(self, image, **kwargs):
        self.calls.append(kwargs)
        if self._masks is None:
            empty = SimpleNamespace(cls=FakeTensor(np.array([])))
            result = SimpleNamespace(masks=None, boxes=empty)
        else:
            result = SimpleNamespace(
                masks=SimpleNamespace(data=FakeTensor(self._masks)),
                boxes=SimpleNamespace(cls=FakeTensor(np.array(self._classes, dtype=np.float32))),
            )
        return [result]


def image() -> Image.Image:
    return Image.new("RGB", (WIDTH, HEIGHT), color=(20, 30, 40))


def instance(x0: int, x1: int) -> np.ndarray:
    """One instance mask covering a vertical band of the canvas."""
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    mask[:, x0:x1] = 1.0
    return mask


def test_single_person_mask_is_returned() -> None:
    model = FakeModel(np.stack([instance(10, 20)]), classes=[PERSON_CLASS])

    alpha = YoloSegmenter(device="cpu", model=model).segment(image())

    assert alpha[:, 10:20].all()
    assert not alpha[:, :10].any()


def test_multiple_people_are_unioned() -> None:
    """Multi-person is supported by default; both bands must survive."""
    masks = np.stack([instance(5, 15), instance(40, 50)])
    model = FakeModel(masks, classes=[PERSON_CLASS, PERSON_CLASS])

    alpha = YoloSegmenter(device="cpu", model=model).segment(image())

    assert alpha[:, 5:15].all()
    assert alpha[:, 40:50].all()
    assert not alpha[:, 20:35].any()


def test_non_person_instances_are_discarded() -> None:
    """A dog beside the subject must not become foreground.

    Fails if the class filter is dropped and every instance is unioned.
    """
    masks = np.stack([instance(5, 15), instance(40, 50)])
    model = FakeModel(masks, classes=[PERSON_CLASS, 16])  # 16 = COCO dog

    alpha = YoloSegmenter(device="cpu", model=model).segment(image())

    assert alpha[:, 5:15].all()
    assert not alpha[:, 40:50].any()


def test_no_detections_yields_an_empty_alpha() -> None:
    """No person found is a valid outcome: the whole canvas is background."""
    alpha = YoloSegmenter(device="cpu", model=FakeModel(None)).segment(image())

    assert alpha.shape == (HEIGHT, WIDTH)
    assert not alpha.any()


def test_detections_without_a_person_yield_an_empty_alpha() -> None:
    model = FakeModel(np.stack([instance(5, 15)]), classes=[16])

    assert not YoloSegmenter(device="cpu", model=model).segment(image()).any()


def test_alpha_is_float32_in_unit_range_at_image_resolution() -> None:
    model = FakeModel(np.stack([instance(10, 20)]), classes=[PERSON_CLASS])

    alpha = YoloSegmenter(device="cpu", model=model).segment(image())

    assert alpha.dtype == np.float32
    assert alpha.shape == (HEIGHT, WIDTH)
    assert alpha.min() >= 0.0
    assert alpha.max() <= 1.0


def test_masks_returned_at_another_resolution_are_resized() -> None:
    """`retina_masks=True` should give original-resolution masks, but the alpha
    must match the image regardless — a mismatch would break the derivation."""
    small = np.ones((1, HEIGHT // 2, WIDTH // 2), dtype=np.float32)
    model = FakeModel(small, classes=[PERSON_CLASS])

    alpha = YoloSegmenter(device="cpu", model=model).segment(image())

    assert alpha.shape == (HEIGHT, WIDTH)


def test_prediction_uses_retina_masks_and_the_configured_settings() -> None:
    model = FakeModel(np.stack([instance(10, 20)]), classes=[PERSON_CLASS])

    YoloSegmenter(device="cuda", model=model, imgsz=1280, confidence=0.4).segment(image())

    kwargs = model.calls[0]
    assert kwargs["retina_masks"] is True
    assert kwargs["imgsz"] == 1280
    assert kwargs["device"] == "cuda"
    assert kwargs["conf"] == 0.4


def test_device_is_taken_from_the_caller_not_probed() -> None:
    """`device.py` owns the cuda/cpu decision; this backend must not re-decide it.

    Fails if the segmenter starts calling torch.cuda.is_available() itself.
    """
    model = FakeModel(np.stack([instance(10, 20)]), classes=[PERSON_CLASS])

    YoloSegmenter(device="cpu", model=model).segment(image())

    assert model.calls[0]["device"] == "cpu"


def test_weights_resolve_into_the_cache_not_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ultralytics downloads checkpoints into the CWD given a bare filename."""
    monkeypatch.setenv("FUTSEG_CACHE_DIR", str(tmp_path))

    resolved = YoloSegmenter(device="cpu", model=FakeModel(None)).weights_path

    assert resolved.is_absolute()
    assert tmp_path in resolved.parents


@pytest.mark.slow
def test_real_model_finds_people_in_a_real_photo() -> None:
    """Downloads real weights and runs them against a photo that contains people.

    The fakes above can only prove the code is self-consistent; this proves it
    matches the ultralytics API surface. It asserts people are actually found, so
    the mask/class parsing path is exercised rather than the empty-result
    shortcut. `bus.jpg` ships inside the ultralytics package — no fixture to
    commit and no image licensing question.
    """
    from ultralytics.utils import ASSETS

    from futseg.io import load_image

    photo = load_image(Path(ASSETS) / "bus.jpg")
    alpha = YoloSegmenter(device="cpu", imgsz=640).segment(photo)

    assert alpha.dtype == np.float32
    assert alpha.shape == (photo.height, photo.width)
    assert alpha.min() >= 0.0
    assert alpha.max() <= 1.0
    assert alpha.any(), "expected at least one person in bus.jpg"
    # A handful of pedestrians, not the whole frame.
    assert 0.01 < alpha.mean() < 0.6


def test_pinholes_in_the_mask_are_filled() -> None:
    """Same defect as the refined tier: small enclosed regions are noise (#28)."""
    holed = instance(10, 50).copy()
    holed[20, 20] = 0.0
    model = FakeModel(np.stack([holed]), classes=[PERSON_CLASS])

    alpha = YoloSegmenter(device="cpu", model=model).segment(image())

    assert alpha[20, 20] == 1.0
