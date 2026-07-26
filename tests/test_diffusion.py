"""Generative inpainting backend and its model registry.

The pipeline object is injected, so nothing here downloads weights. The tests
that matter most are the signature checks: they call the *real* diffusers classes'
`__call__` signatures and assert every kwarg an adapter emits is actually accepted.
That is what "verified against the installed diffusers version rather than assumed"
means in #6, and it costs no download.
"""

import inspect

import numpy as np
import pytest
from PIL import Image

from futseg.inpaint.diffusion import (
    DEFAULT_MODEL,
    REGISTRY,
    DiffusionInpainter,
    ModelSpec,
)

WIDTH, HEIGHT = 96, 64


def photo() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 200, (HEIGHT, WIDTH, 3), dtype=np.uint8), mode="RGB")


def mask() -> np.ndarray:
    array = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    array[:, WIDTH // 2 :] = 1.0
    return array


class FakePipeline:
    """Stands in for a diffusers pipeline, recording how it was called."""

    def __init__(self, size: tuple[int, int] | None = None) -> None:
        self.calls: list[dict] = []
        self.moved_to: list[str] = []
        self._size = size

    def to(self, device: str) -> "FakePipeline":
        self.moved_to.append(device)
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        size = self._size or (kwargs.get("width", WIDTH), kwargs.get("height", HEIGHT))
        return type("Result", (), {"images": [Image.new("RGB", size, color=(7, 8, 9))]})()


def inpainter(pipeline: FakePipeline, **kwargs) -> DiffusionInpainter:
    defaults = {"device": "cpu", "prompt": "a sunset beach", "pipeline": pipeline}
    return DiffusionInpainter(**{**defaults, **kwargs})


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #


def test_registry_is_not_empty_and_has_a_default() -> None:
    assert DEFAULT_MODEL in REGISTRY


def test_every_spec_is_complete() -> None:
    for key, spec in REGISTRY.items():
        assert isinstance(spec, ModelSpec), key
        assert spec.repo_id and "/" in spec.repo_id, key
        assert spec.pipeline_cls, key
        assert spec.native_res >= 512, key
        assert spec.license, key
        assert callable(spec.to_kwargs), key


def test_specs_are_frozen() -> None:
    """A mutable registry entry could be edited at runtime by a backend."""
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        REGISTRY[DEFAULT_MODEL].repo_id = "someone/else"


def test_unknown_model_key_is_rejected_with_the_available_ones() -> None:
    with pytest.raises(KeyError, match="unknown model"):
        inpainter(FakePipeline(), model="not-a-model")


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_pipeline_class_exists_in_the_installed_diffusers(key: str) -> None:
    """A registry entry naming a class diffusers does not have is a landmine
    that only detonates when someone selects that model."""
    import diffusers

    assert hasattr(diffusers, REGISTRY[key].pipeline_cls), REGISTRY[key].pipeline_cls


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_adapter_emits_only_kwargs_the_real_pipeline_accepts(key: str) -> None:
    """The load-bearing test for the registry.

    FLUX.2 and SDXL do not share a call signature, so an adapter that emits a
    kwarg the pipeline has never heard of fails only at generation time, after
    the weights have been downloaded and loaded. This catches it in the fast
    suite, against the real class, with no download.
    """
    import diffusers

    spec = REGISTRY[key]
    cls = getattr(diffusers, spec.pipeline_cls)
    accepted = set(inspect.signature(cls.__call__).parameters)

    emitted = spec.to_kwargs(
        image=photo(),
        mask=Image.new("L", (WIDTH, HEIGHT)),
        prompt="a sunset beach",
        guidance_scale=7.0,
        strength=0.99,
        steps=8,
        size=(WIDTH, HEIGHT),
    )

    assert set(emitted) <= accepted, f"{key} emits unknown kwargs: {set(emitted) - accepted}"


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_every_adapter_conditions_on_the_mask(key: str) -> None:
    """A backend that ignores the mask would regenerate the whole canvas."""
    spec = REGISTRY[key]
    emitted = spec.to_kwargs(
        image=photo(),
        mask=Image.new("L", (WIDTH, HEIGHT)),
        prompt="p",
        guidance_scale=7.0,
        strength=0.99,
        steps=8,
        size=(WIDTH, HEIGHT),
    )
    assert "mask_image" in emitted, f"{key} does not pass a mask"


# --------------------------------------------------------------------------- #
# the backend
# --------------------------------------------------------------------------- #


def test_satisfies_the_inpainter_protocol() -> None:
    from futseg.inpaint.base import Inpainter

    assert isinstance(inpainter(FakePipeline()), Inpainter)


def test_inpaint_takes_no_prompt_argument() -> None:
    """Prompt is constructor-injected so composite.py is not handed a parameter
    it would ignore."""
    parameters = list(inspect.signature(DiffusionInpainter.inpaint).parameters)
    assert parameters == ["self", "image", "mask"]


def test_the_constructor_prompt_reaches_the_pipeline() -> None:
    pipeline = FakePipeline()

    inpainter(pipeline, prompt="a quiet forest").inpaint(photo(), mask())

    assert pipeline.calls[0]["prompt"] == "a quiet forest"


def test_the_mask_reaches_the_pipeline_as_an_image() -> None:
    pipeline = FakePipeline()

    inpainter(pipeline).inpaint(photo(), mask())

    given = pipeline.calls[0]["mask_image"]
    assert isinstance(given, Image.Image)
    assert given.mode == "L"


def test_the_canvas_is_downscaled_to_the_model_native_resolution() -> None:
    """Diffusion backends have a fixed native canvas; feeding a 4000px photo
    straight in would either fail or produce a mess."""
    pipeline = FakePipeline()
    spec = REGISTRY[DEFAULT_MODEL]
    big = Image.new("RGB", (spec.native_res * 3, spec.native_res * 2), color=(1, 2, 3))
    big_mask = np.ones((spec.native_res * 2, spec.native_res * 3), dtype=np.float32)

    inpainter(pipeline).inpaint(big, big_mask)

    call = pipeline.calls[0]
    assert max(call["width"], call["height"]) <= spec.native_res


def test_the_result_is_returned_at_the_original_resolution() -> None:
    """The generated background is upscaled back; the pipeline composites the
    full-resolution person on top afterwards."""
    spec = REGISTRY[DEFAULT_MODEL]
    big = Image.new("RGB", (spec.native_res * 2, spec.native_res), color=(1, 2, 3))
    big_mask = np.ones((spec.native_res, spec.native_res * 2), dtype=np.float32)

    result = inpainter(FakePipeline()).inpaint(big, big_mask)

    assert result.size == big.size


def test_a_small_image_is_not_upscaled_to_native_resolution() -> None:
    """Enlarging a small photo to 1024 would invent detail and cost time."""
    pipeline = FakePipeline()

    inpainter(pipeline).inpaint(photo(), mask())

    assert pipeline.calls[0]["width"] <= WIDTH


def test_the_resolved_device_is_used_and_never_probed() -> None:
    pipeline = FakePipeline()

    inpainter(pipeline, device="cuda").inpaint(photo(), mask())

    assert pipeline.moved_to == ["cuda"]


def test_sampler_settings_are_constructor_injected() -> None:
    pipeline = FakePipeline()

    inpainter(pipeline, guidance_scale=3.5, steps=12).inpaint(photo(), mask())

    call = pipeline.calls[0]
    assert call["guidance_scale"] == 3.5
    assert call["num_inference_steps"] == 12


def test_the_fp16_variant_is_requested_on_cuda() -> None:
    """`torch_dtype=float16` alone downloads fp32 and casts it, roughly doubling
    the download. The variant is what actually fetches the small files.

    Fails if `variant` stops being passed to `from_pretrained`.
    """
    import torch

    backend = inpainter(FakePipeline(), device="cuda", model="sdxl-inpaint")

    kwargs = backend._from_pretrained_kwargs()
    assert kwargs["variant"] == "fp16"
    assert kwargs["torch_dtype"] is torch.float16


def test_no_fp16_variant_is_requested_on_cpu() -> None:
    """fp16 weights on CPU are slower than fp32 and some ops are unsupported."""
    backend = inpainter(FakePipeline(), device="cpu", model="sdxl-inpaint")

    assert "variant" not in backend._from_pretrained_kwargs()


def test_a_model_without_an_fp16_variant_does_not_request_one() -> None:
    """Asking for a variant a repo does not publish makes from_pretrained fail."""
    backend = inpainter(FakePipeline(), device="cuda", model="flux2-klein")

    assert REGISTRY["flux2-klein"].variant is None
    assert "variant" not in backend._from_pretrained_kwargs()


def test_gated_models_are_marked_as_such() -> None:
    """A gated entry cannot load without a Hub token; that must be declared
    rather than discovered as a 401 after the user picks it."""
    for key, spec in REGISTRY.items():
        assert isinstance(spec.gated, bool), key
    assert REGISTRY[DEFAULT_MODEL].gated is False, "the default must work without a token"


def test_weights_resolve_through_the_cache_not_the_working_directory(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FUTSEG_CACHE_DIR", str(tmp_path))

    backend = inpainter(FakePipeline())

    assert backend.cache_dir.is_absolute()
    assert tmp_path in backend.cache_dir.parents or backend.cache_dir == tmp_path


# --------------------------------------------------------------------------- #
# per-model sampler defaults (#33)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_every_spec_declares_sampler_defaults(key: str) -> None:
    spec = REGISTRY[key]
    assert spec.steps >= 1, key
    assert spec.guidance_scale >= 0.0, key
    assert 0.0 < spec.strength <= 1.0, key


def test_the_distilled_default_uses_the_step_count_from_its_model_card() -> None:
    """FLUX.2 klein is step-distilled: its card asks for 4 steps and guidance 1.0.

    Running it at a generic 28 steps is seven times the work for nothing, which
    is what shipped before this. Fails if someone restores a global default.
    """
    spec = REGISTRY["flux2-klein"]

    assert spec.steps == 4
    assert spec.guidance_scale == 1.0


def test_non_distilled_models_keep_a_conventional_step_count() -> None:
    """The distilled value must not leak onto models that need real steps."""
    for key in ("sdxl-inpaint", "sd15-inpaint"):
        assert REGISTRY[key].steps >= 20, key


def test_the_backend_takes_its_settings_from_the_selected_spec() -> None:
    pipeline = FakePipeline()

    DiffusionInpainter(device="cpu", prompt="p", model="flux2-klein", pipeline=pipeline).inpaint(
        photo(), mask()
    )

    call = pipeline.calls[0]
    assert call["num_inference_steps"] == 4
    assert call["guidance_scale"] == 1.0


def test_switching_model_switches_the_defaults() -> None:
    """The whole point: one global default cannot be right for both."""
    klein, sdxl = FakePipeline(), FakePipeline()

    DiffusionInpainter(device="cpu", prompt="p", model="flux2-klein", pipeline=klein).inpaint(
        photo(), mask()
    )
    DiffusionInpainter(device="cpu", prompt="p", model="sdxl-inpaint", pipeline=sdxl).inpaint(
        photo(), mask()
    )

    assert klein.calls[0]["num_inference_steps"] != sdxl.calls[0]["num_inference_steps"]


def test_explicit_arguments_still_win() -> None:
    """`--steps` has to remain useful for experimenting."""
    pipeline = FakePipeline()

    DiffusionInpainter(
        device="cpu", prompt="p", model="flux2-klein", steps=12, guidance_scale=3.0,
        pipeline=pipeline,
    ).inpaint(photo(), mask())

    call = pipeline.calls[0]
    assert call["num_inference_steps"] == 12
    assert call["guidance_scale"] == 3.0
