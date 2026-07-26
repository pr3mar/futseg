"""Diffusion backends behind a ModelSpec registry, swappable via --model.

A registry rather than one hardcoded checkpoint, so changing model is config
rather than code. The `to_kwargs` adapter is the load-bearing part: the pipelines
do not share a call signature, so mapping model ids to strings alone would not
actually make them interchangeable.

Repo ids and pipeline classes below were verified against the installed
`diffusers` and against the Hub rather than taken from the plan — two of the
plan's entries were wrong. See `docs/wiki.md`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from futseg.masking import Alpha, alpha_to_mask
from futseg.paths import configure_caches


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to drive one diffusion model as an inpainter."""

    repo_id: str
    #: `diffusers` class name, imported lazily so `--help` costs nothing.
    pipeline_cls: str
    #: Canvas size the model was trained for.
    native_res: int
    license: str
    #: Adapts our arguments to this pipeline's call signature.
    to_kwargs: Callable[..., dict]


def _mask_conditioned_kwargs(
    *,
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    guidance_scale: float,
    strength: float,
    steps: int,
    size: tuple[int, int],
) -> dict:
    """Kwargs shared by the classic mask-conditioned inpaint pipelines.

    `Flux2KleinInpaintPipeline` and `StableDiffusionXLInpaintPipeline` turned out
    to agree on this core, where the plan expected them to diverge. They still
    differ in what *else* they accept — FLUX.2 has `image_reference`, SDXL has
    `negative_prompt` and friends — which is why the adapter stays per-model.
    """
    return {
        "prompt": prompt,
        "image": image,
        "mask_image": mask,
        "width": size[0],
        "height": size[1],
        "guidance_scale": guidance_scale,
        "strength": strength,
        "num_inference_steps": steps,
    }


def _sdxl_kwargs(**kwargs) -> dict:
    """SDXL additionally accepts a negative prompt, which keeps people out."""
    adapted = _mask_conditioned_kwargs(**kwargs)
    adapted["negative_prompt"] = "person, people, blurry, distorted"
    return adapted


#: Model key -> spec. `--model` selects by key.
REGISTRY: dict[str, ModelSpec] = {
    "flux2-klein": ModelSpec(
        repo_id="black-forest-labs/FLUX.2-klein-4B",
        pipeline_cls="Flux2KleinInpaintPipeline",
        native_res=1024,
        license="apache-2.0",
        to_kwargs=_mask_conditioned_kwargs,
    ),
    "sdxl-inpaint": ModelSpec(
        repo_id="diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        pipeline_cls="StableDiffusionXLInpaintPipeline",
        native_res=1024,
        license="openrail++",
        to_kwargs=_sdxl_kwargs,
    ),
    "sd15-inpaint": ModelSpec(
        repo_id="stable-diffusion-v1-5/stable-diffusion-inpainting",
        pipeline_cls="StableDiffusionInpaintPipeline",
        native_res=512,
        license="creativeml-openrail-m",
        to_kwargs=_mask_conditioned_kwargs,
    ),
}

#: Apache-2.0 and ungated, so it works without a Hub token.
DEFAULT_MODEL = "flux2-klein"


def _fit_within(size: tuple[int, int], limit: int) -> tuple[int, int]:
    """Scale `size` down to fit `limit`, rounded down to a multiple of 16.

    Never scales *up*: enlarging a small photo to the native canvas invents
    detail and spends generation time for nothing.
    """
    width, height = size
    scale = min(1.0, limit / max(width, height))
    fitted = (int(width * scale), int(height * scale))
    return (max(16, fitted[0] // 16 * 16), max(16, fitted[1] // 16 * 16))


class DiffusionInpainter:
    """Generative background replacement through a `diffusers` pipeline.

    Implements the `Inpainter` protocol. Prompt and sampler settings are injected
    here rather than passed to `inpaint`, so the non-generative backends are not
    forced to accept arguments they would ignore.
    """

    def __init__(
        self,
        *,
        device: str,
        prompt: str,
        model: str = DEFAULT_MODEL,
        guidance_scale: float = 7.0,
        strength: float = 0.99,
        steps: int = 28,
        pipeline: object | None = None,
    ) -> None:
        if model not in REGISTRY:
            raise KeyError(f"unknown model {model!r}; available: {sorted(REGISTRY)}")
        self.spec = REGISTRY[model]
        self.device = device
        self.prompt = prompt
        self.guidance_scale = guidance_scale
        self.strength = strength
        self.steps = steps
        # Point the HF caches at the futseg cache dir before anything downloads.
        self.cache_dir: Path = configure_caches()
        self._pipeline = pipeline

    def _load(self) -> object:
        if self._pipeline is None:
            import diffusers
            import torch

            cls = getattr(diffusers, self.spec.pipeline_cls)
            dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
            self._pipeline = cls.from_pretrained(self.spec.repo_id, torch_dtype=dtype)
        return self._pipeline

    def inpaint(self, image: Image.Image, mask: Alpha) -> Image.Image:
        """Regenerate the masked region, returning a full canvas at input size.

        The canvas is downscaled to the model's native resolution, generated, then
        scaled back up. The *subject* never passes through that round trip:
        `pipeline.py` composites the full-resolution person on top afterwards, so
        only the background carries the softness.
        """
        original_size = image.size
        target = _fit_within(original_size, self.spec.native_res)

        small_image = image.convert("RGB").resize(target, Image.LANCZOS)
        small_mask = alpha_to_mask(np.asarray(mask)).resize(target, Image.NEAREST)

        pipeline = self._load().to(self.device)
        kwargs = self.spec.to_kwargs(
            image=small_image,
            mask=small_mask,
            prompt=self.prompt,
            guidance_scale=self.guidance_scale,
            strength=self.strength,
            steps=self.steps,
            size=target,
        )
        generated = pipeline(**kwargs).images[0]

        if generated.size != original_size:
            generated = generated.resize(original_size, Image.LANCZOS)
        return generated.convert("RGB")
