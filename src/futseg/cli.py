"""Typer application exposing `futseg run` and `futseg segment`.

Both commands must work non-interactively on a server or in a container: no
prompts, and exit codes that mean something — `0` success, `1` no person found,
`2` usage error. An empty mask is never reported as success.

This module is where the device is resolved (once, via `resolve_device`) and
where backends are constructed with their prompt and sampler settings injected.
Nothing downstream re-decides either.
"""

from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from PIL import Image

from futseg import pipeline
from futseg.device import resolve_device
from futseg.io import load_image, save_image
from futseg.masking import Alpha, alpha_to_mask, derive_masks
from futseg.paths import configure_caches

app = typer.Typer(
    help="Person segmentation and prompt-driven background replacement.",
    no_args_is_help=True,
    add_completion=False,
)

QUALITIES = ("fast", "best")
BACKENDS = ("composite", "diffusion")


def _build_segmenter(quality: str, device: str):
    """Construct a segmenter. Imported lazily so `--help` loads no models."""
    if quality == "fast":
        from futseg.segmentation.yolo import YoloSegmenter

        return YoloSegmenter(device=device)
    from futseg.segmentation.refined import RefinedSegmenter

    return RefinedSegmenter(device=device)


def _build_inpainter(backend: str, device: str, prompt: str, model: str):
    """Construct an inpainter with prompt and sampler settings injected here.

    The `Inpainter` protocol takes no prompt, so this is the only place it can
    be supplied — which is what keeps the composite backends free of an argument
    they would ignore.
    """
    if backend == "composite":
        from futseg.inpaint.composite import BlurInpainter

        return BlurInpainter()
    from futseg.inpaint.diffusion import DiffusionInpainter

    return DiffusionInpainter(device=device, prompt=prompt, model=model)


def _overlay(photo: Image.Image, alpha: Alpha, strength: float = 0.45) -> Image.Image:
    """Tint everything outside the mask.

    The only view in which the silhouette is judged against the photograph rather
    than against black, which is where hair and fingers expose a coarse mask.
    """
    pixels = np.asarray(photo.convert("RGB"), dtype=np.float32)
    tinted = pixels * (1.0 - strength) + np.array([255.0, 0.0, 0.0]) * strength
    weights = np.asarray(alpha, dtype=np.float32)[..., None]
    blended = pixels * weights + tinted * (1.0 - weights)
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8), mode="RGB")


def _cutout(photo: Image.Image, alpha: Alpha) -> Image.Image:
    """The subject on transparency, using the feathered composite alpha."""
    rgba = photo.convert("RGBA")
    rgba.putalpha(alpha_to_mask(np.asarray(alpha)))
    return rgba


def _fail(message: str, code: int) -> typer.Exit:
    typer.secho(message, fg=typer.colors.RED, err=True)
    return typer.Exit(code)


@app.command()
def segment(
    images: Annotated[
        list[Path], typer.Argument(exists=True, dir_okay=False, help="photos to segment")
    ],
    out: Annotated[Path, typer.Option(help="directory for artefacts")] = Path("out"),
    quality: Annotated[str, typer.Option(help="fast|best")] = "best",
    device: Annotated[str, typer.Option(help="auto|cuda|cpu")] = "auto",
    weights_dir: Annotated[Path | None, typer.Option(help="override the cache location")] = None,
    inpaint_grow: Annotated[
        int, typer.Option(help="k: grow the inpaint region into the subject")
    ] = pipeline.DEFAULT_INPAINT_GROW,
    composite_shrink: Annotated[
        int, typer.Option(help="j: pull the composited subject in")
    ] = pipeline.DEFAULT_COMPOSITE_SHRINK,
    feather: Annotated[
        int, typer.Option(help="soften the composited edge")
    ] = pipeline.DEFAULT_FEATHER_RADIUS,
) -> None:
    """Segment photos and write the masks, without inpainting anything.

    Needs no diffusion weights and no prompt: this is how mask edge quality is
    judged, and it is useful on its own.
    """
    if quality not in QUALITIES:
        raise _fail(f"--quality must be one of {'|'.join(QUALITIES)}", 2)
    if inpaint_grow <= composite_shrink + feather:
        raise _fail(
            "--inpaint-grow must exceed --composite-shrink + --feather "
            f"({inpaint_grow} <= {composite_shrink} + {feather}); "
            "otherwise a rim of stale background survives at the silhouette",
            2,
        )

    configure_caches(weights_dir)
    resolved = resolve_device(device)
    segmenter = _build_segmenter(quality, resolved)

    found_any = False
    for path in images:
        photo = load_image(path)
        alpha = segmenter.segment(photo)
        if not alpha.any():
            typer.secho(f"{path.name}: no person found", fg=typer.colors.YELLOW, err=True)
            continue
        found_any = True

        masks = derive_masks(
            alpha,
            inpaint_grow=inpaint_grow,
            composite_shrink=composite_shrink,
            feather_radius=feather,
        )
        stem = path.stem
        for name, artefact in (
            ("alpha", alpha_to_mask(alpha)),
            ("inpaint-mask", alpha_to_mask(masks.inpaint_mask)),
            ("composite-alpha", alpha_to_mask(masks.composite_alpha)),
            ("overlay", _overlay(photo, alpha)),
            ("cutout", _cutout(photo, masks.composite_alpha)),
        ):
            save_image(artefact, out / f"{stem}-{name}.png")
        typer.echo(f"{path.name}: subject {alpha.mean():.1%} -> {out}/{stem}-*.png")

    if not found_any:
        raise typer.Exit(1)


@app.command()
def run(
    image: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="photo to process")],
    prompt: Annotated[str, typer.Option(help="what the new background should be")],
    out: Annotated[Path, typer.Option(help="output image")] = Path("out/result.png"),
    backend: Annotated[str, typer.Option(help="composite|diffusion")] = "diffusion",
    quality: Annotated[str, typer.Option(help="fast|best")] = "best",
    model: Annotated[str, typer.Option(help="diffusion registry key")] = "flux2-klein",
    device: Annotated[str, typer.Option(help="auto|cuda|cpu")] = "auto",
    weights_dir: Annotated[Path | None, typer.Option(help="override the cache location")] = None,
) -> None:
    """Replace the background of a photo, keeping the subject pixel-identical."""
    if quality not in QUALITIES:
        raise _fail(f"--quality must be one of {'|'.join(QUALITIES)}", 2)
    if backend not in BACKENDS:
        raise _fail(f"--backend must be one of {'|'.join(BACKENDS)}", 2)

    configure_caches(weights_dir)
    resolved = resolve_device(device)

    photo = load_image(image)
    segmenter = _build_segmenter(quality, resolved)
    if not segmenter.segment(photo).any():
        typer.secho(f"{image.name}: no person found", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    inpainter = _build_inpainter(backend, resolved, prompt, model)
    result = pipeline.run(photo, segmenter=segmenter, inpainter=inpainter)
    save_image(result, out)
    typer.echo(f"{image.name} -> {out}")


if __name__ == "__main__":  # pragma: no cover
    app()
