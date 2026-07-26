"""The typer CLI: `futseg run` and `futseg segment`.

Backends are swapped out via the module's two factory functions, so no test here
loads a model. What is exercised is everything the CLI itself owns: artefact
names, exit codes, flag plumbing, and the cache override.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from futseg import cli

runner = CliRunner()


def photo_at(path: Path, size: tuple[int, int] = (64, 48)) -> Path:
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 200, (size[1], size[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="RGB").save(path)
    return path


class StubSegmenter:
    def __init__(self, coverage: float = 0.5) -> None:
        self.coverage = coverage

    def segment(self, image: Image.Image) -> np.ndarray:
        alpha = np.zeros((image.height, image.width), dtype=np.float32)
        if self.coverage:
            cut = int(image.width * self.coverage)
            alpha[:, :cut] = 1.0
        return alpha


class StubInpainter:
    def __init__(self) -> None:
        self.calls = 0

    def inpaint(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        self.calls += 1
        return Image.new("RGB", image.size, color=(255, 0, 255))


@pytest.fixture(autouse=True)
def _stub_backends(monkeypatch: pytest.MonkeyPatch):
    """Replace the factories so no test downloads or loads weights."""
    built: dict = {}

    def fake_segmenter(quality, device, confidence=0.25, imgsz=1280, fill_holes=True, **kwargs):
        built.update(
            quality=quality,
            segmenter_device=device,
            confidence=confidence,
            imgsz=imgsz,
            fill_holes=fill_holes,
        )
        return StubSegmenter()

    def fake_inpainter(
        backend, device, prompt, model,
        steps=None, guidance_scale=None, strength=None, **kwargs,
    ):
        built.update(
            backend=backend,
            inpainter_device=device,
            prompt=prompt,
            model=model,
            steps=steps,
            guidance_scale=guidance_scale,
            strength=strength,
        )
        return StubInpainter()

    monkeypatch.setattr(cli, "_build_segmenter", fake_segmenter)
    monkeypatch.setattr(cli, "_build_inpainter", fake_inpainter)
    return built


# --------------------------------------------------------------------------- #
# general
# --------------------------------------------------------------------------- #


def test_help_lists_both_commands() -> None:
    result = runner.invoke(cli.app, ["--help"])

    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "segment" in result.stdout


def test_a_missing_image_is_a_usage_error(tmp_path: Path) -> None:
    """Exit 2, not a traceback: this has to be usable non-interactively."""
    result = runner.invoke(cli.app, ["segment", str(tmp_path / "absent.jpg")])

    assert result.exit_code == 2


# --------------------------------------------------------------------------- #
# segment
# --------------------------------------------------------------------------- #

ARTEFACTS = ("alpha", "inpaint-mask", "composite-alpha", "overlay", "cutout")


def test_segment_writes_every_artefact(tmp_path: Path) -> None:
    image = photo_at(tmp_path / "in" / "shot.jpg")
    out = tmp_path / "out"

    result = runner.invoke(cli.app, ["segment", str(image), "--out", str(out)])

    assert result.exit_code == 0
    for suffix in ARTEFACTS:
        assert (out / f"shot-{suffix}.png").is_file(), suffix


def test_segment_names_artefacts_from_the_input_stem(tmp_path: Path) -> None:
    image = photo_at(tmp_path / "IMG_1234.jpg")
    out = tmp_path / "out"

    runner.invoke(cli.app, ["segment", str(image), "--out", str(out)])

    assert (out / "IMG_1234-alpha.png").is_file()


def test_segment_accepts_several_images(tmp_path: Path) -> None:
    """`futseg segment input/*.jpg` must work — the shell expands the glob."""
    first = photo_at(tmp_path / "one.jpg")
    second = photo_at(tmp_path / "two.jpg")
    out = tmp_path / "out"

    result = runner.invoke(cli.app, ["segment", str(first), str(second), "--out", str(out)])

    assert result.exit_code == 0
    assert (out / "one-alpha.png").is_file()
    assert (out / "two-alpha.png").is_file()


def test_segment_exits_1_when_no_person_is_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty mask is never reported as success."""
    monkeypatch.setattr(
        cli, "_build_segmenter", lambda *a, **k: StubSegmenter(0.0)
    )
    image = photo_at(tmp_path / "empty.jpg")

    result = runner.invoke(cli.app, ["segment", str(image), "--out", str(tmp_path / "out")])

    assert result.exit_code == 1


def test_segment_needs_no_prompt_and_no_inpainter(tmp_path: Path, _stub_backends) -> None:
    """Segmentation must not require diffusion weights or a prompt."""
    runner.invoke(
        cli.app, ["segment", str(photo_at(tmp_path / "s.jpg")), "--out", str(tmp_path / "o")]
    )

    assert "backend" not in _stub_backends


def test_segment_quality_flag_selects_the_tier(tmp_path: Path, _stub_backends) -> None:
    runner.invoke(
        cli.app,
        ["segment", str(photo_at(tmp_path / "s.jpg")), "--out", str(tmp_path / "o"),
         "--quality", "fast"],
    )

    assert _stub_backends["quality"] == "fast"


def test_segment_defaults_to_the_best_tier(tmp_path: Path, _stub_backends) -> None:
    runner.invoke(
        cli.app, ["segment", str(photo_at(tmp_path / "s.jpg")), "--out", str(tmp_path / "o")]
    )

    assert _stub_backends["quality"] == "best"


def test_segment_rejects_mask_offsets_that_would_leave_a_halo(tmp_path: Path) -> None:
    """`derive_masks` guards k > j + feather; the CLI must surface that, not crash."""
    result = runner.invoke(
        cli.app,
        ["segment", str(photo_at(tmp_path / "s.jpg")), "--out", str(tmp_path / "o"),
         "--inpaint-grow", "2", "--composite-shrink", "2", "--feather", "4"],
    )

    assert result.exit_code == 2
    # Errors go to stderr so stdout stays parseable when piped.
    assert "inpaint-grow" in result.stderr


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def test_run_writes_the_output_image(tmp_path: Path) -> None:
    image = photo_at(tmp_path / "shot.jpg")
    out = tmp_path / "result.png"

    result = runner.invoke(
        cli.app, ["run", str(image), "--prompt", "a sunset beach", "--out", str(out)]
    )

    assert result.exit_code == 0, result.stdout
    assert out.is_file()
    assert Image.open(out).size == Image.open(image).size


def test_run_passes_the_prompt_to_the_backend(tmp_path: Path, _stub_backends) -> None:
    runner.invoke(
        cli.app,
        ["run", str(photo_at(tmp_path / "s.jpg")), "--prompt", "a quiet forest",
         "--out", str(tmp_path / "o.png")],
    )

    assert _stub_backends["prompt"] == "a quiet forest"


def test_run_defaults_to_the_diffusion_backend(tmp_path: Path, _stub_backends) -> None:
    runner.invoke(
        cli.app,
        ["run", str(photo_at(tmp_path / "s.jpg")), "--prompt", "p",
         "--out", str(tmp_path / "o.png")],
    )

    assert _stub_backends["backend"] == "diffusion"


def test_run_can_select_the_composite_backend(tmp_path: Path, _stub_backends) -> None:
    runner.invoke(
        cli.app,
        ["run", str(photo_at(tmp_path / "s.jpg")), "--prompt", "p",
         "--out", str(tmp_path / "o.png"), "--backend", "composite"],
    )

    assert _stub_backends["backend"] == "composite"


def test_run_exits_1_when_no_person_is_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cli, "_build_segmenter", lambda *a, **k: StubSegmenter(0.0)
    )

    result = runner.invoke(
        cli.app,
        ["run", str(photo_at(tmp_path / "s.jpg")), "--prompt", "p",
         "--out", str(tmp_path / "o.png")],
    )

    assert result.exit_code == 1


def test_run_requires_a_prompt(tmp_path: Path) -> None:
    result = runner.invoke(cli.app, ["run", str(photo_at(tmp_path / "s.jpg"))])

    assert result.exit_code == 2


# --------------------------------------------------------------------------- #
# shared plumbing
# --------------------------------------------------------------------------- #


def test_device_is_resolved_once_and_shared(tmp_path: Path, _stub_backends) -> None:
    """`resolve_device` is the single decision point; both backends get the same
    resolved string rather than each deciding for itself."""
    runner.invoke(
        cli.app,
        ["run", str(photo_at(tmp_path / "s.jpg")), "--prompt", "p",
         "--out", str(tmp_path / "o.png"), "--device", "cpu"],
    )

    assert _stub_backends["segmenter_device"] == "cpu"
    assert _stub_backends["inpainter_device"] == "cpu"


def test_weights_dir_overrides_the_cache_location(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise weights land wherever the environment happens to point."""
    import os

    monkeypatch.delenv("HF_HOME", raising=False)
    weights = tmp_path / "weights"

    runner.invoke(
        cli.app,
        ["segment", str(photo_at(tmp_path / "s.jpg")), "--out", str(tmp_path / "o"),
         "--weights-dir", str(weights)],
    )

    assert str(weights) in os.environ["HF_HOME"]


def test_run_leaves_sampler_settings_to_the_model_by_default(
    tmp_path: Path, _stub_backends
) -> None:
    """None means "whatever this checkpoint wants" (#33), not a global number."""
    runner.invoke(
        cli.app,
        ["run", str(photo_at(tmp_path / "s.jpg")), "--prompt", "p",
         "--out", str(tmp_path / "o.png")],
    )

    assert _stub_backends["steps"] is None
    assert _stub_backends["guidance_scale"] is None


def test_run_passes_explicit_sampler_overrides_through(tmp_path: Path, _stub_backends) -> None:
    runner.invoke(
        cli.app,
        ["run", str(photo_at(tmp_path / "s.jpg")), "--prompt", "p",
         "--out", str(tmp_path / "o.png"), "--steps", "9", "--guidance-scale", "2.5"],
    )

    assert _stub_backends["steps"] == 9
    assert _stub_backends["guidance_scale"] == 2.5


# --------------------------------------------------------------------------- #
# full parameter surface (#37)
# --------------------------------------------------------------------------- #


def test_run_can_tune_the_seam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The mask offsets decide seam quality, and `run` is what composites the
    finished image — tuning them only on `segment` is backwards.

    Fails if `run` stops forwarding them to the pipeline.
    """
    seen: dict = {}
    real_run = cli.pipeline.run

    def spy(image, **kwargs):
        seen.update(kwargs)
        return real_run(image, **kwargs)

    monkeypatch.setattr(cli.pipeline, "run", spy)

    runner.invoke(
        cli.app,
        ["run", str(photo_at(tmp_path / "s.jpg")), "--prompt", "p",
         "--out", str(tmp_path / "o.png"),
         "--inpaint-grow", "20", "--composite-shrink", "5", "--feather", "8"],
    )

    assert seen["inpaint_grow"] == 20
    assert seen["composite_shrink"] == 5
    assert seen["feather_radius"] == 8


def test_run_rejects_seam_settings_that_would_leave_a_halo(tmp_path: Path) -> None:
    """Same guard `segment` already has; a traceback here would be worse."""
    result = runner.invoke(
        cli.app,
        ["run", str(photo_at(tmp_path / "s.jpg")), "--prompt", "p",
         "--out", str(tmp_path / "o.png"),
         "--inpaint-grow", "2", "--composite-shrink", "2", "--feather", "4"],
    )

    assert result.exit_code == 2
    assert "inpaint-grow" in result.stderr


@pytest.mark.parametrize("command", ["run", "segment"])
def test_segmentation_options_apply_to_both_commands(
    command: str, tmp_path: Path, _stub_backends
) -> None:
    """Detection sensitivity is not a property of which command you ran."""
    args = [command, str(photo_at(tmp_path / "s.jpg"))]
    if command == "run":
        args += ["--prompt", "p", "--out", str(tmp_path / "o.png")]
    else:
        args += ["--out", str(tmp_path / "o")]
    args += ["--confidence", "0.4", "--imgsz", "960", "--no-fill-holes"]

    result = runner.invoke(cli.app, args)

    assert result.exit_code == 0, result.stderr
    assert _stub_backends["confidence"] == 0.4
    assert _stub_backends["imgsz"] == 960
    assert _stub_backends["fill_holes"] is False


def test_run_can_set_diffusion_strength(tmp_path: Path, _stub_backends) -> None:
    runner.invoke(
        cli.app,
        ["run", str(photo_at(tmp_path / "s.jpg")), "--prompt", "p",
         "--out", str(tmp_path / "o.png"), "--strength", "0.8"],
    )

    assert _stub_backends["strength"] == 0.8


def test_hole_filling_is_on_by_default(tmp_path: Path, _stub_backends) -> None:
    runner.invoke(
        cli.app, ["segment", str(photo_at(tmp_path / "s.jpg")), "--out", str(tmp_path / "o")]
    )

    assert _stub_backends["fill_holes"] is True
