"""Image load/save helpers."""

from pathlib import Path

import pytest
from PIL import Image

from futseg.io import load_image, save_image


def test_load_returns_rgb_for_an_rgb_file(tmp_path: Path) -> None:
    path = tmp_path / "photo.png"
    Image.new("RGB", (8, 6), color=(10, 20, 30)).save(path)

    loaded = load_image(path)

    assert loaded.mode == "RGB"
    assert loaded.size == (8, 6)


def test_load_converts_greyscale_to_rgb(tmp_path: Path) -> None:
    """Downstream code indexes three channels; a greyscale input must not
    reach it as a single-channel array."""
    path = tmp_path / "grey.png"
    Image.new("L", (4, 4), color=128).save(path)

    assert load_image(path).mode == "RGB"


def test_load_drops_the_alpha_channel(tmp_path: Path) -> None:
    """A transparent PNG would otherwise carry a fourth channel into the
    segmenter, whose alpha output is the thing we actually derive masks from."""
    path = tmp_path / "transparent.png"
    Image.new("RGBA", (4, 4), color=(255, 0, 0, 0)).save(path)

    assert load_image(path).mode == "RGB"


def test_load_accepts_a_string_path(tmp_path: Path) -> None:
    path = tmp_path / "photo.png"
    Image.new("RGB", (4, 4)).save(path)

    assert load_image(str(path)).size == (4, 4)


def test_load_raises_for_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "absent.png")


def test_save_round_trips_pixels(tmp_path: Path) -> None:
    path = tmp_path / "out.png"
    original = Image.new("RGB", (5, 5), color=(1, 2, 3))

    save_image(original, path)

    assert load_image(path).getpixel((0, 0)) == (1, 2, 3)


def test_save_creates_missing_parent_directories(tmp_path: Path) -> None:
    """`--out results/today/x.png` should work without the user pre-creating it."""
    path = tmp_path / "results" / "today" / "out.png"

    save_image(Image.new("RGB", (2, 2)), path)

    assert path.is_file()
