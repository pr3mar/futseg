"""Image load/save helpers."""

from pathlib import Path

from PIL import Image


def load_image(path: Path | str) -> Image.Image:
    """Load an image as RGB.

    Always RGB: a greyscale or transparent input would otherwise reach the
    segmenter with the wrong channel count, and alpha in the *input* is unrelated
    to the alpha this pipeline derives.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such image: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")


def save_image(image: Image.Image, path: Path | str) -> None:
    """Write an image, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
