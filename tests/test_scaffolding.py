"""Smoke tests for the package skeleton.

The modules are still empty, so there is no behaviour to assert yet. These tests
exist so `uv run pytest` reports a real pass rather than "no tests collected"
(exit code 5), and so a missing `__init__.py` or a syntax error in a scaffold
module fails immediately instead of at the milestone that first imports it.
Replaced by behavioural tests as each module is implemented.
"""

import importlib

import pytest

MODULES = [
    "futseg",
    "futseg.cli",
    "futseg.device",
    "futseg.io",
    "futseg.masking",
    "futseg.paths",
    "futseg.pipeline",
    "futseg.inpaint",
    "futseg.inpaint.base",
    "futseg.inpaint.composite",
    "futseg.inpaint.diffusion",
    "futseg.segmentation",
    "futseg.segmentation.base",
    "futseg.segmentation.refined",
    "futseg.segmentation.yolo",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)
