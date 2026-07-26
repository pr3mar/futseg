"""Device resolution policy.

`resolve_device` is the only place in the codebase allowed to ask whether CUDA is
available, so these tests pin both halves of that contract: the answer it gives,
and the fact that an explicit preference never reaches torch at all.
"""

import builtins

import pytest

from futseg.device import resolve_device


@pytest.mark.parametrize("preference", ["cpu", "cuda"])
def test_explicit_preference_is_returned_unchanged(preference: str) -> None:
    assert resolve_device(preference) == preference


def test_explicit_preference_does_not_import_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The torch import is deferred so `futseg --help` does not pay for it.

    Fails if the import moves to module scope or above the preference check.
    """
    real_import = builtins.__import__

    def explode_on_torch(name: str, *args: object, **kwargs: object) -> object:
        if name == "torch":
            raise AssertionError("torch was imported for an explicit device preference")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", explode_on_torch)
    assert resolve_device("cpu") == "cpu"


def test_auto_selects_cuda_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


def test_auto_falls_back_to_cpu_when_cuda_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"


def test_auto_is_the_default() -> None:
    torch = pytest.importorskip("torch")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device() == expected
