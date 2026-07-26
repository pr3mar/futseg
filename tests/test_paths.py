"""Cache location policy.

Precedence, per #3:

    --weights-dir  >  $FUTSEG_CACHE_DIR  >  $XDG_CACHE_HOME/futseg  >  ~/.cache/futseg

The point of the resolver is that nothing is ever written to the current working
directory or the install directory, so these tests pin the precedence chain and
the "creates it, and it is absolute" guarantees callers rely on.
"""

from pathlib import Path

import pytest

from futseg.paths import configure_caches, resolve_cache_dir, weights_dir


@pytest.fixture(autouse=True)
def _clear_cache_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("FUTSEG_CACHE_DIR", "XDG_CACHE_HOME", "HF_HOME"):
        monkeypatch.delenv(var, raising=False)


def test_explicit_override_wins_over_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FUTSEG_CACHE_DIR", str(tmp_path / "env"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    assert resolve_cache_dir(tmp_path / "explicit") == tmp_path / "explicit"


def test_futseg_cache_dir_beats_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FUTSEG_CACHE_DIR", str(tmp_path / "env"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    assert resolve_cache_dir() == tmp_path / "env"


def test_xdg_cache_home_is_suffixed_with_futseg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    assert resolve_cache_dir() == tmp_path / "xdg" / "futseg"


def test_falls_back_to_home_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))

    assert resolve_cache_dir() == tmp_path / "home" / ".cache" / "futseg"


def test_directory_is_created(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "cache"

    assert resolve_cache_dir(target).is_dir()


def test_result_is_absolute(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A relative override would put weights next to the CWD, which is the whole
    failure mode this module exists to prevent."""
    monkeypatch.chdir(tmp_path)

    assert resolve_cache_dir(Path("relative-cache")).is_absolute()


def test_weights_dir_lives_under_the_cache_dir(tmp_path: Path) -> None:
    resolved = weights_dir(tmp_path / "cache")

    assert resolved.parent == tmp_path / "cache"
    assert resolved.is_dir()


def test_configure_caches_points_hf_home_at_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    cache = configure_caches(tmp_path / "cache")

    assert cache == tmp_path / "cache"
    assert os.environ["HF_HOME"] == str(tmp_path / "cache" / "huggingface")


def test_configure_caches_does_not_override_an_existing_hf_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The container sets HF_HOME to a mounted volume; clobbering it would send
    multi-GB downloads somewhere unmounted."""
    import os

    monkeypatch.setenv("HF_HOME", "/cache/huggingface")

    configure_caches(tmp_path / "cache")

    assert os.environ["HF_HOME"] == "/cache/huggingface"
