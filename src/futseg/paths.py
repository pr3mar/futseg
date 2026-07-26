"""XDG cache resolver: keeps model weights out of CWD and the install dir.

`ultralytics` downloads checkpoints into the current working directory on first
use, which is lost on restart at best and a hard failure on a read-only
filesystem at worst. Every writable location resolves through here instead, so a
container has exactly one directory to mount.
"""

import os
from pathlib import Path

_APP = "futseg"


def resolve_cache_dir(override: Path | None = None) -> Path:
    """Return the cache directory, creating it if needed.

    Precedence: explicit override, then ``$FUTSEG_CACHE_DIR``, then
    ``$XDG_CACHE_HOME/futseg``, then ``~/.cache/futseg``.
    """
    if override is not None:
        base = Path(override)
    elif env := os.environ.get("FUTSEG_CACHE_DIR"):
        base = Path(env)
    elif xdg := os.environ.get("XDG_CACHE_HOME"):
        base = Path(xdg) / _APP
    else:
        base = Path.home() / ".cache" / _APP

    # absolute(), not resolve(): a relative path here would put weights beside
    # the CWD, but resolving symlinks would rewrite paths the caller passed in.
    base = base.expanduser().absolute()
    base.mkdir(parents=True, exist_ok=True)
    return base


def weights_dir(override: Path | None = None) -> Path:
    """Return the directory model checkpoints are downloaded into."""
    path = resolve_cache_dir(override) / "weights"
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_caches(override: Path | None = None) -> Path:
    """Point third-party caches at the resolved cache directory.

    ``HF_HOME`` is only set when absent: the container already points it at a
    mounted volume, and clobbering that would send multi-GB downloads somewhere
    unmounted.
    """
    cache = resolve_cache_dir(override)
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    return cache
