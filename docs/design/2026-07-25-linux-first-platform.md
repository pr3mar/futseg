# Linux-first platform strategy

**Date:** 2026-07-25
**Issue:** [#13](https://github.com/pr3mar/futseg/issues/13)
**Status:** Approved

## Problem

The plan targeted whichever platform it happened to be written on, and paid for it in a dependency
workaround that fails silently.

PyPI's default `torch` wheel is **not the same package on every platform**:

| Platform | Wheel | CUDA |
|---|---|---|
| Linux x86_64 | 527 MB | **bundled** |
| Windows x86_64 | 122 MB | **absent** |

Every CUDA dependency torch declares is gated on Linux:

```
nvidia-cudnn-cu13==9.20.0.48;   platform_system == "Linux"
nvidia-cusparselt-cu13==0.8.1;  platform_system == "Linux"
nvidia-nccl-cu13==2.29.7;       platform_system == "Linux"
nvidia-nvshmem-cu13==3.4.5;     platform_system == "Linux"
```

(Verified against the PyPI JSON API for torch 2.13.0, 2026-07-25.)

On Windows, `uv add torch` therefore yields a working but CPU-only PyTorch. The failure is silent:
`uv sync` succeeds, `import torch` works, `ruff` and `pytest` pass. The only symptom is
`torch.cuda.is_available() == False`, discovered several milestones later when the diffusion
backend crawls or fails on `.to("cuda")` — by which point scaffolding has been "green" for weeks
and is the last place anyone looks.

The previously-planned `[[tool.uv.index]]` pin to `download.pytorch.org` existed solely to rescue
Windows. futseg's real deployment targets are developer workstations and servers, which are Linux.

## Decision

**Target Linux. Develop in WSL2. Delete the workaround rather than document it.**

### Platform contract

| Platform | Status |
|---|---|
| Linux x86_64 (glibc) | Supported target. CUDA optional, auto-detected |
| macOS | Developable: segmentation + composite backend. Diffusion runs on CPU, slowly |
| WSL2 | The maintainer's development environment; indistinguishable from Linux to the code |
| Windows native | **Not supported**, and documented as such |

Declaring Windows unsupported is what buys everything else — one dependency path, one device path,
one set of filesystem assumptions, and no untested platform in the matrix.

macOS gets no MPS backend. MPS has real gaps in `diffusers` (dtype quirks, unsupported ops, fp16
issues) and there is no Mac available to test on; shipping an unverifiable claim is worse than
scoping it out. It is a one-line addition to `resolve_device()` if a contributor with a Mac wants
it and can test it.

## Design

### Dependencies

```toml
[project]
requires-python = ">=3.12"
dependencies = [
  "torch", "ultralytics", "diffusers", "transformers", "accelerate",
  "opencv-python-headless", "pillow", "numpy", "typer",
]
```

No `[[tool.uv.index]]`, no `[tool.uv.sources]`, no platform markers. The Linux wheel carries CUDA
from PyPI.

`opencv-python-headless` rather than `opencv-python` is load-bearing on this platform: the GUI build
links `libGL`, which is absent from slim container images and headless servers, producing
`ImportError: libGL.so.1`.

Because the failure mode being designed out is *silent*, scaffolding carries an explicit assertion:

```bash
uv run python -c "import torch; assert torch.cuda.is_available()"
```

A CPU-only resolution must fail the milestone, not pass it quietly. On a machine without a GPU this
check is expected to fail and is not a release gate — it is a gate on *this* development setup.

### Device selection

One decision point, so the policy is testable and backends stay dumb:

```python
# futseg/device.py
def resolve_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    import torch                    # lazy: keeps `futseg --help` responsive
    return "cuda" if torch.cuda.is_available() else "cpu"
```

- CLI exposes `--device auto|cuda|cpu`, default `auto`.
- **Backends receive a resolved device string.** No backend calls `torch.cuda.is_available()`
  itself. Tests substitute a string; no GPU or mocking of torch internals required.
- The `torch` import is deferred so `--help` and argument errors do not pay for it.

### Cache and filesystem

`ultralytics` downloads checkpoints into the **current working directory** on first use. On a
server or in a container that means writing into the workdir — lost on restart at best, a hard
failure on a read-only filesystem at worst. This is the single most container-hostile behaviour in
the current design.

A single resolver owns all writable locations, XDG-compliant so it is idiomatic on Unix and gives a
container exactly one volume to mount:

```
--weights-dir  >  $FUTSEG_CACHE_DIR  >  $XDG_CACHE_HOME/futseg  >  ~/.cache/futseg
```

It configures ultralytics' weights directory and `HF_HOME` from that one value. Rules:

- Nothing is ever written to the package installation directory.
- Nothing is written to the current working directory.
- Outputs go only where `--out` specifies.
- All paths are `pathlib.Path`; no separator or drive-letter assumptions.

### Line endings

```gitattributes
* text=auto eol=lf
*.png binary
*.jpg binary
*.pt binary
```

Editing from Windows tooling while executing on Linux produces CRLF-terminated files, and a CRLF
entrypoint fails inside a container as `bad interpreter: /bin/bash^M` — an error that reads like
nonsense the first time. Every commit during the session that produced this document emitted
`LF will be replaced by CRLF` warnings, so the exposure already exists.

### Development environment

The source tree lives on the Windows filesystem and is reached from WSL2 through `/mnt/c/`, because
it is simultaneously a JetBrains project opened natively on Windows. The **virtualenv does not**:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/futseg"
```

Cross-filesystem access through drvfs is slow enough to matter when torch is roughly 5 GB of small
files — but that cost is almost entirely the environment, not the ~200 KB of tracked source pytest
walks. Redirecting only the environment puts the fast path where it actually pays and leaves the
IDE setup alone.

The split is a correctness requirement as much as a performance one. A single in-tree `.venv/`
shared by a Windows and a Linux interpreter is a collision: `uv sync` from either side overwrites
the other's layout (`Scripts/` vs `bin/`) with no warning and no error. Keeping the Linux
environment outside the work tree removes the possibility rather than documenting the hazard.

Accepted tradeoff: `inotify` does not propagate across drvfs, so a watcher running *inside* WSL2
will not see edits made on the Windows side. The IDE is unaffected — it watches the Windows
filesystem natively. Nothing in the toolchain watches files today; revisit if one enters the stack.

## Deferred

- **Docker image** (milestone 10). Shape intentionally undecided until the CLI exists and its
  mount/configuration needs are known. One constraint is fixed now: **model weights are mounted,
  never baked into the image** — they are multi-GB and would make the image unusable to distribute.
- **CI** (milestone 11, optional). GitHub Actions on `ubuntu-latest`, CPU only: `ruff check` plus
  the fast test suite. `@pytest.mark.slow` tests stay skipped, which the existing test design
  already accommodates. Deferred until the pipeline is built, per the project owner's instruction.

## Consequences

- The milestone-1 dependency landmine is deleted rather than documented.
- Windows contributors are turned away explicitly instead of silently getting a CPU-only install.
- CI can run the full fast suite on a stock `ubuntu-latest` runner with no GPU and no special
  index configuration.
- `device.py` and `paths.py` are new modules in milestone 2, alongside the existing protocols.
- The Docker image inherits a working directory contract it can rely on rather than one it must
  work around.
