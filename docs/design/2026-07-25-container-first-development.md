# Container-first development

**Date:** 2026-07-25
**Issue:** [#20](https://github.com/pr3mar/futseg/issues/20)
**Status:** Approved
**Supersedes:** the Linux-first platform strategy ([#13](https://github.com/pr3mar/futseg/issues/13)
/ [#16](https://github.com/pr3mar/futseg/pull/16)) and its follow-up on repository location
([#17](https://github.com/pr3mar/futseg/issues/17) / [#18](https://github.com/pr3mar/futseg/pull/18)).
Those documents are in git history; their surviving decisions are restated below rather than
referenced, so this file stands alone.

## Problem

The superseded design put the source tree on the Windows filesystem, reached from WSL2 through
`/mnt/c/`, and kept the virtualenv off drvfs by exporting `UV_PROJECT_ENVIRONMENT` from
`~/.bashrc`. The environment was therefore a property of *the developer's shell*, which is not a
property anything can rely on. It failed twice in a single session:

1. **The export did not hold.** Ubuntu's `.bashrc` returns at its interactive guard before reaching
   an appended line, so non-interactive shells never saw it — and neither did a terminal opened
   before the line was added. uv does not warn when the variable is missing; it silently chose
   `.venv` in the project root and built **4.1 GB on drvfs**, hardlinking disabled because cache and
   target were on different filesystems.
2. **Two interpreters, one directory.** A `.venv` shared between a Windows and a Linux interpreter
   collides: `Scripts/` versus `bin/`, and whichever ran last owned the tree. The IDE's uv
   integration would have re-triggered this on its own schedule, outside any guard the project
   controls.

Both failures share a shape: correctness depending on host state nobody verifies. A Makefile guard
narrowed the window but could not close it, because the IDE invokes uv directly.

## Decision

**Develop inside a GPU-enabled Ubuntu LTS container. The container is the environment.**

There is no host virtualenv to create, activate, or keep in sync, and no host Python at all. The
only host requirement is a working docker with GPU support.

### Platform contract

| Platform | Status |
|---|---|
| The dev container (Ubuntu LTS) | **The** development and test environment |
| Linux x86_64 (glibc) | Supported runtime target. CUDA optional, auto-detected |
| macOS | Developable outside the container for segmentation and the composite backend |
| Windows native | Not supported, and not developed against |

The host OS becomes uninteresting: it runs docker and an editor. That is the entire point — the
previous contract had to describe filesystems, shells and path translation because the toolchain
lived on the host.

## Design

### Base image: Ubuntu LTS, not `nvidia/cuda`

torch's PyPI wheels bundle the CUDA runtime, and the container runtime injects the host driver at
`--gpus all`. A CUDA base image would ship a second copy of libraries the torch wheel already
carries — gigabytes for nothing. Verified by running `nvidia-smi` inside a stock `ubuntu:26.04`
container with `--gpus all` before committing to the base.

This is the same reasoning that keeps `torch` on plain PyPI with no custom index: the wheel is
already the right artefact on Linux, so nothing needs rescuing.

### Python is pinned through uv, not inherited from the distro

```dockerfile
ENV UV_PYTHON=3.12
RUN uv python install 3.12
```

The interpreter matches what `uv.lock` was resolved and tested against, so the base OS can be
bumped without silently moving Python underneath the lockfile — which would change wheel selection
for every compiled dependency in the graph.

### The virtualenv lives outside the mount

`UV_PROJECT_ENVIRONMENT=/opt/venv`, with the source bind-mounted at `/workspace`.

An in-tree `.venv` inside a bind mount is written straight back to the host, which reintroduces
exactly the collision described above. Keeping it in the image removes the possibility rather than
documenting the hazard.

The image is built in two layers. Dependencies first
(`uv sync --frozen --no-install-project`, ~5.4 GB, invalidated only by `uv.lock`), then the project
itself (`COPY src` + `uv sync --frozen`, seconds). The second step exists so that a plain
`/opt/venv/bin/python` can import `futseg` **without going through `uv run`** — which is what an
IDE's remote interpreter, a debugger or a profiler will do, none of them knowing about uv. Without
it, `import futseg` fails in any fresh container that is not driven by `uv run`.

That install is editable, recording a path to `/workspace/src`, which the bind mount supplies at run
time. The copied source is therefore build-time scaffolding that the mount shadows: **code changes
never require a rebuild**, new modules included. Verified by writing a file on the host and reading
it back from a fresh container, and by importing a module that did not exist when the image was
built.

### Cache and filesystem

`ultralytics` downloads checkpoints into the **current working directory** on first use, which in a
container means the bind mount at best and a read-only layer at worst. One resolver owns every
writable location:

```
--weights-dir  >  $FUTSEG_CACHE_DIR  >  $XDG_CACHE_HOME/futseg  >  ~/.cache/futseg
```

`XDG_CACHE_HOME=/cache` and `HF_HOME=/cache/huggingface` point at a named volume, so multi-GB
downloads survive rebuilds and never enter the image or the work tree.

Setting `XDG_CACHE_HOME` also moves **uv's own wheel cache** to `/cache/uv`, which measured 5.2 GB
after the first runs — a second copy of wheels already installed in the image. That is accepted
rather than split into a second volume: both are caches, both are reconstructible, and one volume
with one `make clean-cache` is easier to reason about than two. Revisit if the duplication starts
costing more than the simplicity buys.

Rules unchanged:

- nothing is written to the package installation directory;
- nothing is written to the current working directory;
- outputs go only where `--out` says;
- all paths are `pathlib.Path`.

### Dependencies

Unchanged from the superseded design, and the reasoning still holds:

- **No custom package index.** `torch` is a plain PyPI dependency; the Linux x86_64 wheel bundles
  CUDA. Reaching for `[[tool.uv.index]]` means solving a Windows problem this project does not have.
- **A CPU-only resolution fails silently** — `uv sync` succeeds, `import torch` works, `ruff` and
  `pytest` pass. `scripts/cuda_check.py` asserts real device work, not just
  `torch.cuda.is_available()`, which returns `True` on a wheel carrying no kernels for the installed
  GPU architecture.
- **`opencv-python` is overridden out of the graph.** `ultralytics` depends on the GUI build
  transitively; both distributions own the same `cv2/` directory, so install order decides which one
  wins and uninstalling either deletes `cv2` from under the other. The GUI build also drags Qt5
  shared objects into the image.

### Line endings

`.gitattributes` forces `* text=auto eol=lf`. A CRLF entrypoint fails inside a container as
`bad interpreter: /bin/bash^M`. This mattered when editing happened on Windows; it still matters
because the working tree may be checked out on a Windows filesystem.

### Entry points

`make` drives the container: `build`, `shell`, `sync`, `lint`, `test`, `check`, `cuda`, `gpu`,
`clean`, `clean-cache`. The environment is a property of the command, which was the goal the
Makefile guards were reaching for and could not reach while the toolchain lived on the host.

## Deleted with the superseded design

- the `/mnt/c` source + `UV_PROJECT_ENVIRONMENT` split, and the `~/.bashrc` export behind it;
- the Makefile's `uname -s` platform guard and stray in-tree `.venv` warning;
- WSL2 setup as the documented development path.

## Open

**Where the working tree lives** — a Windows filesystem reached through `/mnt/c/`, or the Linux
filesystem — is deliberately undecided here. Bind-mount I/O cost is measurable once the image
exists, and the decision is worth more with numbers than without.

## Deferred

- **Runtime/distribution image** (milestone 10, [#14](https://github.com/pr3mar/futseg/issues/14)).
  A different artefact from this one: no source mount, weights mounted rather than baked, minimal
  surface. Not superseded by this document, though its inputs change now that a dev image exists.
- **CI** (milestone 11, [#15](https://github.com/pr3mar/futseg/issues/15)). `ubuntu-latest`, CPU
  only, `ruff check` plus the fast suite with `@pytest.mark.slow` skipped.

## Consequences

- Dev, CI and production converge on one Linux image instead of three approximations of one.
- The class of bug that produced both incidents above — environment depending on unverified host
  state — is removed rather than guarded against.
- The host OS is no longer part of the contract, so the platform table shrinks to runtime targets.
- Onboarding is `make build` plus a GPU driver, rather than a WSL2 setup procedure.
- GPU access now depends on the container runtime being configured for it, which is a new
  precondition the previous design did not have.
