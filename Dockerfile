# syntax=docker/dockerfile:1

# Development image for futseg.
#
# The base is plain Ubuntu LTS, deliberately not an nvidia/cuda image: torch's
# PyPI wheels already bundle the CUDA runtime, and the container runtime injects
# the host driver at `--gpus all`. A CUDA base would ship a second copy of
# libraries the torch wheel carries anyway -- gigabytes for nothing. Verified by
# running nvidia-smi in a stock ubuntu container with --gpus all.
ARG UBUNTU_VERSION=26.04
FROM ubuntu:${UBUNTU_VERSION}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      git \
      make \
 && rm -rf /var/lib/apt/lists/*

# uv comes from its own published image rather than a curl-pipe-sh installer:
# pinned to a known version and reproducible across rebuilds.
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /usr/local/bin/

# The interpreter is pinned through uv instead of inherited from the distro, so
# it matches what uv.lock was resolved against and the base OS can be bumped
# without silently changing Python underneath the lockfile.
ENV UV_PYTHON=3.12 \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH

RUN uv python install 3.12

# Weights and HF caches resolve into one mounted volume. Nothing multi-GB is
# baked into the image and nothing is written to the working directory.
ENV XDG_CACHE_HOME=/cache \
    HF_HOME=/cache/huggingface
RUN mkdir -p /cache

WORKDIR /workspace

# Dependency layer, cached independently of source edits.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Then the project itself, so that a plain `/opt/venv/bin/python` can import
# futseg without going through `uv run`. This matters for anything that invokes
# the interpreter directly -- an IDE's remote interpreter, a debugger, a profiler
# -- none of which know about uv.
#
# The install is editable, recording a path to /workspace/src. At run time the
# bind mount supplies that directory, so the source copied here is build-time
# scaffolding that the mount replaces; edits on the host are live with no
# reinstall. Kept as a separate layer so editing source does not invalidate the
# multi-GB dependency layer above.
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

CMD ["bash"]
