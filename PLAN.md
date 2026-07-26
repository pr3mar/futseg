# futseg: person segmentation + background inpainting — development plan

## Context

`futseg` is scaffolded but has no behaviour yet: dependencies and `uv.lock` are in place and
every module under `src/futseg` is still a docstring. The goal: given an input photo,
automatically segment the person(s)
in it, then apply an inpainting model to regenerate/replace everything **outside** the
person mask (the background) while keeping the person(s) intact — chosen over "remove
person" or "edit region on person" as alternative directions.

Deliverables: (1) a `README.md` for the project, (2) the pipeline below, built as a CLI +
importable library, using local open-source models only.

## Platform

futseg targets **Linux**, because that is where it runs: developer workstations and server
environments. Development happens inside a GPU-enabled Ubuntu LTS container, so the host OS is not
part of the contract. Full rationale in
[`docs/design/2026-07-25-container-first-development.md`](docs/design/2026-07-25-container-first-development.md).

| Platform | Status |
|---|---|
| The dev container (Ubuntu LTS) | **The** development and test environment |
| Linux x86_64 (glibc) | Supported runtime target. CUDA optional, auto-detected |
| macOS | Developable outside the container: segmentation + composite backend. Diffusion on CPU, slowly |
| Windows native | **Not supported**, and not developed against |

This is not incidental. PyPI's `torch` wheel **bundles CUDA on Linux x86_64 (527 MB) but is
CPU-only on Windows (122 MB)** — every `nvidia-*` dependency is gated `platform_system == "Linux"`.
Targeting Linux means plain `torch` from PyPI simply works, with no custom index, no platform
markers, and no silently-CPU-only install to discover five milestones later.

Consequences that shape the design below: device selection is resolved in exactly one place
(`device.py`); nothing is ever written to the current working directory or the install directory
(`paths.py`); `uv` is the package manager throughout.

## Quality bar, and the known gap

`CLAUDE.md` sets the bar at **pixel-perfect** person segmentation. Two things follow, and
the second one is a deliberate, temporary shortfall that must not be papered over:

1. **Raw YOLO11-seg cannot meet that bar and is not the primary segmenter.** Ultralytics
   segmentation emits 32 mask *prototypes* at 1/4 stride — 160×160 at the default
   `imgsz=640` — and each instance mask is a linear combination of those, upsampled and
   bbox-cropped. On a 4000px-wide photo that is 160→640 (4×) then 640→4000 (6.25×), so one
   mask pixel covers roughly **25 original pixels**. Hair, fingers, glasses arms and gaps
   between limbs are destroyed before any post-processing runs. YOLO11-seg is therefore
   demoted to the *fast* tier; the default path refines with SAM2 (see Segmentation).

2. **Binary masks cannot be pixel-perfect on hair — that needs alpha matting, which is
   deferred.** Hair, motion blur and defocused edges are genuinely semi-transparent: those
   pixels are a blend of person and background, and no boolean mask can represent them.
   Gaussian-feathering a binary edge approximates this with uniform geometric blur; it is
   not a per-pixel opacity estimate. **Decision (2026-07-24): ship SAM2-refined binary
   masks first, add matting as a planned upgrade** — see Deferred/future. Until that lands,
   the pipeline is *high-quality* but not literally pixel-perfect at hair boundaries, and
   the README should say so rather than overclaim.

To keep that upgrade a drop-in rather than a breaking migration, the `Segmenter` protocol
returns a **float32 alpha in [0,1]**, not a boolean mask, from day one. Current backends
emit hard 0.0/1.0; the matting stage later fills in the intermediate values with nothing
downstream changing. This is not speculative abstraction — the upgrade is scheduled, and
`masking.py` already has to produce float alpha for the final composite regardless.

## Architecture

```
src/futseg/
  __init__.py
  pipeline.py          # orchestrates segment -> mask derivation -> inpaint -> compose
  cli.py                # typer-based CLI entrypoint (futseg run/segment)
  io.py                 # image load/save helpers
  device.py             # resolve_device(): the single cuda/cpu decision point
  paths.py              # XDG-compliant cache resolver; keeps weights out of CWD
  segmentation/
    base.py             # Segmenter protocol: segment(image) -> HxW float32 alpha in [0,1]
    yolo.py              # YOLO11-seg, person class, instance union      (--quality fast)
    refined.py           # YOLO11 person boxes -> SAM2 refinement (default, --quality best)
  masking.py            # alpha -> (inpaint_mask, composite_alpha); dilate/erode/feather
  inpaint/
    base.py             # Inpainter protocol: inpaint(image, mask) -> full-canvas Image
    diffusion.py         # model registry (FLUX.2 / SDXL / SD2), swappable via --model
    composite.py           # non-generative: solid color / blur / static image swap
tests/
  ...                    # unit tests, model calls mocked; integration tests marked slow
README.md
pyproject.toml
```

Two `Protocol` abstractions (`Segmenter`, `Inpainter`) keep segmentation and inpainting
swappable — e.g. adding the deferred matting segmenter or a new diffusion backend without
touching `pipeline.py` or the CLI.

**Segmentation** — two tiers behind one protocol, selected by `--quality`:

- `best` (**default**): `ultralytics` YOLO11 **detection** on the COCO "person" class
  produces one box per person; each box is used as a **box prompt for SAM2**, and the
  resulting instance masks are unioned. This is the standard high-quality pipeline and it
  dissolves the original objection to SAM ("needs external box/point prompts") — the
  detector *is* the prompt source. Ultralytics ships a SAM2 wrapper, so this adds no second
  heavyweight dependency; the exact API surface is to be confirmed at milestone 4.
- `fast`: YOLO11-seg alone, person class, instance masks unioned. Subject to the ~25px
  quantization above; for quick iteration only. Set `retina_masks=True` and a larger
  `imgsz` to get the most out of it.

**Mask post-processing** — one segmentation produces **two different masks**, and
conflating them is what causes the classic halo artifact. The pixels immediately outside a
silhouette are camera-anti-aliased blends of person and *old* background; if they fall
outside the inpaint region they survive into the output, leaving a rim of the original
background hugging the subject that reads as a cheap cutout against the new scene.

```
alpha  (from Segmenter, float32 [0,1])
  |
  |-- inpaint_mask    = 1 - erode(alpha, k)        # background, grown INTO the person by k
  |-- composite_alpha = feather(erode(alpha, j))   # person, pulled in by j, then softened
```

Same source, opposite offsets, with **k > j + feather_radius** so the generated background
always extends underneath the composite edge and no gap or stale pixel can appear. `k` and
`j` are exposed as tunables; their defaults are a quality decision to be validated on real
photos at milestone 5.

**Inpainting** — the backend is a `diffusers` pipeline chosen from a **model registry**, so
swapping models is a config change rather than a code change:

```python
@dataclass(frozen=True)
class ModelSpec:
    repo_id: str            # e.g. "black-forest-labs/FLUX.2-klein"
    pipeline_cls: str       # diffusers class, imported lazily
    native_res: int         # canvas size the model was trained for
    license: str            # "apache-2.0" | "flux-noncommercial" | "openrail"
    to_kwargs: Callable     # adapts (image, mask, prompt, cfg) -> pipeline kwargs
```

The `to_kwargs` adapter is the load-bearing part: FLUX.2 and SDXL-inpaint do **not** share a
call signature (FLUX.2 editing is instruction- and reference-image-driven via `Flux2Pipeline`,
while SDXL uses a classic mask-conditioned inpaint pipeline), so a registry that only mapped
model IDs to strings would not actually make them interchangeable. Registry entries:

| Model | Res | License | Notes |
|---|---|---|---|
| **FLUX.2 [klein] 4B** (default) | 1024+ | **Apache 2.0** | Unified generate/edit/inpaint in one checkpoint; fast on consumer GPUs; commercially clean |
| FLUX.2 [dev] 32B | 1024+ | FLUX non-commercial | Highest quality; needs 4-bit quantization (`bitsandbytes`) to fit a consumer GPU |
| SDXL-inpaint | 1024 | OpenRAIL | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`; mature, dedicated mask inpainting |
| SD2-inpaint | 512 | permissive | Legacy/comparison only — 512 native is the quality bottleneck |

Default is **FLUX.2 [klein]**: it is the newest generation, permissively licensed (unlike
FLUX.2 [dev] and FLUX.1 Fill, both non-commercial), and handles masked inpainting in the
same weights. Changing the default is a one-line registry edit if a different tradeoff is
wanted. Exact `to_kwargs` shapes per model must be verified against the installed
`diffusers` version at milestone 6 — they are not assumed here.

**Resolution handling.** Diffusion backends have a fixed native canvas, and the input photo
generally does not match it. v1 strategy: downscale the canvas to the model's `native_res`,
inpaint, upscale the *generated background* back to full resolution, then composite the
**full-resolution** person on top. The person therefore never passes through a
resize/generate round-trip, but the background is softer than a native-resolution capture —
an accepted, documented v1 tradeoff. Tiled inpainting and a super-resolution pass on the
background are deferred (see below).

**The invariant**: the pipeline always ends with a feathered composite of the original
person over whichever backend produced the background, so the person is never altered by
the generative model, even at the mask boundary. This holds for every backend.

**Prompt plumbing.** The `Inpainter` protocol is deliberately `inpaint(image, mask)` with no
prompt argument — the prompt and sampler settings are **constructor-injected** into the
diffusion backend (`DiffusionInpainter(model=..., prompt=..., steps=..., guidance=...)`) and
the CLI builds the backend. This keeps `composite.py`, which has no use for a prompt, from
accepting a parameter it would ignore.

## Milestones

1. **Scaffolding** — flesh out `pyproject.toml` (deps below), `src/futseg` package
   layout, `uv sync`-able dev environment, `ruff` for lint, `pytest` for tests,
   `.gitattributes` enforcing LF. Assert CUDA actually resolved rather than assuming it
   (see Dependencies) — the failure being guarded against is silent.
2. **Core abstractions + I/O** — `Segmenter`/`Inpainter` protocols (float alpha; no prompt
   in the inpaint signature), `io.py`, `device.py` (`resolve_device()`), `paths.py` (cache
   resolver), and `masking.py` with the two-mask derivation above as unit-testable pure
   functions (dilate/erode/feather on synthetic masks, no model needed).
3. **Segmentation: fast tier** — `yolo.py` implementing `Segmenter` with YOLO11-seg,
   person-class filtering, multi-instance union, `retina_masks=True`.
4. **Segmentation: refined tier (default)** — `refined.py`: YOLO11 person detection →
   SAM2 box-prompted refinement → union. Becomes the default segmenter. Includes the
   boundary-quality fixtures from Verification, since this is the milestone whose entire
   purpose is edge quality.
5. **Composite backend + end-to-end MVP** — wire `pipeline.py` with the composite
   (non-generative) `Inpainter` first, so segmentation quality, the two-mask derivation and
   pipeline plumbing can be validated on real photos without waiting on/downloading
   diffusion weights. Tune `k`/`j` defaults here.
6. **Generative inpainting backend** — `diffusion.py`: the `ModelSpec` registry, lazy
   pipeline import, device selection (cuda/cpu), the resolution strategy above, and
   verified `to_kwargs` adapters for at least FLUX.2 [klein] and SDXL-inpaint. Wired as the
   default backend.
7. **CLI** — `typer` app: `futseg run <image> --prompt "..." [--backend composite|diffusion]
   [--quality fast|best] [--model <registry-key>] [--device auto|cuda|cpu]
   [--weights-dir <path>] --out out.png`, plus `futseg segment` (mask-only debug output,
   useful for inspecting the two derived masks).
8. **Tests** — unit tests for masking/pipeline wiring with mocked `Segmenter`/`Inpainter`;
   a small number of `@pytest.mark.slow` integration tests that run real models, skipped
   by default.
9. **README** — install, quickstart CLI example, architecture overview (the module list
   above), model/weight download notes, the per-model license table, the platform contract
   and container development setup (`make build`, `make check`), hardware notes (GPU
   recommended for the diffusion backend, composite backend works CPU-only), and an honest
   statement of the hair/matting limitation from "Quality bar" above.
10. **Docker image** — a runtime image for server use. Shape deliberately undecided until
    the CLI exists and its mount/configuration needs are known; the one fixed constraint is
    that **model weights are mounted, never baked in** (multi-GB, would make the image
    undistributable). `paths.py` exists so the container has a single volume to mount.
11. **CI (optional)** — GitHub Actions on `ubuntu-latest`, CPU only: `ruff check` plus the
    fast test suite, with `@pytest.mark.slow` staying skipped. Deliberately last: there is
    no value in wiring CI before there is something for it to run.

Deferred/future (mention in README as roadmap, not built now):

- **Alpha matting for true pixel-perfect edges** — trimap from the existing dilate/erode
  band (`erode(alpha,j)` = definite foreground, `1-erode(alpha,k)` = definite background,
  the band between = unknown) fed to a matting model (e.g. ViTMatte/BiRefNet class) to
  recover per-pixel opacity on hair. This is the upgrade the float-alpha protocol exists to
  absorb, and the one thing standing between this pipeline and the `CLAUDE.md` bar.
- Tiled inpainting and/or a super-resolution pass on the generated background, to remove the
  v1 downscale-upscale softness.
- Batch/video input; packaging as a REST API.

## Dependencies (to be added to `pyproject.toml`)

- `ultralytics` (YOLO11 detection + segmentation, and its SAM2 wrapper)
- `torch` — **plain, from PyPI, with no custom index.** The Linux x86_64 wheel bundles CUDA, so
  targeting Linux removes the need for `[[tool.uv.index]]` / `[tool.uv.sources]` entirely. Because
  a CPU-only resolution would otherwise pass silently, milestone 1 asserts
  `uv run python -c "import torch; assert torch.cuda.is_available()"`.
- `diffusers`, `transformers`, `accelerate` (diffusion inpainting)
- `bitsandbytes` — optional extra, only for the quantized FLUX.2 [dev] registry entry
- `opencv-python-headless` (morphology + Gaussian feather in `masking.py`) — the `-headless` build
  specifically: the GUI build links `libGL`, which is absent from slim container images and
  headless servers (`ImportError: libGL.so.1`).
- `pillow`, `numpy`
- `typer` (CLI)
- dev: `pytest`, `ruff`

## Verification

- Unit tests (`uv run pytest`) for `masking.py` pure functions — including that
  `inpaint_mask` and `composite_alpha` maintain `k > j + feather_radius` coverage with no
  gap — and `pipeline.py` with fake `Segmenter`/`Inpainter` stubs. No model downloads
  required, runs at CI speed.
- **Boundary quality metric**: a small set of hand-annotated fixtures scored with
  **boundary IoU** rather than plain IoU. Plain IoU is dominated by the torso and barely
  moves when the hair is wrong, which makes it useless as a signal for the thing this
  project actually cares about. Without this, "pixel-perfect" is unfalsifiable and there is
  no regression signal when swapping segmentation backends.
- Manual end-to-end check: `uv run futseg run sample.jpg --prompt "sunset beach" --out
  out.png` on a real photo, first with `--backend composite` (fast sanity check of
  segmentation + the two-mask derivation), then with the diffusion backend to visually
  confirm the person is preserved and the background is replaced cleanly — checking
  specifically for a stale-background rim at the silhouette.
- `uv run ruff check .` for lint.
