# futseg project wiki

Cumulative knowledge base: decisions, conventions, and gotchas. Update whenever something new is
learned, and always review before opening a PR (see `github-issue-workflow` skill). Unlike
`transcript.md` (chronological, append-only), this file is organized by topic and should be edited
in place as understanding changes — keep it consistent, not just additive.

## Conventions

- **Development happens inside the container.** `make build` once, then `make check` / `shell` /
  `cuda`; `make help` lists everything. There is no host virtualenv, no host Python, and nothing to
  activate; the only host requirement is docker with GPU support. See
  `docs/design/2026-07-25-container-first-development.md`.
- **`make up` is optional.** Task targets `exec` into a resident container when one is running and
  fall back to a one-off `run --rm` otherwise, so `make test` behaves the same either way. Keep one
  resident (`make up`) when you want a shell and an IDE sharing the same container.
- Package management: `uv` only (`uv sync`, `uv run ...`). No `pip install` / `poetry` / raw `venv`.
- **The virtualenv lives at `/opt/venv`, outside the bind-mounted source.** An in-tree `.venv`
  inside a bind mount is written back to the host, where it collides with anything the host built
  there. Never point `UV_PROJECT_ENVIRONMENT` into `/workspace`.
- **The image installs the project editable, so `/opt/venv/bin/python` works without `uv run`.**
  Anything driving the interpreter directly — an IDE interpreter, a debugger, a profiler — knows
  nothing about uv, and `import futseg` fails without this. Because the install is editable against
  `/workspace/src`, **editing code never requires a rebuild**, new modules included; only a change
  to `uv.lock` invalidates the dependency layer.
- **Console scripts are the exception: `[project.scripts]` changes DO need a rebuild.** Entry points
  are generated into `/opt/venv/bin` at *install* time from `pyproject.toml` metadata, so an
  editable install does not pick up a newly added one. Symptom: `futseg: command not found` in a
  container built before the entry point existed, even though `import futseg` works fine. Either
  `make build`, or `uv sync` inside the running container to regenerate it.
- Packaging: `hatchling` with a `src/` layout (`src/futseg`). Boring on purpose — any PEP 517
  frontend builds it, and `src/` makes tests import the installed package rather than the working
  tree.
- `ruff` and `pytest` are configured in `pyproject.toml`, not in separate files. `ruff`: py312,
  line-length 100, rules `E,F,I,UP,B`. `pytest`: `testpaths = ["tests"]`, `slow` marker registered.
- **Linux is the target platform.** Windows native is unsupported and not developed against; the dev
  container makes dev == CI == prod one image rather than three approximations. macOS is developable
  outside the container for segmentation and the composite backend only.
- **The base image is plain Ubuntu LTS, not `nvidia/cuda`.** torch's wheels bundle the CUDA runtime
  and the container runtime injects the host driver, so a CUDA base image would ship a second copy
  of libraries already inside the torch wheel. Python is pinned via uv (`UV_PYTHON=3.12`) rather
  than inherited from the distro, so bumping the base OS cannot silently move the interpreter out
  from under `uv.lock`.
- Device selection happens **only** in `device.py` (`resolve_device()`). Backends receive a resolved
  `"cuda"` / `"cpu"` string and never probe for CUDA themselves.
- Writable locations resolve **only** through `paths.py`. Nothing is written to the current working
  directory or the package install directory; outputs go where `--out` says.
- Two `Protocol` abstractions, `Segmenter` and `Inpainter` (see `PLAN.md`), keep segmentation and
  inpainting backends swappable. New backends implement one of these protocols rather than being
  special-cased into `pipeline.py`.
- `Segmenter.segment()` returns a **float32 alpha in [0,1]**, not a boolean mask. Both current
  backends only emit hard 0.0/1.0; the intermediate values arrive with the deferred matting stage.
  Don't "simplify" the protocol back to `bool`.
- `Inpainter.inpaint(image, mask)` takes **no prompt argument**. Prompt and sampler settings are
  constructor-injected into the diffusion backend, so `composite.py` isn't forced to accept a
  parameter it ignores.
- Portability: no hardcoded local paths or machine-specific assumptions anywhere that gets
  committed. A fresh `uv sync` must work on any machine.
- **Photographs go in `input/`, artefacts in `out/`.** Both are gitignored; `input/` is tracked as a
  directory via `.gitkeep` so it exists in a fresh clone. This repo is public, so photographs of
  real people must never enter it. Neither path is hardcoded — `--out` overrides the destination and
  any path is accepted as input.
- **Boundary-quality fixtures live in a gitignored `fixtures/`, local-only.** A gitignored fixture
  is visible to exactly one machine, so a metric reading one is a local tool, not a regression gate
  — committing it is what would make it CI-checkable. That is deliberate for now: there is no CI
  (#15), and scoring a segmenter against a ground-truth mask that a segmenter produced is circular.
  Note the image itself never needs committing — `bus.jpg` ships inside the installed `ultralytics`
  package. Only a hand-annotated mask would, and a binary silhouette is not a photograph, so the
  privacy rule is not the blocker here; the absence of a real annotation is.
- **`futseg segment` is a first-class command, not a debug flag.** Segmenting without inpainting is
  how mask edge quality gets judged, needs no diffusion weights or prompt, and is useful on its own.
- **The CLI is the only entry point; don't add scripts that duplicate it.** `make exec
  CMD="futseg segment input/x.jpg"` runs it in the container, so no `make` target needs to wrap it
  either. A second argument parser means a second set of defaults that drifts.
- **Exit codes are part of the contract**: `0` success, `1` no person found, `2` usage error.
  Errors go to stderr so stdout stays parseable when piped. An empty mask is never success.
- **Backends are constructed in `cli.py` and nowhere else** (`_build_segmenter`,
  `_build_inpainter`), imported lazily so `--help` loads no models. Tests monkeypatch those two
  functions, which is why the CLI suite needs no weights.

## Decisions

- **Default segmenter is YOLO11 *detection* → SAM2 box-prompted refinement** (`--quality best`),
  person class, instance masks unioned. Raw YOLO11-seg is demoted to `--quality fast`. The earlier
  objection to SAM — "needs external box/point prompts" — was self-defeating, since the detector is
  the prompt source. MediaPipe Selfie Segmentation remains rejected (single-portrait only).
- **One segmentation produces two derived masks**: `inpaint_mask = 1 - erode(alpha, k)` and
  `composite_alpha = feather(erode(alpha, j))`, with `k > j + feather_radius`. Same source,
  opposite offsets. Conflating them leaves a rim of stale original background at the silhouette.
- **Inpainting backend is a `ModelSpec` registry** (`inpaint/diffusion.py`), not a single
  hardcoded checkpoint, so models are swappable via `--model`. Verified against the installed
  `diffusers` at #6, which corrected two guesses: `Flux2KleinInpaintPipeline` *is* mask-conditioned
  and shares the core call signature with `StableDiffusionXLInpaintPipeline`
  (`prompt`/`image`/`mask_image`/`strength`/`guidance_scale`). The adapter is still per-model
  because the *extras* differ — FLUX.2 has `image_reference`, SDXL has `negative_prompt` — but the
  gap is narrower than the plan assumed.
- **Use `Flux2KleinInpaintPipeline`, never `Flux2Pipeline`, for inpainting.** `Flux2Pipeline` has no
  `mask_image` parameter at all: it is instruction-driven whole-canvas editing. Selecting it would
  regenerate the entire frame and silently ignore the mask.
- **Default model is FLUX.2 [klein] 4B**, repo `black-forest-labs/FLUX.2-klein-4B` — Apache 2.0 and
  **ungated**, so it needs no Hub token. Note the repo id: plain `FLUX.2-klein` does not exist, and
  `FLUX.2-klein-9B` is gated and non-commercial despite the sibling naming. FLUX.2 [dev] (~165 GB)
  and FLUX.1 Fill are gated and non-commercial. SDXL-inpaint (ungated, openrail++) is the
  alternative; **SD2-inpaint was dropped — `stabilityai/stable-diffusion-2-inpainting` no longer
  exists** — replaced by `stable-diffusion-v1-5/stable-diffusion-inpainting` for comparison only.
  Per-model licenses must be listed in the README.
- **Registry entries are verified in the fast suite, without downloading weights.** One test asserts
  each `pipeline_cls` exists in the installed `diffusers`; another asserts every kwarg an adapter
  emits appears in that class's real `__call__` signature. Both failure modes otherwise surface only
  after multi-GB weights have downloaded and loaded.
- **Resolution strategy (v1)**: downscale the canvas to the model's native resolution, inpaint,
  upscale the *generated background*, then composite the **full-resolution** person on top. The
  person never passes through a resize/generate round-trip; the softer background is an accepted,
  documented v1 tradeoff. Tiling / super-resolution are deferred.
- The pipeline always does a final feathered composite of the original person over the backend
  output, so the person is never altered by the generative model, even at the mask boundary.
- **Alpha matting is deferred, and the pixel-perfect bar is therefore not yet met.** Binary masks
  cannot represent semi-transparent hair. The trimap for a future matting stage comes free from the
  existing dilate/erode band. Until it lands, the README must say so rather than overclaim — see
  `PLAN.md` "Quality bar, and the known gap".
- **GitHub milestones 1–11 match `PLAN.md` one-to-one, but the issue numbering does not run in
  order.** Later additions hold higher issue numbers while sitting earlier in the sequence:
  milestones 1–11 map to issues #2, #3, #4, **#10**, #5, #6, #7, #8, #9, #14, #15 respectively.
  `#1` (roadmap), `#13` (the Linux-first platform decision) and `#20` (the dev container that
  superseded it) carry no milestone. Don't assume issue N corresponds to milestone N−1. Milestone 10
  (#14) remains the *runtime/distribution* image — a different artefact from the dev container.

## Gotchas

- Repo is public: no hardware specs or other unnecessary private details in issues/PRs/commits/docs.
- Never commit or disclose secrets, API keys, tokens, or credentials in plain text (global rule,
  not futseg-specific — see `~/.claude/CLAUDE.md`).
- A `PreToolUse` anonymization hook (`~/.claude/hooks/anonymization-guard.sh`, user-level) blocks
  `git commit`/`git push` when added lines match hardware-spec, credential, or local-home-path
  patterns. It scans the staged diff on commit and everything unpushed on push. If it fires, it
  names the rule and file but deliberately withholds the matched value.
- **`ultralytics` downloads checkpoints into the current working directory** on first run, so
  `yolo11*.pt` will appear at the repo root uninvited. Already covered by `.gitignore`.
- **Raw YOLO11-seg masks are much coarser than they look.** Ultralytics emits 32 mask prototypes at
  1/4 stride (160×160 at `imgsz=640`), so on a 4000px-wide photo one mask pixel spans ~25 source
  pixels. Use `retina_masks=True` and a larger `imgsz` for the fast tier; this is the whole reason
  the default tier refines with SAM2.
- **`torch` needs no custom index on Linux — and adding one is a smell.** The Linux x86_64 wheel
  bundles CUDA (527 MB; the Windows wheel is 122 MB and CPU-only, with every `nvidia-*` dependency
  gated `platform_system == "Linux"`). An earlier revision pinned `download.pytorch.org` purely to
  rescue Windows. Because a CPU-only resolution fails *silently* — `uv sync` succeeds, `import
  torch` works, tests pass — milestone 1 asserts CUDA explicitly.
- **`torch.cuda.is_available()` alone is not proof CUDA works.** It returns `True` on a wheel that
  carries no kernels for the installed GPU architecture; that failure surfaces only when a real op
  runs, which is the same silent-until-late shape the index rule above guards against. Run
  `make cuda`: it prints the wheel's compiled arch list against the device's compute capability and
  then does real work (fp32 matmul checked against CPU, fp16 matmul, cuDNN conv). Exit code 0 means
  usable. Diagnostic only — it needs a GPU, so it is not in the test suite.
- **Summing an fp16 tensor can report `inf` and look like a GPU fault.** `256**3 = 16777216`
  overflows fp16's ~65504 ceiling, so half-precision accumulation saturates. Cast to fp32 before
  reducing. Cost an unnecessary debugging detour while writing `scripts/cuda_check.py`.
- **`ultralytics` writes checkpoints into the current working directory**, which is actively hostile
  in a container (read-only or ephemeral workdir). `paths.py` redirects it and `HF_HOME` to one
  XDG-compliant cache dir, which also gives Docker a single volume to mount.
- **Use `opencv-python-headless`, never `opencv-python`.** The GUI build links `libGL`, absent from
  slim images and headless servers → `ImportError: libGL.so.1`. **Declaring the headless build is
  not enough**: `ultralytics` depends on `opencv-python` transitively, and both distributions own
  the same `cv2/` directory, so the second one installed overwrites the first's `RECORD` — which of
  them you get is install-order dependent, and uninstalling either deletes the shared `cv2/` out
  from under the other. `pyproject.toml` neutralises the transitive requirement with
  `[tool.uv] override-dependencies = ["opencv-python; sys_platform == 'never'"]`. If `cv2` ever
  goes missing after a dependency change, recreate the venv rather than reinstalling on top.
- **The `/cache` volume holds more than model weights.** `XDG_CACHE_HOME=/cache` also relocates uv's
  wheel cache to `/cache/uv` (~5 GB), so the volume carries weights, HF downloads and wheels
  together. All reconstructible; `make clean-cache` deletes all of it and forces a re-download.
- **`docker images` overstates the image size.** It reported 17.8 GB where `docker image inspect`
  reports 5.71 GB, because buildkit's attestation/manifest-list entries get counted too. Trust
  `docker image inspect --format '{{.Size}}'`.
- **A `.venv` appearing in the project root means something bypassed the container.** uv does not
  warn when it cannot tell where the environment belongs; it just picks `.venv` in the working
  directory. Inside a bind mount that lands on the host. Delete it and run the `make` target rather
  than a bare `uv` on the host — this cost 4 GB once already.
- **`.gitattributes` forces LF.** A CRLF entrypoint fails inside a container as
  `bad interpreter: /bin/bash^M`, and the working tree may be checked out on a Windows filesystem.
- **`derive_masks` raises rather than producing a haloed image.** `inpaint_grow` must exceed
  `composite_shrink + feather_radius`; at equality the softened composite edge reaches exactly as
  far as the regenerated region and original pixels show through the partially transparent border.
  The check is in `masking.py`, not in the caller, so no backend can get it wrong quietly.
- **`union` takes a per-pixel maximum, never a sum.** Overlapping instances in a multi-person photo
  would otherwise exceed 1.0 and clip into a hard-edged blob.
- **`paths.configure_caches` sets `HF_HOME` with `setdefault`.** The container already points it at
  the mounted `/cache` volume; overwriting it would send multi-GB downloads somewhere unmounted.
- **`resolve_cache_dir` uses `Path.absolute()`, not `.resolve()`.** Absolute is required so a
  relative override cannot drop weights beside the CWD, but resolving symlinks would silently
  rewrite a path the caller passed in.
- **`YoloSegmenter` filters the person class in our code, not via the `classes=` predict kwarg.**
  The kwarg would work, but the guarantee then lives in a library keyword whose behaviour could
  change unnoticed; filtering on `boxes.cls` is what the tests can actually exercise.
- **Its `imgsz` defaults to 1280, above the ultralytics default of 640.** The prototype stride is
  what limits this tier (`PLAN.md`), so the larger input is the only lever that helps.
- **Slow tests use `ultralytics.utils.ASSETS / "bus.jpg"`.** It ships inside the installed package
  and contains people, so real-model tests need no committed image fixture and raise no licensing
  or privacy question for a public repo.
- **SAM2's `result.boxes.cls` is a prompt ordinal, not a COCO class.** It comes back as
  `[0., 1., 2., ...]` — one per box you prompted with. Filtering it against `PERSON_CLASS`, which is
  the correct pattern on *detection* output in `yolo.py`, would silently keep only the first person.
  Person filtering belongs before the prompt, on the detector's output. `test_refined.py` pins this.
- **SAM2 returns `masks.data` as `dtype=bool`**, unlike YOLO11-seg's float masks. The `Segmenter`
  protocol requires float32, so `refined.py` converts explicitly.
- **`SAM("sam2_b.pt")` downloads into the current working directory** exactly like `YOLO(...)` does.
  Observed the hard way: a probe with a bare filename dropped 78 MB into the repo root. Always pass
  an absolute path from `weights_dir()`.
- **`BlurInpainter` blurs the original image, subject included**, so the subject's colours smear
  outward into the background near the silhouette. That soft fringe is the backend's own artefact,
  not a stale-background halo — easy to confuse when eyeballing a seam. It is accepted: this is a
  dev-loop backend, not a quality fallback. Use `SolidColorInpainter` with a colour absent from the
  photo when you want halo leakage to be unambiguous.
- **`k`/`j`/`feather` are absolute pixel counts, so their effect is resolution-dependent.** At
  810x1080 the shipped `j=3` trims about 10% of subject *area* because the figures are small; the
  same 3px is negligible on a 4000px portrait. Defaults were tuned on a small image, which is the
  conservative direction, but scaling them with the image diagonal is unresolved — see #7, where the
  CLI exposes them.
- **Score mask quality with boundary IoU, not plain IoU.** Plain IoU is dominated by the torso and
  barely moves when the hair is wrong, making it useless as a signal for the thing this project
  actually cares about.
- `uv.lock` is committed on purpose (futseg ships as an application), and `.idea/` is only
  partially ignored — the shareable JetBrains project files are tracked deliberately.
