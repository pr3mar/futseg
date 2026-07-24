# futseg project wiki

Cumulative knowledge base: decisions, conventions, and gotchas. Update whenever something new is
learned, and always review before opening a PR (see `github-issue-workflow` skill). Unlike
`transcript.md` (chronological, append-only), this file is organized by topic and should be edited
in place as understanding changes — keep it consistent, not just additive.

## Conventions

- Package management: `uv` only (`uv sync`, `uv run ...`). No `pip install` / `poetry` / raw `venv`.
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

## Decisions

- **Default segmenter is YOLO11 *detection* → SAM2 box-prompted refinement** (`--quality best`),
  person class, instance masks unioned. Raw YOLO11-seg is demoted to `--quality fast`. The earlier
  objection to SAM — "needs external box/point prompts" — was self-defeating, since the detector is
  the prompt source. MediaPipe Selfie Segmentation remains rejected (single-portrait only).
- **One segmentation produces two derived masks**: `inpaint_mask = 1 - erode(alpha, k)` and
  `composite_alpha = feather(erode(alpha, j))`, with `k > j + feather_radius`. Same source,
  opposite offsets. Conflating them leaves a rim of stale original background at the silhouette.
- **Inpainting backend is a `ModelSpec` registry** (`inpaint/diffusion.py`), not a single
  hardcoded checkpoint, so models are swappable via `--model`. The registry's `to_kwargs` adapter
  is load-bearing: FLUX.2 (`Flux2Pipeline`, instruction/reference-driven) and SDXL-inpaint
  (classic mask-conditioned) do **not** share a call signature, so an ID-only mapping would not
  make them interchangeable.
- **Default model is FLUX.2 [klein] 4B** — Apache 2.0, handles inpainting in the same weights.
  FLUX.2 [dev] 32B and FLUX.1 Fill are both non-commercial (and dev-32B needs 4-bit quantization
  to fit a consumer GPU); SDXL-inpaint and SD2-inpaint remain in the registry as alternatives.
  Per-model licenses must be listed in the README.
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
- **GitHub milestones 1–9 match `PLAN.md` one-to-one, but the issue numbering does not run in
  order.** SAM2 refinement was added after the original eight issues were filed, so it holds the
  highest issue number while sitting fourth in the sequence: milestones 1–9 map to issues
  #2, #3, #4, **#10**, #5, #6, #7, #8, #9 respectively. `#1` is the unscheduled roadmap issue and
  carries no milestone. Don't assume issue N corresponds to milestone N−1.

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
- **A bare `torch` dependency does not resolve to a CUDA build on every platform.** It needs an
  explicit `[[tool.uv.index]]` pin to the PyTorch CUDA wheel index, and must still degrade to CPU
  on machines without CUDA. Get this wrong and milestone 1 "succeeds" while milestone 6 is
  mysteriously unusable.
- **Score mask quality with boundary IoU, not plain IoU.** Plain IoU is dominated by the torso and
  barely moves when the hair is wrong, making it useless as a signal for the thing this project
  actually cares about.
- `uv.lock` is committed on purpose (futseg ships as an application), and `.idea/` is only
  partially ignored — the shareable JetBrains project files are tracked deliberately.
