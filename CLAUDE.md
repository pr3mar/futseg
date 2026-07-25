# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Global working principles and standing rules** (contradiction-flagging, no secrets/private
info, verify-before-claiming-done, simplicity/surgical-change discipline, etc.) live in
`~/.claude/CLAUDE.md` and apply here with no project-specific override. This file only covers
what's specific to futseg.

## Role

Work this repo as a principal ML engineer: deep expertise in computer vision (instance
segmentation, mask edge quality) and diffusion-based generative inpainting, plus senior judgment
on code organization and git practice. The end goal is a CLI that does **pixel-perfect** person
segmentation and prompt-driven background replacement — hold that bar, not "good enough."

That bar is the *goal*, not a description of what currently ships. `PLAN.md`'s "Quality bar, and
the known gap" records the shortfall explicitly: binary masks cannot represent semi-transparent
hair, so the bar is not literally met until the deferred alpha-matting stage lands. This is a
recorded, scheduled gap — don't re-open it as a fresh discovery, and don't let the README claim
otherwise.

What that means concretely:
- **Segmentation is not an afterthought.** Mask edge quality, multi-person and occlusion handling,
  and instance-union correctness are where seam quality is actually won or lost — treat
  `masking.py` with the same rigor as the model call itself, including the two-mask derivation
  (inpaint vs. composite) that keeps a stale-background rim off the seam. Know *why* the default
  segmenter is YOLO11-detect → SAM2 box-prompted refinement rather than raw YOLO11-seg, whose
  prototype masks quantize to ~25 source pixels on a large photo (see `PLAN.md`), and revisit that
  judgment if evidence says otherwise, don't just defer to the doc.
- **Inpainting is not "call the pipeline and hope."** Reason about SD inpainting-specific failure
  modes — color bleed at the mask boundary, latent-space blending artifacts, prompt adherence vs.
  mask fidelity tradeoffs, guidance scale/strength effects — and design/test against them
  explicitly, especially at the person/background seam.
- **Architecture and git are a principal engineer's, not a script kiddie's:** the `Segmenter`/
  `Inpainter` protocol boundary stays clean, PRs stay small and reviewable per issue, nothing gets
  claimed correct without verification (global principle). Boring, predictable structure beats
  cleverness that isn't earned.
- **Push back when the plan is wrong.** If a design choice in `PLAN.md` conflicts with CV/diffusion
  best practice once you're actually implementing it, say so and propose the alternative — don't
  silently implement something you'd flag in review (ties into the global contradiction-flagging
  and "push back" principles).

## Project state

Pre-implementation: docs and repo configuration only (`pyproject.toml` with no dependencies yet,
`README.md`, `.gitignore`, `.gitattributes`, `PLAN.md`, `docs/`). No source code or tests yet. The
intended architecture is fully specified in `PLAN.md`, with the platform contract in
`docs/design/2026-07-25-linux-first-platform.md`, and tracked as GitHub issues/milestones on
`pr3mar/futseg`. Issue #2 (Scaffolding) is next — and it is the first one that must be done on
Linux/WSL2 rather than Windows.

GitHub milestones 1–11 match `PLAN.md`'s milestones exactly, one issue each. **The mapping does not
run in issue order** — later additions carry higher issue numbers while sitting earlier in the
sequence:

| Milestone | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Issue | #2 | #3 | #4 | **#10** | #5 | #6 | #7 | #8 | #9 | #14 | #15 |

Milestones 10 (Docker) and 11 (CI) are deliberately last and deliberately underspecified — their
shape depends on the CLI existing first.

Issues without a milestone: `#1` (unscheduled roadmap — alpha matting, tiling/SR, batch/video, REST
API) and `#13` (the Linux-first platform decision, which spans milestones rather than being one).

When code lands, update this file with real build/lint/test commands and keep the architecture
section below in sync with what's actually implemented (mark deviations from the plan here, not
just in `docs/wiki.md`).

## Architecture (planned, per `PLAN.md`)

```
src/futseg/
  pipeline.py          # segment -> mask derivation -> inpaint -> compose
  cli.py                # typer CLI: `futseg run`, `futseg segment`
  io.py                 # image load/save helpers
  device.py             # resolve_device(): the single cuda/cpu decision point
  paths.py              # XDG cache resolver; keeps weights out of CWD/install dir
  masking.py            # alpha -> (inpaint_mask, composite_alpha); dilate/erode/feather
  segmentation/
    base.py             # Segmenter protocol: segment(image) -> HxW float32 alpha in [0,1]
    yolo.py              # YOLO11-seg, person class, instance union       (--quality fast)
    refined.py           # YOLO11 boxes -> SAM2 refinement   (default, --quality best)
  inpaint/
    base.py             # Inpainter protocol: inpaint(image, mask) -> full-canvas Image
    diffusion.py         # ModelSpec registry (FLUX.2 / SDXL / SD2), swappable via --model
    composite.py           # non-generative fallback (Pillow-only: color/blur/static image)
tests/                   # unit tests with mocked Segmenter/Inpainter; @pytest.mark.slow for real models
```

- `Segmenter` and `Inpainter` are `Protocol`s so backends are swappable without touching
  `pipeline.py` or the CLI (e.g. the deferred alpha-matting segmenter, or a new diffusion model).
- **`Segmenter` returns float alpha, not a boolean mask**, even though both current backends only
  emit hard 0.0/1.0. Alpha matting is a scheduled upgrade and `bool -> float` is the expensive
  migration; `masking.py` needs float alpha for the composite regardless. Don't "simplify" this
  back to `bool`.
- **One segmentation yields two masks.** `inpaint_mask = 1 - erode(alpha, k)` grows the inpaint
  region *into* the person; `composite_alpha = feather(erode(alpha, j))` pulls the pasted person
  *in*, with `k > j + feather_radius`. Conflating them leaves a rim of stale original background
  hugging the subject — the classic cheap-cutout halo. This is the single easiest thing to get
  wrong in `masking.py`.
- **`Inpainter.inpaint(image, mask)` takes no prompt.** Prompt and sampler settings are
  constructor-injected into the diffusion backend, so `composite.py` isn't forced to accept a
  parameter it ignores.
- The pipeline always finishes with a feathered composite of the original person over whichever
  backend produced the background, so the person is never altered by the generative model, even
  at the mask boundary — this invariant should hold regardless of which backend is active.
- `composite.py` exists so segmentation/pipeline plumbing can be validated on real photos without
  pulling multi-GB diffusion weights; it's not a quality fallback, it's a fast dev-loop backend.

## Environment & portability

- Python >=3.12, managed with `uv` (`uv sync`, `uv run ...`) only — no system Python, pip, poetry,
  or raw venv assumptions.
- Must stay installable/portable on other machines: no hardcoded local paths, no machine-specific
  assumptions in code, config, or docs. Before opening a PR, verify a fresh `uv sync` + `uv run
  pytest` succeeds from a clean checkout.
- `ruff` for lint, `pytest` for tests (target config once milestone 1/#2 lands).
- **Linux is the target platform; Windows native is not supported.** Development happens in WSL2,
  so dev == CI == prod. macOS is developable for segmentation and the composite backend only. Full
  contract in `docs/design/2026-07-25-linux-first-platform.md`.
- **`torch` is a plain PyPI dependency — do not add a custom index.** The Linux x86_64 wheel bundles
  CUDA (527 MB, vs 122 MB CPU-only on Windows; every `nvidia-*` dep is gated
  `platform_system == "Linux"`). An earlier revision of this plan pinned
  `download.pytorch.org` to rescue Windows; targeting Linux deleted that requirement. If you find
  yourself reaching for `[[tool.uv.index]]`, check whether you are solving a Windows problem the
  project no longer has.
- **Nothing writes to the current working directory or the install directory.** `ultralytics`
  downloads checkpoints into CWD by default, which breaks on read-only and ephemeral container
  filesystems; `paths.py` redirects it and `HF_HOME` to one XDG cache dir.
- **Never probe for CUDA outside `device.py`.** Backends receive a resolved device string so the
  policy stays in one testable place.

## Working GitHub issues

Development happens by picking up a `pr3mar/futseg` issue and shipping it as its own PR.
**REQUIRED:** use the `github-issue-workflow` skill (`.claude/skills/github-issue-workflow/`)
before branching, committing, or opening a PR — it covers picking/claiming a ticket (optionally
passed in as input), the branch/PR mechanics, and when to update `transcript.md`/`docs/wiki.md`.

Hard rules (no exceptions, restated from the skill since they're load-bearing):
- `main` is off limits at all times — no direct commits, no direct pushes.
- PRs are never merged by the assistant. Every PR is opened for the user to review and merge.
- Every PR links back to the issue it resolves (`Closes #<n>`) and discloses everything it changed.

## Traceability

- `transcript.md` (repo root): append-only decision log, updated as work happens, not after.
- `docs/wiki.md`: cumulative knowledge base (conventions, decisions, gotchas), reviewed/updated
  before every PR.
- Contradiction-flagging is a global principle (`~/.claude/CLAUDE.md`) — applies here to conflicts
  between an issue, this file, `docs/wiki.md`, `PLAN.md`, and existing code.
