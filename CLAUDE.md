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

Scaffolding stage: `pyproject.toml` and `.gitignore` only (`futseg`, Python >=3.12, no dependencies
yet), no source code or tests yet. The intended architecture is fully specified in `PLAN.md` and
tracked as GitHub issues/milestones on `pr3mar/futseg`. All issues are currently open/unstarted —
issue #2 (Scaffolding) is next.

GitHub milestones 1–9 match `PLAN.md`'s milestones exactly, one issue each. **The mapping is not
`issue N → milestone N−1`** — SAM2 refinement was added later, so it carries the highest issue
number while sitting fourth in the sequence:

| Milestone | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Issue | #2 | #3 | #4 | **#10** | #5 | #6 | #7 | #8 | #9 |

`#1` is the unscheduled roadmap issue (alpha matting, tiling/SR, batch/video, REST API) and has no
milestone.

When code lands, update this file with real build/lint/test commands and keep the architecture
section below in sync with what's actually implemented (mark deviations from the plan here, not
just in `docs/wiki.md`).

## Architecture (planned, per `PLAN.md`)

```
src/futseg/
  pipeline.py          # segment -> mask derivation -> inpaint -> compose
  cli.py                # typer CLI: `futseg run`, `futseg segment`
  io.py                 # image load/save helpers
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
- **A bare `torch` dependency does not resolve to a CUDA build on every platform.** It needs an
  explicit `[[tool.uv.index]]` pin to the PyTorch CUDA wheel index, and must still degrade to a CPU
  wheel on a machine without CUDA so a clean checkout syncs. Getting this wrong makes milestone 1
  appear to succeed and milestone 6 mysteriously unusable — see `PLAN.md` "Dependencies".

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
