# futseg

Person segmentation and prompt-driven background replacement, as a CLI and an importable Python
library. Give it a photo and a text prompt: it segments the people, regenerates everything *behind*
them from the prompt, and composites the original subjects back on top untouched.

Local open-source models only — nothing is sent to a hosted API.

> **Status: pre-implementation.** The architecture, milestones and design rationale are fully
> specified in [`PLAN.md`](PLAN.md) and tracked as GitHub issues. No source code has landed yet.

## How it works

```
photo ──> segment ──> derive two masks ──> inpaint background ──> composite person back
             │              │                      │                        │
        YOLO11 + SAM2   inpaint_mask /        diffusion model         original pixels,
                        composite_alpha       (prompt-driven)         feathered edge
```

1. **Segment** — YOLO11 detects each person and hands the boxes to SAM2 as prompts; the refined
   instance masks are unioned into one foreground alpha.
2. **Derive two masks** — one segmentation yields *two* masks with opposite offsets: the inpaint
   region is grown *into* the subject, while the composite mask is pulled *in* and feathered.
3. **Inpaint** — a diffusion model regenerates the background from the prompt.
4. **Composite** — the original person is pasted back over the generated background.

## Design commitments

- **The generative model never touches the person.** The pipeline always ends by compositing the
  original subject pixels over whatever the backend produced, so the person is preserved exactly —
  even at the mask boundary. This holds for every backend.
- **Two masks, not one.** The pixels just outside a silhouette are camera-anti-aliased blends of
  subject and *old* background. Leaving them outside the inpaint region is what produces the
  telltale rim of stale background hugging a cheap cutout. Growing the inpaint region inward and
  pulling the composite edge in prevents it.
- **Backends are swappable.** `Segmenter` and `Inpainter` are `Protocol`s, and diffusion models
  live behind a registry, so changing model — or adding a whole new approach — doesn't touch the
  pipeline or the CLI.
- **Edge quality is the point.** Mask post-processing is treated with the same rigour as the model
  calls; it is where seam quality is actually won or lost.

## Quality bar, stated honestly

The goal is *pixel-perfect* person segmentation. The current design does not fully reach it yet,
and the docs say so rather than overclaiming:

Binary masks cannot represent semi-transparent hair — those pixels are genuinely a blend of subject
and background, and feathering a hard edge approximates that with uniform blur rather than
recovering true per-pixel opacity. Closing that gap needs an alpha-matting stage, which is planned
but deliberately deferred so the rest of the pipeline can ship first. Until it lands, edges are
high quality but not literally per-strand accurate.

The `Segmenter` protocol already returns float alpha rather than a boolean mask specifically so
that upgrade drops in without a breaking change.

## Roadmap

- Alpha matting for true per-strand edge accuracy
- Tiled inpainting / background super-resolution, removing the current downscale-upscale softness
- Batch and video input
- Packaging as a REST API

## Documentation

| File | Purpose |
|---|---|
| [`PLAN.md`](PLAN.md) | Architecture, milestones, model choices and their rationale |
| [`CLAUDE.md`](CLAUDE.md) | Working conventions and project-specific guidance |
| [`docs/wiki.md`](docs/wiki.md) | Cumulative decisions, conventions and gotchas |
| [`transcript.md`](transcript.md) | Append-only decision log |

Install instructions, a quickstart example and the per-model license table land with the CLI —
tracked in issue #9.

## Requirements

Python 3.12+, managed with [`uv`](https://docs.astral.sh/uv/). A CUDA-capable GPU is recommended
for the diffusion backend; the non-generative composite backend runs on CPU.

## License

Not yet specified.
