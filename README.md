# futseg

Person segmentation and prompt-driven background replacement, as a CLI and an importable Python
library. Give it a photo and a text prompt: it segments the people, regenerates everything *behind*
them from the prompt, and composites the original subjects back on top untouched.

Local open-source models only — nothing is sent to a hosted API.

## Quickstart

Everything runs in a container. You need **docker with GPU support** and an NVIDIA driver; you do
not need Python, `uv`, or CUDA installed on the host.

```bash
git clone https://github.com/pr3mar/futseg.git
cd futseg
make build            # once: ~5 min, builds the dev image
```

Drop photos into `input/` (gitignored), then:

```bash
make shell            # a shell inside the container, futseg on PATH
```

```bash
# Replace the background. Downloads ~15 GB of model weights on first use.
futseg run input/photo.jpg \
  --prompt "a windswept cliff at sunset, dramatic clouds" \
  --out out/result.png

# Masks only: no diffusion weights, no prompt, no GPU strictly required.
futseg segment input/photo.jpg --out out
```

Or without opening a shell:

```bash
make exec CMD="futseg run input/photo.jpg --prompt 'a quiet forest' --out out/result.png"
```

To process everything in `input/` at once:

```bash
scripts/run_all.sh
PROMPT="a neon-lit street at night" MODEL=sdxl-inpaint scripts/run_all.sh
```

## Commands

### `futseg run` — replace the background

```
futseg run <image> --prompt "..." [--out out/result.png]
                   [--backend diffusion|composite] [--model <key>]
                   [--quality best|fast] [--device auto|cuda|cpu]
                   [--steps N] [--guidance-scale F] [--weights-dir PATH]
```

| Flag | Default | Notes |
|---|---|---|
| `--prompt` | *required* | Describes the **background**, not the people — see below |
| `--out` | `out/result.png` | Parent directories are created |
| `--backend` | `diffusion` | `composite` needs no model weights; useful for a fast loop |
| `--model` | `flux2-klein` | Registry key, see the model table |
| `--quality` | `best` | `best` = YOLO11 → SAM2; `fast` = YOLO11-seg alone, coarser |
| `--device` | `auto` | Resolved once and passed down |
| `--steps` | per-model | Leave unset; each checkpoint declares its own |
| `--guidance-scale` | per-model | Leave unset; ignored entirely by distilled models |
| `--weights-dir` | XDG cache | Where model weights are downloaded |

### `futseg segment` — masks only

```
futseg segment <image>... [--out out] [--quality best|fast] [--device auto|cuda|cpu]
                          [--inpaint-grow K] [--composite-shrink J] [--feather R]
                          [--weights-dir PATH]
```

Takes several paths, so `futseg segment input/*.jpg` works. Writes five files per input:

| File | Answers |
|---|---|
| `<stem>-alpha.png` | what the segmenter actually returned |
| `<stem>-inpaint-mask.png` | what a generative backend would regenerate |
| `<stem>-composite-alpha.png` | the feathered mask the subject is pasted back with |
| `<stem>-overlay.png` | is the silhouette right? — background tinted, judged against the photo |
| `<stem>-cutout.png` | the subject on transparency |

The **overlay** is the one to look at when judging quality: hair and fingers only reveal how coarse
a mask is when seen against the photograph rather than against black.

`--inpaint-grow` must exceed `--composite-shrink + --feather`, or the tool refuses with a usage
error. That inequality is what keeps a rim of stale background off the seam.

### Exit codes

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | no person found — an empty mask is never reported as success |
| `2` | usage error |

Errors go to stderr, so stdout stays parseable when piped.

## What the prompt does — and does not do

**The prompt describes the background only.** The people in your photo are composited back on top
untouched, by design. Asking for "three crew members in flight armor" will not put your subjects in
armor; it influences what the model paints *around* them, and may add extra figures in the
background.

If you want the subjects themselves restyled, that is a different operation than background
replacement, and this tool deliberately cannot do it.

## Models

Weights download on first use into `$XDG_CACHE_HOME/futseg` (override with `--weights-dir`) and are
never written to the working directory.

### Inpainting backends (`--model`)

| Key | Repo | Download | Native | Licence |
|---|---|---|---|---|
| `flux2-klein` *(default)* | `black-forest-labs/FLUX.2-klein-4B` | ~15 GB | 1024 | **Apache-2.0** |
| `sdxl-inpaint` | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | ~6.5 GB | 1024 | OpenRAIL++ |
| `sd15-inpaint` | `stable-diffusion-v1-5/stable-diffusion-inpainting` | ~2.6 GB | 512 | CreativeML OpenRAIL-M |

All three are ungated — no Hugging Face token required. `flux2-klein` is the default because it is
the only permissively licensed one, and it is step-distilled, so it generates in 4 steps.

Use `sd15-inpaint` if you want the smallest download; its 512px native canvas is the quality
bottleneck, since the result is upscaled to your photo's resolution.

### Segmentation

YOLO11 (detection and `-seg`) and SAM2, via [`ultralytics`](https://github.com/ultralytics/ultralytics).

> **Licence note:** `ultralytics` and the YOLO11 weights are **AGPL-3.0**. If you intend to
> distribute a product built on futseg, that obligation applies to you and is stricter than
> anything else in this list. It is stated here because it is easy to miss behind a `pip install`.

## How it works

```
photo ──> segment ──> derive two masks ──> inpaint background ──> composite person back
             │              │                      │                        │
        YOLO11 + SAM2   inpaint_mask /        diffusion model         original pixels,
                        composite_alpha       (prompt-driven)         feathered edge
```

1. **Segment** — YOLO11 detects each person and hands the boxes to SAM2 as prompts; the refined
   instance masks are unioned into one foreground alpha, then small enclosed holes are filled.
2. **Derive two masks** — one segmentation yields *two* masks with opposite offsets: the inpaint
   region is grown *into* the subject, while the composite mask is pulled *in* and feathered.
3. **Inpaint** — a diffusion model regenerates the background from the prompt, at its native
   resolution, and the result is scaled back up.
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

The goal is *pixel-perfect* person segmentation. The current implementation does not fully reach
it, and the docs say so rather than overclaiming.

**Binary masks cannot represent semi-transparent regions.** Hair is the obvious case — those pixels
are genuinely a blend of subject and background, and feathering a hard edge approximates that with
uniform blur rather than recovering true per-pixel opacity. Transparent spectacle lenses are the
same problem: segmenters classify them as background, and no amount of mask post-processing fixes a
region that is genuinely both. Closing this needs an alpha-matting stage, which is planned and
deliberately deferred.

**Worn occluders are excluded.** Seatbelts and bag straps crossing a subject are classified as
not-person, leaving a stripe of regenerated background across the chest. Tracked in
[issue #28](https://github.com/pr3mar/futseg/issues/28).

The `Segmenter` protocol already returns float alpha rather than a boolean mask specifically so the
matting upgrade drops in without a breaking change.

## Development

```bash
make help          # every target
make check         # ruff + the full test suite
make shell         # interactive shell in the container
make cuda          # verify CUDA actually works, not just that it is "available"
make up / down     # keep a container resident; task targets reuse it
```

Tests run without downloading any weights — model backends are injected. Tests that do use real
models are marked `@pytest.mark.slow` and run with the rest by default; deselect with
`-m "not slow"`.

## Roadmap

- Alpha matting for true per-strand edge accuracy and semi-transparent regions
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

## Requirements

Docker with GPU support, and an NVIDIA driver on the host. Nothing else — Python 3.12 and `uv` live
inside the image.

The `composite` backend runs without a GPU. The diffusion backends will run on CPU via
`--device cpu`, but slowly enough that it is only useful for testing.

## Licence

**Not yet specified.** Note that the AGPL-3.0 obligation from `ultralytics` (see Models above)
constrains what this project can be licensed as, and that decision has not been made.
