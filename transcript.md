# Transcript

Append-only log of decisions made while working on this repo, and the reasoning behind them.
Newest entries at the bottom. Reference the issue/PR a decision belongs to when applicable.

Record decisions and their reasoning, not the conversations that produced them: no personal
details, no machine/hardware specifics, no credentials. This repo is public.

## 2026-07-24

- Created GitHub milestones 1–8 on `pr3mar/futseg` matching `PLAN.md`'s phases, and one issue per
  milestone (#2–#9) plus a roadmap issue (#1) for deferred work (SAM2 refinement, batch/video
  input, REST API). Reasoning: PLAN.md's milestones map 1:1 onto independently shippable units of
  work, so issues track them directly rather than introducing a different breakdown.
- Created a GitHub Projects (v2) board ("futseg CLI Development") reusing an existing empty
  untitled project under the `pr3mar` account rather than creating a new one, since it had 0 items
  and no title — avoided leaving a duplicate empty project behind.
- Removed a hardware-specifics line (GPU model and VRAM figure) from issue #6. Standing rule:
  hardware specs and other private machine details never go into repo content — issues, PRs,
  commits, or docs — regardless of repo visibility. See `~/.claude/CLAUDE.md`.
- Added the `github-issue-workflow` skill (`.claude/skills/github-issue-workflow/SKILL.md`) and
  the corresponding CLAUDE.md sections (Environment & portability, Working GitHub issues,
  Traceability, Privacy). Decision: kept the *generic* pick-up/PR procedure and hard git rules in
  the skill, and kept *project-specific* facts (uv, portability requirement, file locations,
  privacy rule) in CLAUDE.md — skills are for reusable technique, CLAUDE.md is for durable
  project-specific convention.
- Established `~/.claude/CLAUDE.md` as a global, cross-project standing-instructions file:
  contradiction-flagging and no-secrets/no-private-info rules, plus a set of "Working Principles"
  (simplicity, surgical changes, verify-before-claiming-done, etc.). Iterated during review —
  notably removed a "live instructions override defaults" clause, since a blanket override clause
  defeats every principle it supposedly qualifies (classic unscoped exemption-clause failure).
  futseg's own privacy section was collapsed to a pointer once the rule became global, and the
  matching project-scoped memory was deleted rather than left as a stale duplicate.
- Rewrote `CLAUDE.md` to point at the new global principles file instead of restating anything
  covered there, added a concrete "Architecture (planned)" section (module layout + protocol
  design + the person-preservation invariant) summarized from `PLAN.md` so it's visible without
  opening that file, and noted current issue/milestone status (#2–#9 open, #2 next) after checking
  `gh issue list` rather than assuming.
- Added a "Role" section to `CLAUDE.md`: work this repo as a principal ML engineer (CV
  segmentation + diffusion inpainting expertise, senior code/git judgment), holding a
  pixel-perfect segmentation/background-replacement bar rather than "good enough." Kept it
  project-scoped (not global) since the domain expertise is futseg-specific; tied its "push back
  on the plan" clause to the existing global contradiction-flagging principle rather than
  duplicating it.
- Removed the hardware-specifics line from `PLAN.md`'s Context section (it named a GPU model and
  VRAM figure), replacing it with "the target environment assumes a CUDA-capable GPU" — the
  reasoning the plan needed, without the machine detail. The same rule had already been applied to
  issue #6; `PLAN.md` had been missed.
- Scrubbed this file before its first commit: it had quoted the removed hardware spec verbatim
  while recording its removal, and asserted the repo was private when `gh repo view` reports it
  PUBLIC (matching `docs/wiki.md`). Also converted conversational narration into impersonal
  decision-log voice. Convention going forward, now stated in the header: log the decision and its
  reasoning, never the private value being removed.
- Revised `PLAN.md` substantially after a technical review of it against the `CLAUDE.md` vision.
  Decisions, with the reasoning:
  - **Segmenter changed from YOLO11-seg to YOLO11-detect → SAM2 box-prompted refinement**, with
    YOLO11-seg demoted to a `--quality fast` tier. Reason: ultralytics emits 32 mask prototypes at
    1/4 stride (160×160 at `imgsz=640`), so on a 4000px photo one mask pixel spans ~25 original
    pixels — irreconcilable with a pixel-perfect bar. The plan's original objection to SAM ("needs
    external box/point prompts") was self-defeating: the detector is the prompt source.
  - **`Segmenter` protocol returns float32 alpha [0,1], not a boolean mask**, even though both
    current backends only emit hard 0/1. Reason: alpha matting is a scheduled upgrade (below), and
    bool→float is the expensive migration; `masking.py` needs float alpha for compositing anyway.
    Chose forward-compatible typing over a later breaking change — not speculative, the upgrade is
    planned.
  - **Alpha matting deferred to roadmap**: ship the basic path first, upgrade later. Recorded
    honestly in a new "Quality bar, and the known gap" section: binary masks cannot represent
    semi-transparent hair, so until matting lands the pipeline is high-quality but *not* literally
    pixel-perfect, and the README must say so rather than overclaim. Trimap construction is already
    available for free from the dilate/erode band.
  - **Split one mask into two derived masks**: `inpaint_mask = 1 - erode(alpha, k)` and
    `composite_alpha = feather(erode(alpha, j))` with `k > j + feather_radius`. Reason: pixels just
    outside a silhouette are camera-anti-aliased person/old-background blends; leaving them outside
    the inpaint region leaves a stale-background rim at the seam (the classic cheap-cutout halo).
    The original plan described one generically post-processed mask.
  - **Inpaint backend generalized from `sd_inpaint.py` to `diffusion.py` + a `ModelSpec` registry**
    so models are easily swappable. The `to_kwargs` adapter is load-bearing: FLUX.2 editing
    (`Flux2Pipeline`, instruction/reference-driven) and SDXL-inpaint (classic mask-conditioned) do
    not share a call signature, so an ID-only mapping would not make them interchangeable.
  - **Default model = FLUX.2 [klein] 4B, not FLUX.1 Fill or FLUX.2 [dev].** Verified via web search
    (assistant knowledge cutoff predates FLUX.2): klein-4B is **Apache 2.0** and handles inpainting
    in the same weights, whereas FLUX.2 [dev] 32B and FLUX.1 Fill are both non-commercial, and
    dev-32B needs 4-bit quantization to fit a consumer GPU. A non-commercial license had been
    deemed acceptable, but the newer model removes the need for one.
  - **Prompt removed from the `Inpainter` call signature**, constructor-injected instead. The
    original plan had `inpaint(image, mask)` while calling the backend "prompt-driven" and giving
    the CLI a `--prompt` — an internal contradiction. Constructor injection also keeps
    `composite.py` from accepting a parameter it ignores.
  - **Explicit resolution strategy added** (downscale to model native res → inpaint → upscale the
    generated background → composite the full-res person). The original plan's "full-canvas Image"
    return type silently hid the question, and SD2's 512px native res would have put a sharp person
    on a soft background.
  - **`--quality fast|best` knob added** to the CLI.
  - **Boundary IoU on hand-annotated fixtures added to Verification.** Plain IoU is dominated by the
    torso and barely moves when hair is wrong; without a boundary metric the pixel-perfect claim is
    unfalsifiable and backend swaps have no regression signal.
  - Deps added as a consequence: `opencv-python-headless` (morphology/feather), optional
    `bitsandbytes` (quantized FLUX.2 [dev]), and an explicit `[[tool.uv.index]]` pin for the torch
    CUDA wheel — a bare `torch` dependency does not resolve to a CUDA build on all platforms, which
    would have made milestone 1 "succeed" and milestone 6 mysteriously unusable.
  - Milestone count went 8 → 9 (SAM2 refinement is its own shippable unit), which **breaks the
    documented 1:1 mapping** between milestones and issues #2–#9. Flagged for review; no GitHub
    issues created or edited without approval.
- Added an anonymization guard as a `PreToolUse` hook on `Bash`
  (`~/.claude/hooks/anonymization-guard.sh`, wired in `~/.claude/settings.json`). It blocks
  `git commit`/`git push` when added lines match hardware-spec, credential, or local-home-path
  patterns. Installed at **user level rather than project level**, because the underlying rule in
  `~/.claude/CLAUDE.md` is explicitly cross-project — a futseg-only hook would under-enforce it.
  Design decisions worth keeping:
  - **No `jq` dependency.** `jq` is not installed on this machine; every stock hook example uses
    it, and a hook whose command silently fails is worse than no hook. Uses grep/awk/sed only.
  - **Mode detected by scanning the raw payload** for "git push"/"git commit" instead of parsing
    JSON or using an `if: Bash(git *)` prefix filter. Prefix filters miss compound commands like
    `cd x && git push`; over-triggering on a command that merely mentions the words is harmless
    (an extra scan that passes), so the failure direction is deliberately biased toward catching.
  - **Never echoes the matched value** — reports only which rule fired and in which file. A guard
    that prints the secret it caught into the transcript defeats its own purpose.
  - **Push is scanned more strictly than commit**: commit checks the staged diff, push checks
    everything that would leave the machine (unpushed commits, or full history when there is no
    upstream yet).
  - Verified end-to-end: silent on non-git commands, allows clean content, denies staged hardware
    specs and credentials, still denies on push when the offending content is already committed and
    staging is clean, catches compound commands, and produces **no false positives** against the
    current repo contents.
- Added `.gitignore`. Two deliberate deviations from a stock Python template:
  - **`.idea/` is not blanket-ignored.** `misc.xml`, `modules.xml`, `vcs.xml` and the JetBrains-
    generated `.idea/.gitignore` were already staged as the shareable project definition, so only
    per-developer state (`workspace.xml`, `usage.statistics.xml`, shelf, dataSources, …) is ignored.
  - **`.claude/` is not blanket-ignored.** `.claude/skills/` is project infrastructure and stays
    tracked; only local state (`settings.local.json`, plans, history, shell snapshots) is ignored.
  Also ignores assistant chat histories/session state from other tools (aider, cursor, windsurf,
  continue, …), since those transcripts routinely capture paths, machine details and occasionally
  credentials — same threat model as the scrub above. Model weights are ignored with a note that
  `ultralytics` downloads checkpoints into the *current working directory* on first run, so
  `yolo11*.pt` would otherwise land at the repo root uninvited. `uv.lock` is deliberately left
  tracked (futseg ships as an application, so the lockfile belongs in git).
- Synced `CLAUDE.md` and `docs/wiki.md` to the revised `PLAN.md`. Beyond the mechanical updates
  (module tree, float-alpha protocol, two-mask derivation, model registry), two corrections worth
  noting: `CLAUDE.md`'s Role section had told future sessions to know "why this project chose
  YOLO11-seg over SAM/MediaPipe" — stale, since the default is now YOLO11-detect → SAM2 — and the
  pixel-perfect bar is now explicitly labelled a *goal* with a recorded, scheduled shortfall, so it
  does not get re-litigated as a fresh discovery each session.
- Reconciled the GitHub backlog with the revised plan. Renumbered milestones so GitHub matches
  `PLAN.md` 1:1 (created "4. Segmentation: SAM2 refinement", renamed the old 4–8 to 5–9, renamed 3
  to "Segmentation: YOLO11-seg fast tier"), rewrote all nine existing issue bodies, and opened #10
  for the SAM2 tier. Three issues were not merely stale but actively misleading and would have sent
  whoever picked them up in the wrong direction:
  - **#1** listed SAM2 refinement as deferred roadmap work when it is now the default segmenter.
    Replaced with alpha matting and tiled-inpainting/SR; recorded why the original deferral
    rationale was self-defeating.
  - **#4** was titled and scoped as *the* segmentation backend, and still carried the argument
    against SAM ("needs external box/point prompts") that the revision overturned. Retitled to the
    `--quality fast` tier, with the ~25px prototype-quantization arithmetic written into the body
    so the demotion is self-explanatory.
  - **#6** specified `sd_inpaint.py` and `stable-diffusion-2-inpainting`. Retitled to the swappable
    model registry, with the per-model license table and a note that the `to_kwargs` adapters must
    be verified against the installed `diffusers` rather than assumed.
  Note the resulting mapping is **not** `issue N → milestone N−1`: #10 sits at milestone 4, so the
  order is #2, #3, #4, #10, #5, #6, #7, #8, #9. Both `CLAUDE.md` and `docs/wiki.md` state this
  explicitly, because the obvious assumption is now wrong.
- **Explicit, scoped waiver of the "`main` is off limits" rule, for the bootstrap commit only.**
  The repo had no commits and GitHub reported it empty with no default branch, so there was no
  `origin/main` to branch from and no base for a PR to target — the workflow in
  `.claude/skills/github-issue-workflow/SKILL.md` was unexecutable as written. The deadlock was
  flagged rather than silently resolved, and the user directed the initial commit. `README.md` was
  therefore committed and pushed directly to `main`; every subsequent change goes through a branch
  and PR as normal. Recording this so it does not later read as an undisclosed rule violation.
- Added `README.md`: project description, the segment → mask-derivation → inpaint → composite
  pipeline, design commitments, and an explicit "quality bar, stated honestly" section carrying the
  hair/matting limitation forward from `PLAN.md`. Deliberately *not* the milestone-9 README —
  install, quickstart and the per-model license table remain issue #9, and the file says so.
- Opened #11 to track the plan revision and doc sync, so the PR has an issue to close (`CLAUDE.md`
  requires every PR to link one, and this work resolved none of #1–#10 — it reshaped four of them).
- Note on git remotes: pushes cannot be made from the assistant's shell over SSH. The user's
  `id_ed25519` key is present and is offered to GitHub, but it is passphrase-protected and no
  `ssh-agent` is reachable from a non-interactive shell (`SSH_AUTH_SOCK` unset, no TTY to prompt
  on), so the signature cannot be completed. The Windows `ssh-agent` service is disabled (error
  1058) and would not have helped regardless: `git` here resolves to Git Bash's `/usr/bin/ssh`,
  which speaks `SSH_AUTH_SOCK`, while the Windows agent uses a named pipe. **Decision: `origin`
  stays on HTTPS and `gh`'s credential helper handles auth** — chosen over a fixed-socket
  `ssh-agent` because the agent approach leaves the key unlocked for anything that can reach the
  socket, whereas the `gh` token is scoped and revocable. SSH remains fine from the user's own
  terminal; this constraint applies to the assistant's shell only.
- Revised `.claude/skills/github-issue-workflow/SKILL.md` with what this session exposed. The
  workflow had three gaps that each only surface *after* you have started working:
  - It assumed `origin/main` exists. On an empty repo there is nothing to branch from and no PR
    base, so the workflow is unexecutable — and the only way out collides with its own hard rule.
    Added a preconditions check, and wrote the empty-repo case into the excuse/reality table as a
    *stop and ask*, explicitly not an exception, so the rule keeps its integrity.
  - It assumed an issue always exists for the work. Added: create one first rather than opening a
    PR without `Closes #<n>`.
  - Step 5 told you to verify with `uv sync` / `uv run pytest`, impossible pre-scaffolding. Added:
    say so explicitly in the PR rather than leaving Testing blank, since a blank section reads as
    "verified".
  Also added the HTTPS/SSH precondition, the anonymization hook's behaviour, `--body-file` over
  inline `--body` (PR bodies contain backticks and code blocks that break shell quoting — this bit
  once this session), and a reminder to fix the issue body when the work changes the design it
  describes.
  Committed onto the existing PR branch rather than a new PR: the skill file is *introduced* by
  #12 and does not exist on `main`, so a separate PR would have to re-add it, and branching off the
  PR branch would stack unmerged work for no benefit.

## 2026-07-25

- **Adopted a Linux-first platform strategy (#13).** Design spec:
  `docs/design/2026-07-25-linux-first-platform.md`.
  The trigger was verifying, rather than recalling, how `torch` ships: the PyPI wheel is 527 MB with
  CUDA bundled on Linux x86_64 and 122 MB CPU-only on Windows, with every `nvidia-*` dependency
  gated `platform_system == "Linux"` (checked against the PyPI JSON API for torch 2.13.0). The
  previously-planned `[[tool.uv.index]]` pin existed *only* to rescue Windows, and its failure mode
  was silent — `uv sync` succeeds, `import torch` works, tests pass, and the problem surfaces
  several milestones later at the diffusion backend.
  Decisions:
  - **Windows native is unsupported; development moves to WSL2**, so dev == CI == prod and there is
    one code path. Chosen over "Linux-first, Windows best-effort" because the latter keeps two
    dependency paths and an untested platform alive permanently for one developer's convenience.
    macOS stays developable for segmentation and the composite backend.
  - **No custom package index.** The `[[tool.uv.index]]` / `[tool.uv.sources]` pin is deleted, not
    documented. Targeting Linux removed the requirement rather than working around it.
  - **No MPS backend.** `diffusers` on MPS has real dtype/unsupported-op gaps and there is no Mac
    available to test on; shipping an unverifiable platform claim is worse than scoping it out.
    One-line addition later if a contributor with a Mac can test it.
  - **Device policy lives only in `device.py`.** `resolve_device()` returns `cuda`/`cpu`; backends
    take a resolved string and never probe. Keeps the policy in one testable place and lets tests
    pass a string instead of mocking torch internals. The `torch` import is lazy so `--help` and
    argument errors don't pay for it.
  - **`paths.py` owns every writable location** (`--weights-dir` → `$FUTSEG_CACHE_DIR` →
    `$XDG_CACHE_HOME/futseg` → `~/.cache/futseg`). `ultralytics` otherwise downloads checkpoints
    into the *current working directory*, which is actively hostile in a container — read-only or
    ephemeral workdir — and it gives Docker a single volume to mount.
  - **`.gitattributes` forces LF.** Editing from Windows while running on Linux yields
    `bad interpreter: /bin/bash^M` in containers. Not hypothetical: every commit in the preceding
    session emitted `LF will be replaced by CRLF` warnings.
  - **Docker (milestone 10) and CI (milestone 11) added at the end, deliberately underspecified.**
    Designing a container around a CLI that does not exist yet produces a Dockerfile that fights the
    tool. Only one constraint is fixed now: weights are mounted, never baked into the image.
  - Recorded a WSL2 gotcha for setup: clone onto the Linux filesystem, not `/mnt/c/`, or drvfs I/O
    makes torch's ~5 GB of small files painful.
  Backlog reconciled: milestones 10 and 11 created with issues #14 and #15; issues #2, #3, #6, #7
  and #9 updated for the new modules, flags and platform contract.
- Deviated from the `superpowers:brainstorming` skill's default spec location
  (`docs/superpowers/specs/`) in favour of `docs/design/`. Reason: the default bakes a tooling name
  into a public project's documentation tree, which is noise for anyone reading the repo. Flagged
  rather than silently changed.
- **Revised the repo-location rule from #16 (#17).** The Linux-first spec required the repository to
  live on the WSL2 Linux filesystem and an explicit fresh clone rather than reuse of an existing
  Windows checkout. That conflicted with how the project is actually developed: the working copy is
  a JetBrains project on the Windows filesystem, reached from WSL2 through `/mnt/c/`. The rule is
  superseded; its rationale is not — the drvfs cost is real, but the ~5 GB virtualenv pays it, not
  the ~200 KB of tracked source.
  Decisions:
  - **Source on `/mnt/c/`, virtualenv on the Linux filesystem** via
    `UV_PROJECT_ENVIRONMENT=$HOME/.venvs/futseg`. Chosen over relocating the source, which breaks
    the IDE setup to avoid a cost that is not actually in the source tree, and over accepting drvfs
    for everything, which pays the 5 GB penalty the original rule correctly identified.
  - **The split is a correctness fix, not only a performance one.** One in-tree `.venv/` shared by a
    Windows and a Linux interpreter collides: `uv sync` from either side overwrites the other's
    layout (`Scripts/` vs `bin/`) with no warning. The original text did not mention this.
  - The `~/code/futseg` gotcha recorded earlier on this date under #13 is superseded by this entry.
    `docs/wiki.md` and the design spec were corrected in place; the earlier entry stands as history,
    since this log is append-only.
- **Scaffolding, milestone 1 (#2).** `pyproject.toml` dependencies, `src/futseg` package skeleton,
  `tests/`, `ruff` and `pytest` configuration, `uv.lock` committed.
  Decisions:
  - **`hatchling` + `src/` layout.** Any PEP 517 frontend builds it and `src/` forces tests to
    import the installed package rather than the working tree. Chosen over `uv_build` purely for
    boringness: no contributor or container has to know a uv-specific backend exists.
  - **No `[project.scripts]` entry yet.** It would point at `futseg.cli:app`, which does not exist
    until #7 — an entry point that installs cleanly and crashes on first invocation is worse than
    no entry point. Deferred to the CLI milestone.
  - **`opencv-python` is overridden out of the graph.** `ultralytics` depends on the GUI build
    transitively, so declaring `opencv-python-headless` does not displace it; both were installed
    at once. They own the same `cv2/` directory, so which build wins is install-order dependent,
    and removing `opencv-python` deleted `cv2` out from under the headless distribution — observed
    directly, not theorised. Fixed with
    `[tool.uv] override-dependencies = ["opencv-python; sys_platform == 'never'"]`, verified from a
    freshly recreated venv. Rejected: leaving both installed (non-reproducible, and ~70 MB of Qt5
    shared objects land in any image built from it) and pinning ultralytics (punishes the wrong
    dependency for a packaging conflict upstream of it).
  - **CUDA verification asserts a real device op, not just `torch.cuda.is_available()`**, which
    returns `True` on a wheel with no kernels for the installed GPU architecture and fails only
    once work runs. Issue #2's checklist was updated to match — the issue's own rationale is that
    this failure mode is silent, and `is_available()` alone preserves the silence.
  - **Version floors only where they encode a requirement**: `torch>=2.13` (CUDA-bundled Linux
    wheel with current-architecture kernels) and `numpy>=2` (keeps opencv and torch on the same
    side of the ABI split). Everything else is unconstrained, with `uv.lock` supplying
    reproducibility.
  - **`tests/test_scaffolding.py` imports every scaffold module.** An empty `tests/` cannot be
    committed at all, and a `tests/` with no tests makes `uv run pytest` exit 5 ("no tests
    collected"), which reads like a failure in CI. Parametrised imports also catch a missing
    `__init__.py` immediately rather than at the milestone that first imports the module.
  - Corrected two stale items in issue #2 before starting: the `.gitattributes` checkbox (delivered
    by #16) and a platform note still requiring the repo to live off `/mnt/c/` (superseded by #18).
  - **Added `scripts/cuda_check.py`** (requested during review of #2, folded into the same PR since
    it delivers that issue's "assert CUDA actually resolved" item). It reports the wheel's compiled
    arch list against the device's compute capability, then runs real work: fp32 matmul verified
    against CPU, fp16 matmul, cuDNN conv. Kept out of `tests/` deliberately — it requires a GPU, and
    a test that cannot run on the CI runner belongs behind `@pytest.mark.slow` at best, whereas this
    is a diagnostic a human runs when a machine misbehaves. `scripts/` is a new top-level directory
    not in `PLAN.md`'s tree.
    Note for future readers: the first version of that script reported `inf` for the fp16 matmul and
    briefly looked like a hardware fault. It was the script's own bug — `256**3` overflows fp16's
    ~65504 ceiling, so the accumulation saturated. Reducing in fp32 fixed it. Recorded in
    `docs/wiki.md` because the same trap will reappear in half-precision diffusion work.
  - **Added a `Makefile` as the developer entry point.** Not ergonomics — a correctness guard. The
    `UV_PROJECT_ENVIRONMENT` export added to `~/.bashrc` turned out not to hold: Ubuntu's `.bashrc`
    returns at its interactive guard before reaching an appended line, so non-interactive shells
    never see it, and a terminal opened before the edit never had it either. A `uv run` in such a
    shell does not warn about the missing variable; it silently built a 4.1 GB `.venv` in-tree on
    drvfs, with hardlinking disabled across filesystems, and removed whatever `.venv` was already
    there — the exact Windows/Linux collision #18 documented, reproduced by accident.
    The `Makefile` sets the variable itself, so the environment is a property of the command rather
    than the shell. Its `guard` prerequisite refuses to run off Linux, checks `uv` is on `PATH`, and
    warns when a stray in-tree `.venv` exists; `clean-venv` removes only that stray and never the
    real environment.
    Rejected: `direnv` (correct, but adds a per-machine install to the setup path for one variable)
    and moving the source into WSL2 (already decided against in #18 for the IDE).
- **Moved development into a GPU-enabled container (#20).** This reverses a recorded deferral: both
  `CLAUDE.md` and the Linux-first design doc placed Docker at milestone 10 (#14), "deliberately last
  and deliberately underspecified — their shape depends on the CLI existing first". The reversal was
  the project owner's call, taken after the WSL2 arrangement failed twice in one session, and is
  recorded here rather than left as a silent disagreement between the repo and its own docs.
  Trigger: the `UV_PROJECT_ENVIRONMENT` export in `~/.bashrc` did not hold (Ubuntu's `.bashrc`
  returns at its interactive guard before an appended line), so `uv run` in an ordinary terminal
  silently built a 4.1 GB `.venv` in-tree on drvfs and removed the one already there. The Makefile
  guard added hours earlier narrowed the window but could not close it, because the IDE's uv
  integration invokes uv directly.
  Decisions:
  - **The container is the environment.** No host virtualenv, no host Python, nothing to activate.
    The class of bug behind both incidents — correctness depending on unverified host state — is
    removed rather than guarded against.
  - **Base is plain `ubuntu:26.04`, not `nvidia/cuda`.** torch's wheels bundle the CUDA runtime and
    the container runtime injects the host driver, so a CUDA base would ship a second copy of
    multi-GB libraries. Verified before committing to it: `nvidia-smi` runs in a stock `ubuntu:26.04`
    container under `--gpus all`.
  - **Python pinned through uv (`UV_PYTHON=3.12`), not inherited from the distro**, so bumping the
    base OS cannot move the interpreter out from under `uv.lock` and change wheel selection for
    every compiled dependency.
  - **Virtualenv at `/opt/venv`, outside the bind mount.** An in-tree `.venv` inside a bind mount is
    written back to the host, which is the same collision in a new costume.
  - **The Linux-first design doc was deleted rather than kept with a "superseded" header.** The
    owner chose a clean end state; the file remains in git history, and its surviving decisions
    (no custom torch index, the opencv-python override, XDG cache paths, LF endings, device.py as
    the only CUDA probe) are restated in the new document so it stands alone.
  - **#14 is not superseded.** The runtime/distribution image is a different artefact: no source
    mount, weights mounted not baked, minimal surface.
  - **Where the working tree lives is deliberately still open**, to be decided once bind-mount I/O
    cost can be measured rather than guessed.
  Measured after the first build, both recorded in `docs/wiki.md`:
  - `docker images` reported **17.8 GB** for the image where `docker image inspect` reports
    **5.71 GB** — buildkit's attestation and manifest-list entries get counted by the former. The
    real content is ~5.4 GB of virtualenv (2.7 GB `nvidia`, 1.2 GB `torch`, 0.7 GB `triton`), which
    is close to the floor for a CUDA-capable torch environment, so no size work was warranted. The
    inflated figure nearly triggered an unnecessary optimisation pass.
  - `XDG_CACHE_HOME=/cache` also relocates **uv's wheel cache** into the named volume (5.2 GB at
    `/cache/uv`), duplicating wheels already installed in the image. Accepted rather than split into
    a second volume: both are caches, both reconstructible, and one `make clean-cache` is easier to
    reason about than two.
  - **The image installs the project editable, not just its dependencies.** Caught while answering
    how to point an IDE at the container: `uv sync --frozen --no-install-project` leaves `futseg`
    absent from `/opt/venv`, so `import futseg` fails for anything that does not go through
    `uv run` — an IDE interpreter, a debugger, a profiler. `make test` hid this, because `uv run`
    reinstalls the project on every invocation into a container layer that `--rm` discards.
    Fixed with a second, cheap layer (`COPY src` + `uv sync --frozen`) below the dependency layer.
    Because the install is editable against `/workspace/src` and the bind mount supplies that path,
    **code changes still require no rebuild** — verified by writing a file on the host and reading
    it back from a fresh container, and by importing a module that did not exist at build time.
    Rejected: `PYTHONPATH=/workspace/src`, which needs no COPY at all but discards distribution
    metadata and console scripts, and contradicts the src-layout rationale already recorded here.
  - **Makefile grew container lifecycle targets** (`up`, `down`, `restart`, `ps`, `logs`, `rebuild`,
    `exec`), grouped in `make help` by section. Task targets now route through one helper that
    `exec`s into a resident container when `make up` has been run and falls back to a one-off
    `run --rm` otherwise. Chosen over two parallel sets of targets (one for each mode), which would
    double the surface to document and let the two drift; and over always using `run --rm`, which
    would ignore the container an IDE is attached to. Verified in both modes, including a
    before/after count of running containers to confirm the resident one is genuinely reused rather
    than a second being spawned silently.

## 2026-07-26

- **Core abstractions and I/O (#3), milestone 2.** `Segmenter`/`Inpainter` protocols, `io.py`,
  `device.py`, `paths.py`, `masking.py`, built test-first: every test was watched failing for the
  right reason (missing symbol, not a typo) before the module existed.
  Decisions:
  - **The halo invariant is enforced in `masking.py`, not left to callers.** `derive_masks` raises
    `ValueError` when `inpaint_grow <= composite_shrink + feather_radius` rather than returning
    subtly wrong masks. At equality the feathered composite edge reaches exactly as far as the
    regenerated region, so original pixels show through the partially transparent border — a defect
    that is invisible in a unit test asserting shapes and obvious only on a real photo. Making it an
    error means no backend can get it wrong quietly.
  - **`union` is a per-pixel maximum, not a sum.** Overlapping instances in a multi-person photo
    would exceed 1.0 and clip into a hard-edged blob.
  - **`resolve_cache_dir` uses `Path.absolute()` rather than `.resolve()`.** Absolute is the point —
    a relative override would drop weights beside the CWD — but resolving symlinks would rewrite
    paths the caller passed in, which surprises anyone pointing the cache at a symlinked volume.
  - **`configure_caches` sets `HF_HOME` with `setdefault`.** The container already points it at the
    mounted `/cache` volume; clobbering it would send multi-GB downloads somewhere unmounted.
  - **`load_image` always converts to RGB.** Greyscale or transparent input would otherwise reach a
    segmenter with the wrong channel count, and alpha in the *input* is unrelated to the alpha this
    pipeline derives.
  - **No validation on `resolve_device`'s preference string.** The issue specifies the function
    exactly; the CLI is the trust boundary and will constrain the choice. Adding a second check here
    would duplicate policy the architecture puts in one place.
  - A test asserts `inspect.signature(Inpainter.inpaint)` is exactly `(self, image, mask)`, so the
    "no prompt argument" rule fails a test rather than only a review comment.
- **Segmentation fast tier (#4), milestone 3.** `segmentation/yolo.py` implementing `Segmenter` with
  YOLO11-seg, person-class filtering and instance union. Built test-first with an injected fake
  model, so the fast suite needs no weights.
  Decisions:
  - **Person filtering happens on `boxes.cls` in our code, not through ultralytics' `classes=`
    predict argument.** The kwarg would be marginally faster, but the guarantee would then live in a
    library keyword whose behaviour could change without us noticing, and a test asserting "we
    passed `classes=[0]`" proves nothing about the output. Filtering explicitly is what the tests
    can exercise. Rejected doing both as belt-and-braces: two mechanisms for one rule.
  - **`imgsz` defaults to 1280 rather than ultralytics' 640.** `PLAN.md` calls for a larger input on
    this tier because the 1/4-stride prototype masks are the binding limit; the larger input is the
    only lever that moves.
  - **Masks are resized to the image if the model returns another resolution.** `retina_masks=True`
    should already return original-resolution masks, but a silent shape mismatch would corrupt the
    two-mask derivation downstream, and the resize is one line.
  - **The slow test uses `ultralytics.utils.ASSETS / "bus.jpg"`.** It ships inside the installed
    package and contains people, so there is no image fixture to commit and no licensing or privacy
    question in a public repo. It asserts people are actually found, so the mask-parsing path is
    exercised rather than the empty-result shortcut a synthetic image would take.
  Verified against the real library, not only the fake: `yolo11n-seg.pt` downloaded into
  `/cache/futseg/weights/` and `find /workspace -name '*.pt'` came back empty — `paths.py` does keep
  ultralytics out of the working directory, which until now was only an intention.
- **Specified `futseg segment` as a first-class command (#24).** Design only; no CLI code, which
  stays in #7.
  Trigger: a request for "a script to segment an arbitrary photo". I wrote one
  (`scripts/segment.py`, argparse, five artefacts) and it was scrapped in favour of putting the
  capability in the product CLI instead — correctly. The script would have duplicated `futseg
  segment` with a second argument parser, a second set of defaults and no tests, and the two would
  have drifted the moment #7 landed. `scripts/cuda_check.py` is not a counter-example: it diagnoses
  the *machine*, and nothing in the product will ever do that.
  Decisions:
  - **`segment` is a command, not a debug flag.** The plan called it "mask-only debug output";
    segmenting without inpainting is how mask edge quality is judged and is useful on its own, so it
    is specified as a product surface with defined artefacts and exit codes.
  - **Five artefacts, and the overlay earns its place.** Alpha and the two derived masks are the
    pipeline's actual intermediates; the cutout shows the composite alpha in use. The overlay —
    background tinted, subject untouched — is the only view where the boundary is seen against the
    photograph rather than against black, which is where hair and fingers actually expose a coarse
    mask.
  - **Exit code 1 when no person is found.** An empty mask is a plausible-looking success otherwise,
    and this tool has to be usable non-interactively.
  - **`input/` is tracked as a directory, contents gitignored.** The repo is public and photographs
    of real people must never enter it; `.gitkeep` means the directory still exists in a fresh clone.
  - Recorded on #7 that `paths.configure_caches()` has no production caller yet, so the cache policy
    is defined but never applied — the CLI is where that gets wired.
- **SAM2 refinement, the default tier (#10), milestone 4.** `segmentation/refined.py`: YOLO11
  detection on the person class, each box a SAM2 prompt, instance masks unioned. Plus `metrics.py`
  with boundary IoU. Built test-first with both models injected.
  The issue said to confirm the ultralytics SAM2 API rather than assume it, which was worth doing —
  two of the three findings would have become bugs:
  - **`result.boxes.cls` from SAM2 is a prompt ordinal, not a COCO class.** It returns
    `[0., 1., 2., 3.]`, one per prompt. Filtering that against `PERSON_CLASS` — exactly the pattern
    `yolo.py` uses correctly on *detection* output — would have silently kept only the first person
    in every multi-person photo, with no error anywhere. `test_refined.py` has a dedicated test so
    copying the pattern across fails loudly.
  - **SAM2 masks arrive as `dtype=bool`**, where YOLO11-seg returns float. The protocol requires
    float32, so the conversion is explicit.
  - **`SAM("sam2_t.pt")` downloads into the CWD** like `YOLO(...)` does; a probe with a bare
    filename dropped 78 MB into the repo root (gitignored, so harmless, but it proves the rule).
  Decisions:
  - **Person filtering happens before prompting, never after.** Prompting SAM with a dog's box and
    discarding the result afterwards would waste a model call and invite filtering SAM's output,
    which is the ordinal trap above.
  - **No SAM call at all when no person is detected.** Loading and running a model to produce
    nothing is worth one branch.
  - **`sam2_b.pt` as the default checkpoint**, configurable. The probe used `sam2_t.pt`; base is the
    balance for a tier whose whole purpose is edge quality.
  Measured, rather than asserted, on `bus.jpg` (810x1080): plain IoU between the two tiers 0.9011,
  boundary IoU 0.8466. The tiers agree on the bulk and disagree at the rim, which is the premise the
  milestone rests on — but the gap is modest at this resolution, and the ~25px quantization argument
  in `PLAN.md` applies to multi-thousand-pixel photos this repository cannot carry. Recorded as a
  measurement, not as proof that refinement is better: there is no ground truth here.
  Not done, and flagged on the issue: **hand-annotated boundary-quality fixtures.** They require
  either photographs of real people in a public repo, which the project rules forbid, or synthetic
  data that would not measure real segmentation quality. Boundary IoU itself is implemented and
  tested against synthetic shapes with known answers; the fixtures need a decision first.
  Follow-up on the fixtures question: **local-only, in a gitignored `fixtures/`.** The framing in
  the PR was wrong and was corrected — privacy was never the blocker. The photo never needs
  committing (`bus.jpg` ships inside the installed `ultralytics` package, and the repo tracks zero
  images), and a hand-annotated ground-truth *mask* is a binary silhouette, not a photograph. The
  real distinction is narrower: a gitignored fixture is visible to one machine, so a metric reading
  one is a local tool rather than a regression gate; committing it is what would make it
  CI-checkable. Deferred because there is no CI yet (#15) and no hand annotation exists — and
  scoring a segmenter against ground truth produced by a segmenter would be circular, so generating
  one was not an option.
- **Composite backend + end-to-end pipeline (#5), milestone 5.** `inpaint/composite.py` (solid
  colour, blur, static image) and `pipeline.py`. First milestone that emits a finished image.
  Decisions:
  - **Three small backend classes rather than one with a mode flag.** Each is a handful of lines with
    no branching, and the `Inpainter` protocol already makes them interchangeable; a mode enum would
    add a conditional to save two class statements.
  - **The anti-halo guarantee is now a test, not a doctrine.** `test_pipeline.py` fills the generated
    background with a colour that appears nowhere in the input, then asserts every pixel the subject
    does not cover is exactly that colour. A single surviving original pixel outside the subject
    fails it. That is the cheap-cutout rim made arithmetic instead of a thing to squint at.
  - **The pipeline never inspects its backends.** It takes a `Segmenter` and an `Inpainter` and
    composites the original subject over whatever came back, so the person is untouched by the
    backend even at the boundary — the invariant holds for a diffusion model exactly as it does for
    a flat fill.
  - **Defaults `k=12, j=3, feather=5`**, tuned on a real photo. Only `j` and `feather` are visible in
    the output: `k` governs what is regenerated *underneath* the composited subject, so it only has
    to clear `j + feather` with margin. Measured on `bus.jpg` at four settings, zero leaked pixels at
    all of them — as the guard guarantees — while `j` costs subject area: 7.5% at j=2, 10.7% at j=3,
    21.3% at j=6.
  - Validated by looking, not only by asserting: seam crops located from the mask itself rather than
    guessed at, rendered against magenta and against blur. No stale rim; the boundary sits where the
    original's does.
  Two limitations recorded in `docs/wiki.md` rather than silently accepted:
  - **`BlurInpainter` blurs the subject too**, so subject colour smears outward near the silhouette.
    A soft fringe that is easy to mistake for a halo when eyeballing. Accepted for a dev-loop
    backend; `SolidColorInpainter` is the unambiguous instrument.
  - **`k`/`j`/`feather` are absolute pixels, so behaviour is resolution-dependent.** `j=3` trims ~10%
    of subject area at 810x1080 and would be negligible at 4000px. Tuned on the small image, which
    errs conservative, but scaling with the image diagonal is unresolved and belongs with #7.
- **Generative inpainting backend (#6), milestone 6.** `inpaint/diffusion.py`: a `ModelSpec`
  registry with per-model `to_kwargs` adapters, and `DiffusionInpainter` implementing `Inpainter`.
  Both the issue and `PLAN.md` insisted the kwargs be verified against the installed `diffusers`
  rather than assumed. Doing so corrected two errors in the plan and one of my own:
  - **`black-forest-labs/FLUX.2-klein` does not exist.** The real repo is `FLUX.2-klein-4B`
    (Apache-2.0, ungated, ~22 GB). Sibling naming is a trap: `FLUX.2-klein-9B` is gated and
    `license:other`, despite being the same family.
  - **`Flux2Pipeline` has no `mask_image` parameter** — it is instruction-driven whole-canvas
    editing. The plan named it as the FLUX.2 entry, which would have regenerated the entire frame
    and ignored the inpaint mask. The correct class is `Flux2KleinInpaintPipeline`, which does take
    `mask_image` and `strength`. I briefly concluded FLUX.2 could not inpaint at all before finding
    that class; the first conclusion was wrong because I had probed only `Flux2Pipeline`.
  - **`stabilityai/stable-diffusion-2-inpainting` no longer exists.** Replaced with
    `stable-diffusion-v1-5/stable-diffusion-inpainting`, still comparison-only.
  - The plan expected FLUX.2 and SDXL to have incompatible signatures, making `to_kwargs`
    load-bearing. They actually agree on the core (`prompt`/`image`/`mask_image`/`strength`/
    `guidance_scale`); they differ only in extras. The adapter stays per-model, but the plan
    overstated the gap.
  Decisions:
  - **The registry is verified without downloading anything.** One test asserts each `pipeline_cls`
    exists in the installed `diffusers`; another asserts every kwarg an adapter emits appears in
    that class's real `__call__` signature. Both failure modes otherwise surface only after
    multi-GB weights have downloaded and loaded — the most expensive possible moment.
  - **Never upscale to the native canvas.** `_fit_within` scales down only; enlarging a small photo
    to 1024 invents detail and spends generation time for nothing.
  - Nothing has been run against real weights yet: the smallest usable checkpoint is several GB and
    no tiny inpaint pipeline exists on the Hub (only tiny text2img ones), so that is a deliberate,
    disclosed gap rather than an oversight.
  - **`ModelSpec` gained `variant` and `gated`.** Found while comparing Stable Diffusion options:
    `from_pretrained(repo, torch_dtype=float16)` does **not** fetch fp16 weights — it downloads the
    fp32 files and casts them in memory. `variant="fp16"` is what selects the small download. For
    `sdxl-inpaint` that is 6.5 GB against roughly 13 GB, so the omission was a silent doubling of
    every download for repos that publish fp16. The variant is per-repo (FLUX.2-klein-4B publishes
    none, and requesting one that does not exist fails the load), which is exactly the kind of fact
    the registry should carry rather than the loading code assume. `gated` is declared for the same
    reason: a gated entry cannot load without a Hub token, and that should be visible in the
    registry rather than discovered as a 401 after the user selects it.
    Real fp16 download sizes, measured from the Hub file listing rather than the repo totals which
    include fp32 duplicates: sd15-inpaint 2.6 GB, sdxl-inpaint 6.5 GB, flux2-klein-4B 14.9 GB.
- **CLI (#7), milestone 7.** `cli.py`: a typer app with `futseg run` and `futseg segment`, plus the
  `[project.scripts]` entry point deferred from #2 — which now points at something real.
  Decisions:
  - **Backends are constructed in exactly two functions** (`_build_segmenter`, `_build_inpainter`),
    imported lazily so `--help` loads no models. Tests monkeypatch those two, which is why the CLI
    suite needs no weights at all; it exercises artefact naming, exit codes and flag plumbing rather
    than model behaviour, which is already covered where the models live.
  - **Exit codes are contract, not decoration**: `0` success, `1` no person found, `2` usage error.
    An empty mask is never success. Errors go to stderr so stdout stays parseable when piped —
    caught by a test that initially asserted against stdout and failed for the right reason.
  - **The `k > j + feather` guard is checked in the CLI too**, producing a usage error rather than
    letting `derive_masks` raise a traceback at the user. The guard stays in `masking.py` as the
    real enforcement; this is only the friendly surface.
  - **No `make segment` / `make run` targets.** `make exec CMD="futseg segment input/x.jpg"` already
    runs it in the container, and a wrapper would be a second place for defaults to drift — the same
    mistake as the scrapped `scripts/segment.py`.
  Verified through the installed console script, not just `CliRunner`: `which futseg` resolves to
  `/opt/venv/bin/futseg`, `futseg segment` writes all five artefacts, `futseg run --backend
  composite` produces an image, a person-free photo exits 1, and a missing file exits 2.
- **Mask hole filling (#28, part one).** `masking.fill_holes`, wired into both segmenters and on by
  default. Raised to priority: high at the project owner's request after the FLUX.2 run made the
  defect the visible limiter on output quality rather than a footnote.
  Decisions:
  - **The fill is bounded by area, and that bound is the design, not a tuning knob.** The obvious
    implementation — flood-fill from the frame border, treat everything unreachable as subject — is
    wrong: a hand on a hip encloses genuine background that the border cannot reach either, so
    connectivity alone would weld the arm to the torso. Only regions below `max_hole_ratio` of the
    subject's own area are filled. There is a dedicated test for the large-enclosed-gap case, which
    is the one that fails a naive implementation.
  - **The threshold is a fraction of subject area, not an absolute pixel count**, because the same
    16 px is noise on a 40 MP photo and a real gap on a thumbnail. A test pins that the same hole
    resolves differently at two image scales.
  - **Applied after the resize**, so the threshold is measured at output scale rather than at
    whatever resolution the model happened to return.
  - **Soft alpha is preserved**: only pixels inside a filled hole are written, so the feathered
    values the deferred matting stage will produce survive untouched. Tested.
  Measured on the photographs that exposed the defect, not on synthetic shapes alone:
  IMG_7170 recovered 15,757 px (1.501% of subject) at IoU 0.9852 against the raw mask; IMG_7053
  recovered 274 px (0.042%) at IoU 0.9996. Surgical, not a blob — and confirmed visually: the
  raised hand, the glasses lenses and the sleeve speckle fill, while the seatbelts and the gaps
  between people stay open.
  **Half of #28 remains open.** Worn occluders — seatbelts, bag straps — cross the silhouette and
  reach the frame edge, so they are legitimately reachable and hole filling cannot touch them. That
  needs a different approach and the issue stays open for it.
