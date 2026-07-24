---
name: github-issue-workflow
description: Use when picking up a futseg GitHub issue to work on, or before creating a branch, committing, opening a PR, or merging/pushing to main in this repo
---

# GitHub Issue Workflow (futseg)

## Overview

The procedure for taking a `pr3mar/futseg` issue from "open" to "PR ready for review," with two hard git rules and two traceability requirements that are easy to skip under time pressure. Skipping any of them is not a shortcut — it's the failure mode this skill exists to prevent.

## When to Use

- Starting work on any GitHub issue in this repo
- About to run `git branch`, `git commit`, `git push`, or `gh pr create`/`gh pr merge`
- About to close out work and hand it back to the user

## Input

Accepts an optional ticket to work on: an issue number (`42`) or a full issue URL. Pass it as the
skill argument, e.g. `github-issue-workflow 42`.

- **Given:** skip picking from the backlog — go straight to step 1's "given a ticket" branch.
- **Not given:** pick one per step 1's "no ticket given" branch.

## The Two Hard Rules

**`main` is off limits, always. You are not allowed to merge PRs, ever.**

These are not defaults to override when it seems safe — there is no "safe" exception.

| Excuse | Reality |
|---|---|
| "It's a one-line typo fix" | One-line changes still go through a branch + PR. |
| "The PR is obviously correct, merging saves the user a click" | Merge approval is the user's decision, not a formality you can skip. |
| "I'm just syncing main locally" | Fetch/pull-to-track is fine; committing or pushing to main is not. |
| "No one will notice" | Irrelevant — the rule has no exception clause. |

**Red flags — stop immediately if you're about to:**
- `git checkout main` followed by any edit/commit
- `git push origin main`
- `gh pr merge` in any form (including `--squash`, `--admin`, `--auto`)

If you catch yourself justifying any of these, stop and ask the user instead.

## Workflow

1. **Get an issue.**
   - *Given a ticket* (number or URL passed as input): `gh issue view <n> --repo pr3mar/futseg`. Confirm it's open and not already assigned to/in-progress by someone else — if it's closed, already claimed, or the ticket doesn't otherwise match what you're being asked to do, that's a contradiction: flag it (see global principle below) instead of proceeding on a guess.
   - *No ticket given:* `gh issue list --repo pr3mar/futseg` (or check the project board), prefer unassigned issues, avoid ones already in progress.
   - Either way, assign yourself: `gh issue edit <n> --add-assignee @me`.
2. **Branch from an up-to-date main.** `git fetch origin && git checkout -b issue-<n>-<short-slug> origin/main`. Never branch from a locally-modified main.
3. **Do the work** per this repo's conventions (Python 3.12, `uv`, `ruff`, `pytest` — see `CLAUDE.md`).
4. **Log decisions as you go, not at the end.** Append to `transcript.md` (repo root) whenever you make a non-obvious choice: what you decided, why, and alternatives you rejected. This is a running log — append, don't rewrite history.
5. **Before opening the PR:**
   - Update `docs/wiki.md` with anything future-you or another agent would need: new conventions, gotchas, or decisions this issue introduced. This is cumulative project knowledge, not a per-PR changelog — check whether existing entries still hold before adding new ones.
   - Verify portability: a fresh `uv sync` + `uv run pytest` (and `uv run ruff check .`) succeed from a clean checkout — no hardcoded local paths, no machine-specific assumptions.
   - **Check for contradictions.** Compare what you're about to ship against `CLAUDE.md`, `docs/wiki.md`, and the issue text itself. Flagging contradictions is a global standing principle (`~/.claude/CLAUDE.md`), not repeated here — this bullet is just the reminder of *when* to run that check in this workflow.
6. **Commit** with messages referencing the issue (`#<n>`).
7. **Open the PR**, not a merge: `gh pr create --title "..." --body "Closes #<n>\n\n## Changes\n- ...\n\n## Testing\n- ..."`. The body must disclose every change made, not just a summary — a reviewer should be able to tell what happened without reading the diff.
8. **Stop.** Leave the PR for the user to review and merge. Do not `gh pr merge`, do not push follow-up commits to `main`.

## Quick Reference

| Situation | Action |
|---|---|
| Given a ticket number/URL | `gh issue view <n>`, verify it's open and unclaimed, then work it |
| No ticket given | pick from `gh issue list` / project board, prefer unassigned |
| Starting an issue | branch from `origin/main`, never local `main` |
| Made a non-trivial decision | append to `transcript.md` now, not later |
| About to open a PR | update `docs/wiki.md` first, check for contradictions (global principle) |
| PR is ready | `gh pr create` with `Closes #<n>` and a full changes list |
| PR looks done and correct | still don't merge it — that's the user's call |
| Found conflicting instructions | stop, quote both, flag to the user — see `~/.claude/CLAUDE.md` |
