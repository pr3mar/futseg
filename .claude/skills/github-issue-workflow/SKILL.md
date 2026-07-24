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

## Preconditions — check these before step 1

Two environment facts that will otherwise fail you mid-workflow, after you've already made commits.

**1. Push over HTTPS, not SSH.** The user's SSH key is passphrase-protected and no `ssh-agent` is
reachable from a non-interactive assistant shell (`SSH_AUTH_SOCK` is unset, and there's no TTY to
prompt on), so `git push` over `git@github.com:` fails with `Permission denied (publickey)` even
though the key is valid and offered. The Windows `ssh-agent` service is disabled and would not help
anyway — `git` here resolves to Git Bash's `/usr/bin/ssh`, which speaks `SSH_AUTH_SOCK`, while the
Windows agent uses a named pipe.

    git remote -v            # confirm origin is https://github.com/pr3mar/futseg.git
    gh auth status           # needs 'repo' scope; gh's credential helper does the auth

If `origin` is on SSH, switch it: `git remote set-url origin https://github.com/pr3mar/futseg.git`.
SSH works fine from the *user's* own terminal — this is a limitation of the assistant's shell only,
so don't "fix" the user's setup beyond the remote URL.

**2. `origin/main` must already exist.** Step 2 branches from it and a PR needs it as a base.

    git ls-remote --heads origin main

If that returns nothing, the repo has never been pushed to and **this workflow cannot run** —
bootstrapping requires an initial commit on `main`, which the hard rule below forbids. That is a
genuine deadlock, not a judgment call: **stop and ask the user.** Do not quietly decide the rule
doesn't count "just this once." If the user directs the bootstrap, record the waiver in
`transcript.md` and disclose it in the PR body so it never reads as an undisclosed violation.

## The Two Hard Rules

**`main` is off limits, always. You are not allowed to merge PRs, ever.**

These are not defaults to override when it seems safe — there is no "safe" exception. The empty-repo
case above is not an exception either: it is a *stop and ask*, and only the user can lift it.

| Excuse | Reality |
|---|---|
| "It's a one-line typo fix" | One-line changes still go through a branch + PR. |
| "The PR is obviously correct, merging saves the user a click" | Merge approval is the user's decision, not a formality you can skip. |
| "I'm just syncing main locally" | Fetch/pull-to-track is fine; committing or pushing to main is not. |
| "No one will notice" | Irrelevant — the rule has no exception clause. |
| "The repo is empty so somebody has to commit to main" | True, and that somebody is the user. Stop and ask — see Preconditions. |

**Red flags — stop immediately if you're about to:**
- `git checkout main` followed by any edit/commit
- `git push origin main`
- `gh pr merge` in any form (including `--squash`, `--admin`, `--auto`)

If you catch yourself justifying any of these, stop and ask the user instead.

## Workflow

1. **Get an issue.**
   - *Given a ticket* (number or URL passed as input): `gh issue view <n> --repo pr3mar/futseg`. Confirm it's open and not already assigned to/in-progress by someone else — if it's closed, already claimed, or the ticket doesn't otherwise match what you're being asked to do, that's a contradiction: flag it (see global principle below) instead of proceeding on a guess.
   - *No ticket given:* `gh issue list --repo pr3mar/futseg` (or check the project board), prefer unassigned issues, avoid ones already in progress.
   - *No issue fits the work at all* (refactors, doc/plan revisions, repo hygiene): **create one first**, then work it. Every PR must close an issue, so silently opening one without `Closes #<n>` is not the fallback — filing the missing issue is. Say so when you do it.
   - **Don't assume issue N maps to milestone N−1.** Issues filed later can sit early in the sequence; check `CLAUDE.md` for the current mapping table.
   - Either way, assign yourself: `gh issue edit <n> --add-assignee @me`.
2. **Branch from an up-to-date main.** `git fetch origin && git checkout -b issue-<n>-<short-slug> origin/main`. Never branch from a locally-modified main.
3. **Do the work** per this repo's conventions (Python 3.12, `uv`, `ruff`, `pytest` — see `CLAUDE.md`).
4. **Log decisions as you go, not at the end.** Append to `transcript.md` (repo root) whenever you make a non-obvious choice: what you decided, why, and alternatives you rejected. This is a running log — append, don't rewrite history. Never write the private value you removed *into* the log while recording its removal; record the decision, not the secret.
5. **Before opening the PR:**
   - Update `docs/wiki.md` with anything future-you or another agent would need: new conventions, gotchas, or decisions this issue introduced. This is cumulative project knowledge, not a per-PR changelog — check whether existing entries still hold before adding new ones.
   - Verify portability: a fresh `uv sync` + `uv run pytest` (and `uv run ruff check .`) succeed from a clean checkout — no hardcoded local paths, no machine-specific assumptions.
     **If those can't run yet** (pre-scaffolding: no dependencies, source, or tests exist), say that explicitly in the PR's Testing section instead of omitting it. An empty Testing section reads as "verified"; "not runnable yet, here is what I checked instead" is the honest form.
   - **Check for contradictions.** Compare what you're about to ship against `CLAUDE.md`, `docs/wiki.md`, and the issue text itself. Flagging contradictions is a global standing principle (`~/.claude/CLAUDE.md`), not repeated here — this bullet is just the reminder of *when* to run that check in this workflow.
   - **Check the issue you're closing is still accurate.** If the work changed the design the issue describes, fix the issue body too — a stale issue misleads the next person as much as a stale doc.
6. **Commit** with messages referencing the issue (`#<n>`).
   A `PreToolUse` anonymization hook scans `git commit` (staged diff) and `git push` (everything unpushed) for hardware specs, credential-shaped strings, and local home paths. If it blocks you it names the rule and file but withholds the matched value — go look at the file. Treat a block as a real finding, not an obstacle to route around.
7. **Open the PR**, not a merge: `gh pr create --title "..." --body-file <file>`. Use `--body-file` rather than an inline `--body`: PR bodies contain backticks, quotes and code blocks that break shell quoting, and a heredoc will bite you eventually. The body must include `Closes #<n>` and disclose every change made, not just a summary — a reviewer should be able to tell what happened without reading the diff. Disclose anything unusual (a rule waiver, a change made outside the diff such as edited issues or renamed milestones).
8. **Stop.** Leave the PR for the user to review and merge. Do not `gh pr merge`, do not push follow-up commits to `main`.

## Quick Reference

| Situation | Action |
|---|---|
| Before anything | `git ls-remote --heads origin main` + confirm `origin` is HTTPS (see Preconditions) |
| Repo empty / no `origin/main` | Stop and ask — bootstrapping needs a commit on `main`, which you may not make |
| `Permission denied (publickey)` on push | Expected; use the HTTPS remote + `gh` credential helper, don't chase the SSH key |
| Given a ticket number/URL | `gh issue view <n>`, verify it's open and unclaimed, then work it |
| No ticket given | pick from `gh issue list` / project board, prefer unassigned |
| No issue fits the work | create one first, then work it — never a PR without `Closes #<n>` |
| Starting an issue | branch from `origin/main`, never local `main` |
| Made a non-trivial decision | append to `transcript.md` now, not later |
| About to open a PR | update `docs/wiki.md` first, check for contradictions (global principle) |
| Verification can't run yet | say so explicitly in the PR's Testing section; don't leave it blank |
| Anonymization hook blocked the commit | real finding — open the named file and redact; don't work around it |
| PR body has backticks/code blocks | write it to a file, use `gh pr create --body-file` |
| PR is ready | `gh pr create` with `Closes #<n>` and a full changes list |
| PR looks done and correct | still don't merge it — that's the user's call |
| Found conflicting instructions | stop, quote both, flag to the user — see `~/.claude/CLAUDE.md` |
