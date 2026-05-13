---
description: Fetch upstream, merge/rebase safely, verify the Claude Code harness still works, and summarize what's new.
argument-hint: [agent|webui]
allowed-tools: Bash, Read, Edit, Grep, Glob, TaskCreate, TaskUpdate, TaskList
---

You are running the `/sync-upstream` workflow. The goal: pull upstream changes into the fork, prove the Claude Code harness still works, and produce a summary the user can act on. Do not push, do not restart services, do not run the full test matrix — those are follow-ups the user decides on.

## Target repo

Parse `$ARGUMENTS`:
- empty or `agent` → `/Users/luca/dev/hermes-agent` (fork of `NousResearch/hermes-agent`, upstream branch `main`, our branch `main`, merge strategy)
- `webui` → `/Users/luca/dev/hermes-webui` (fork of `nesquena/hermes-webui`, upstream branch `master`, our branch `master`, rebase strategy)
- anything else → stop and ask which one they meant

Store the path as `$REPO`, the merge strategy as `$STRATEGY` (merge|rebase), and the upstream ref as `$UPSTREAM_REF` (e.g. `upstream/main`). All subsequent shell commands should `cd "$REPO"` or use `git -C "$REPO"`.

## Task tracking

Create tasks with TaskCreate for: Pre-flight, Fetch + preview, Merge/rebase, Post-merge deps, Harness integrity, Summarize. Mark in_progress before each step, completed after. If a step fails, keep it in_progress and stop — do not silently roll past failures.

## Step 1 — Pre-flight

Bail out cleanly if any of these fail. Report the exact failing check:

- `git -C "$REPO" status --porcelain` → must be empty. If not, list the dirty files and stop. The user needs to commit/stash before syncing; don't do it for them.
- `git -C "$REPO" rev-parse --abbrev-ref HEAD` → must be `main` (agent) or `master` (webui). If on another branch, stop — the user is mid-task, don't disrupt it.
- `git -C "$REPO" remote get-url upstream` → must succeed. If the upstream remote is missing, stop and explain which URL to add.
- Check there isn't an in-progress merge/rebase: if `.git/MERGE_HEAD` or `.git/rebase-merge` exists, stop and tell the user to resolve the existing operation first.

## Step 2 — Fetch and preview

- `git -C "$REPO" fetch upstream` — capture output, report any fetch warnings.
- Compute the gap: `git -C "$REPO" log --oneline HEAD..$UPSTREAM_REF`.
  - If zero commits: report "already up to date" and exit the whole workflow. Mark remaining tasks completed with a note.
  - Otherwise: count commits, count lines changed (`git diff HEAD..$UPSTREAM_REF --shortstat`), and list files most changed (`git diff HEAD..$UPSTREAM_REF --stat | tail -20`).
- Harness risk check — grep the diff for files that are load-bearing for the fork:
  - Agent: `run_agent.py`, `agent/claude_agent_adapter.py`, `cli-config.yaml.example`, `hermes_cli/web_server.py`.
  - Webui: `bootstrap.py`, `api/streaming.py`, `api/routes.py`, `static/index.html`.
  - For each risk file with changes, show line counts of the change and warn the user. Do NOT stop — just surface it.
- Report the preview to the user: commit count, lines changed, top files, harness-risk callouts. Do not ask for confirmation — just continue to Step 3. The user can Ctrl+C if they want to bail.

## Step 3 — Merge or rebase

Agent (`$STRATEGY=merge`):

```bash
git -C /Users/luca/dev/hermes-agent merge upstream/main --no-edit
```

Webui (`$STRATEGY=rebase`):

```bash
git -C /Users/luca/dev/hermes-webui rebase upstream/master
```

**Conflict handling:**
- If the command exits non-zero, check for unresolved conflicts: `git -C "$REPO" diff --name-only --diff-filter=U`.
- If conflicts exist:
  - List all conflicted files.
  - For lock files (`uv.lock`, `package-lock.json`, `web/package-lock.json`, etc.): take upstream's version with `git -C "$REPO" checkout --theirs <path>` and plan to regenerate (`uv lock` / `npm install --package-lock-only`). This is safe because lock files are derived.
  - For code files: do NOT auto-resolve. Stop, list the conflicts, and tell the user exactly which files need manual resolution. Leave the merge/rebase in progress so the user can finish it. Do not mark later tasks completed.
- After any auto-resolutions, re-run the conflict query. If clean, stage the resolved files and continue the merge/rebase (`git merge --continue` or `git rebase --continue`).

**Sanity post-merge:**
- `git -C "$REPO" diff --name-only --diff-filter=U` must be empty.
- `git -C "$REPO" log -1 --format="%H %s"` — report the new HEAD commit.

## Step 4 — Post-merge dependencies

Agent only:
- `cd /Users/luca/dev/hermes-agent && uv lock` — regenerates the lockfile against current `pyproject.toml`. Report any added/removed packages from the output.
- `cd /Users/luca/dev/hermes-agent && uv sync --extra all` — ensures the venv has all extras installed (gateway needs `python-telegram-bot` from the `all` extra).
- If `uv.lock` was staged as part of conflict resolution, the merge is not committed yet. After `uv lock`, run `git add uv.lock` and then `git commit --no-edit` to finalize the merge commit.

Webui: no dependency step. It reuses hermes-agent's venv.

## Step 5 — Harness integrity

Agent only. This is the load-bearing check — the whole point of the fork is that the Claude Code harness keeps working. Run from the repo root so `uv run` resolves the dev venv:

```bash
cd /Users/luca/dev/hermes-agent && uv run python -c "
from agent.claude_agent_adapter import ClaudeAgentSession
import run_agent, inspect
sig = inspect.signature(run_agent.AIAgent.__init__)
assert 'harness' in sig.parameters, 'AIAgent lost harness= kwarg'
assert hasattr(run_agent.AIAgent, '_get_harness_mode'), 'AIAgent lost _get_harness_mode'
assert hasattr(run_agent.AIAgent, '_run_claude_code_conversation'), 'AIAgent lost _run_claude_code_conversation'
print('harness OK:', ClaudeAgentSession.__module__)
"
```

If this fails, the merge broke the harness. Stop immediately, report the exact error, and advise the user either to (a) roll back the merge (`git reset --hard ORIG_HEAD`) or (b) inspect the diff in `run_agent.py` for where the harness wiring got cut.

Webui: skip this step (harness lives in the agent repo).

## Step 6 — Optional fast smoke tests

Ask the user whether to run the fast pytest subset. If yes:

```bash
cd /Users/luca/dev/hermes-agent && uv run pytest tests/run_agent/test_run_agent.py -x -q --timeout=30 2>&1 | tail -40
```

If the user declines or tests fail, don't block — include the outcome in the summary. These aren't harness-specific; they're just a safety net for upstream regressions.

## Step 7 — Summarize

Produce a compact report with these sections:

**Pulled:** `<N>` commits, `<insertions>+/<deletions>-` across `<files>` files. New HEAD: `<short-sha> <subject>`.

**Categorized commits:** group by conventional-commit prefix (security, feat, fix, refactor, perf, chore, test, docs). Show counts per category and 1-sentence highlights for security + feat + notable fixes. Don't list every commit — that's what `git log` is for.

**Harness:** OK / BROKEN. If broken, cite the failing check.

**Follow-ups you might want:**
- `git push origin <branch>` to publish — only mention this; do NOT run it.
- Restart services if gateway/webui are running:
  - `launchctl kickstart -k gui/$(id -u)/ai.hermes.gateway`
  - `pkill -f hermes-webui/server.py; nohup /Users/luca/dev/hermes-webui/start.sh ...`
- Any conflicts or skipped tests that need attention.

Keep the summary under ~20 lines. The user wants to skim, not read a novel.

## Notes on robustness

- Always quote paths and use `git -C` to avoid cwd drift between Bash tool calls.
- Never force-push, never `git reset --hard` without the user asking.
- If anything unexpected happens (a new kind of conflict, a harness check that doesn't exist yet, a pyproject change that breaks `uv lock`), stop and surface it. Do not improvise past a failure.
- Trust the pre-merge state. If a check fails, fix the check or the cause — don't skip the check.
