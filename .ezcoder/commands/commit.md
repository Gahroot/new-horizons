---
name: commit
description: Run checks, agent code review, commit with AI message, and push
---

1. Run quality checks (same as `.github/workflows/ci.yml`; no linter/type checker is configured):
   python3 -m compileall -q horizons tests
   python3 -m unittest discover -s tests
   Fix ALL errors before continuing (no auto-fix tool is configured). Never skip or weaken tests.
2. Review changes: run git status and git diff --staged and git diff

3. Fast review gate: spawn ONE subagent with the full diff. Instructions: review ONLY
   the diff for real bugs, regressions, leftover debug code, and unintended changes.
   Score each issue 0-100 confidence (pre-existing issues and stylistic nitpicks = false
   positives, score low). Report ONLY issues with confidence >= 80, with file:line and a
   one-line fix. If none, reply "CLEAR". This is a last check, not a deep audit - be fast.
4. If CLEAR: go straight to step 5 and push WITHOUT asking. If issues >= 80: STOP, show them, then ask
   with `ask_user` - one `choice` question (`id: "land"`, "Want me to fix this first, or commit and push anyway?"):
   "Fix it first, then commit & push" (recommended, hint: keeps the branch green) and
   "Commit & push anyway" (hint: issue stays open in the log). The card is the ONLY ask - never restate
   the options as text. Only if `ask_user` is unavailable, ask the same two options in prose.
   On fix-first: fix, re-run step 1, then continue (no re-review). Otherwise continue as-is.
5. Stage relevant files with git add (specific files, not -A). Never stage .venv, .horizons, or secrets.
6. Generate a commit message: start with a verb (Add/Update/Fix/Remove/Refactor); specific, one line preferred.

7. Commit AND push in one go - never pause for confirmation here (never use --no-verify):
   git commit -m "your generated message"
   git push
