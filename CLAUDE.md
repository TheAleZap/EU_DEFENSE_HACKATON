# Instructions for Claude working in this repo

1. **Before doing anything, read `docs/TEAM_MEMORY.md`** (complete shared context of the team lead's session), then `docs/TEAM_BRIEF.md` (plan, roles, rules, checkpoints) and `docs/TEST_REPORT.md` (what is tested and what is still open).
2. Ask the teammate which role they are (P1 detection, P2 tracking, P3 Ch 9 + Ch 8, P4 Metis + pitch) and stay inside it.
3. Follow `docs/TEAM_BRIEF.md` Section 9 rules: never tune on test; only kit evaluators give reportable numbers; label every number with model, FP32/INT8 and hardware; run `check_submissions.py` before any upload; never commit data, weights or secrets.
4. Don't present synthetic or dummy-model results as real scores. Mark unknowns as unconfirmed.
5. When something new is decided or measured, add it to `docs/TEAM_MEMORY.md` (decision log or measurements) and a row to `EXPERIMENTS.md`.
6. Kits live outside the repo; pass their paths with `--kit` / `--c6` / `--c8` / `--c9`.
7. File names: in the claude.ai project these docs are `EDTH_TEAM_BRIEF.md` / `EDTH_ALL_CHALLENGES.md`; in this repo they are `docs/TEAM_BRIEF.md` / `docs/ALL_CHALLENGES.md`. Same content.
