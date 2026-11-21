# Instructions for Claude working in this repo

1. **Before doing anything, read `docs/1_TEAM_MEMORY_full_shared_context.md`** (complete shared context of the team lead's session), then `docs/2_TEAM_BRIEF_plan_roles_checkpoints.md` (plan, roles, rules, checkpoints) and `docs/3_TEST_REPORT_results_and_coverage.md` (what is tested and what is still open).
2. Ask the teammate which role they are (P1 Alejandro: Ch 6 detection, P2 Salo: Ch 6 tracking, P3 Viktoriia: Ch 9 only (Ch 8 dropped), P4 Dana: Metis + pitch) and stay inside it.
3. Follow `docs/2_TEAM_BRIEF_plan_roles_checkpoints.md` Section 9 rules: never tune on test; only kit evaluators give reportable numbers; label every number with model, FP32/INT8 and hardware; run `check_submissions_before_upload.py` before any upload; never commit data, weights or secrets.
4. Don't present synthetic or dummy-model results as real scores. Mark unknowns as unconfirmed.
5. When something new is decided or measured, add it to `docs/1_TEAM_MEMORY_full_shared_context.md` (decision log or measurements) and a row to `EXPERIMENTS_LOG.md`.
6. Kits live outside the repo; pass their paths with `--kit` / `--c6` / `--c8` / `--c9`.
7. File names: in the claude.ai project these docs are `docs/2_TEAM_BRIEF_plan_roles_checkpoints.md` / `docs/4_ALL_CHALLENGES_reference.md`; in this repo they are `docs/2_TEAM_BRIEF_plan_roles_checkpoints.md` / `docs/4_ALL_CHALLENGES_reference.md`. Same content.
