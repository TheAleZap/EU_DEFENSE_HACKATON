# START HERE: Siesta (EDTH Paris 2026, Challenges 6 + 8 + 9)

This folder is **everything** produced for the team, in one place. It's also a ready-to-push Git repo (first commit included).
**This folder is the single source of truth.** Older copies in the claude.ai project use older file names; ignore those.

## 1. Read in this order
| # | File | What it is |
|---|---|---|
| 1 | `docs/1_TEAM_MEMORY_full_shared_context.md` | The **complete memory** of the team lead's chat: sources, decisions and why, measurements, plan, rules. **Give this to your Claude first.** |
| 2 | `docs/2_TEAM_BRIEF_plan_roles_checkpoints.md` | The **operational plan**: open questions, Gate 0 checklist, kit facts and pitfalls, hardware, roles P1–P4, checkpoints, pitch, rules. |
| 3 | `docs/3_TEST_REPORT_results_and_coverage.md` | **15/15 tests passed**, plus a table of every Ch 6/8/9 requirement and what still needs doing. (`3_TEST_REPORT_raw_results.json` is the raw output.) |
| 4 | `docs/4_ALL_CHALLENGES_reference.md` | Reference: all 9 hackathon challenges and why we chose 6 + 8 + 9. |
| - | `docs/web_pages/*.html` | The same four documents as offline web pages (open in any browser). |
| - | `README.md` | **Exact commands** to run everything + team GitHub workflow. |
| - | `CLAUDE.md` | Read automatically by Claude Code: tells Claude to load the team memory and follow the rules. |

## 2. Folders and code
| Folder / file | Owner | Purpose |
|---|---|---|
| `ch6_detection_tracking/build_tiled_dataset.py` | P1 | Makes native-resolution crops so tiny drones stay visible during training |
| `ch6_detection_tracking/tiled_detect.py` | P1 | Runs the detector over tiles → **COCO detection file** (valid / test) |
| `ch6_detection_tracking/tiling.py` | P1/P2 | Shared helper: tiles, detector wrapper, NMS (reuses the kit's own decoding) |
| `ch6_detection_tracking/detect_video.py` | P2 | Detections for every video frame; `--mode full` for scoring, `--mode motion` for the efficiency story |
| `ch6_detection_tracking/track.py` | P2 | Tracker → **MOT files**; outputs only confirmed drones, fills short gaps, camera-motion compensation |
| `ch6_detection_tracking/synth_test.py` | P2 | Synthetic test of the tracker mechanics (not a real score) |
| `ch9_acoustic_wake/train_cv.py` | P3 | Trains Ch 9 with augmentation; picks the wake threshold by cross-validation on 56 clips |
| `ch9_acoustic_wake/predict_consistent.py` | P3 | Writes the **Ch 9 JSONL**; wake and sound class never contradict |
| `ch8_edge_triage/train_rank.py` | P3 | Trains Ch 8 to rank frames in the right order (ranking loss + augmentation) |
| `ch8_edge_triage/pool_clips.py` | P3 | Ranks whole clips instead of single frames → **Ch 8 JSONL** |
| `metis_axelera_compile/preprocess_cnn.py` | P4 | Input-preparation file Axelera's compiler needs for the Ch 8/9 models |
| `metis_axelera_compile/preprocess_rfdetr.py` | P4 | Same for the Ch 6 RF-DETR model |
| `check_submissions_before_upload.py` | All | **Must print PASS before any upload** |
| `tests/run_all_tests.py` | All | The 15-test suite; rerun with real models once you have them |
| `early_probes/` | Reference | This morning's first experiments (motion and sound probes) |
| `submissions/` | P4 | Put each checked submission set in `submissions/vN/` |
| `EXPERIMENTS_LOG.md` | All | One row per training or eval run |
| `requirements.txt` | All | Extra packages beyond the kits (mainly `opencv-python`) |

**Not in this folder (by design):** the three challenge kits and datasets (unpack them outside, e.g. `~/edth/kits/`), trained models, and API keys.

## 3. Start now
1. One person: create a **private GitHub repo**, then inside this folder run `git remote add origin <url> && git push -u origin main`, and invite the team.
2. Everyone: clone it, create your branch (`p1-detect`, `p2-track`, `p3-audio-triage`, `p4-metis-pitch`), unpack the kits outside the repo, then `pip install opencv-python` and `uv venv && uv pip install -e .` in each kit's `starter-kit/`.
3. Open Claude (Claude Code in the repo, or the app with docs 1 + 2 uploaded) and say **"I am P1/P2/P3/P4"**.
4. **Next checkpoint: 13:00.** One valid submission per challenge that passes `check_submissions_before_upload.py`.
