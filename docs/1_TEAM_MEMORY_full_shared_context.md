# SIESTA: Team Memory (complete shared context)

> **For Claude reading this file:** this is the complete memory of the team lead's working session with Claude (Fri 20 Nov 21:50 → Sat 21 Nov 12:05, Paris time). Treat it as what you already know. It records what the team was given, what was researched and verified, every decision and why, what was built and tested, and what's still open. Labels: **[verified]** = checked against official files, code or docs; **[test]** = measured by our own tests; **[unconfirmed]** = not yet known; **[our view]** = a judgement. Never present [unconfirmed] or synthetic numbers as facts. If something isn't here, in `docs/2_TEAM_BRIEF_plan_roles_checkpoints.md`, or in the official kits, say you don't know.
>
> **Companion files (same repo / project):**
> - `docs/2_TEAM_BRIEF_plan_roles_checkpoints.md`: the operational plan (roles, checkpoints, rules, pitch).
> - `docs/4_ALL_CHALLENGES_reference.md`: all 9 challenges in detail.
> - `docs/3_TEST_REPORT_results_and_coverage.md`: tests and the requirements coverage matrix.
> - `README.md`: how to run the code.
>
> **Teammate quick start:** say "I am P1/P2/P3/P4" and read your role in `docs/2_TEAM_BRIEF_plan_roles_checkpoints.md` Section 6.

---

## 1. The situation in one paragraph

We're a team of **4** at the **European Defence Tech Hackathon (EDTH) Paris, 20–22 Nov 2026**. Pitches are Sunday evening for preselected teams. We're building **"Siesta"**, an edge sensor node that **listens** for drones (Challenge 9), **looks** with a camera detector and tracker only when needed (Challenge 6), and **sends** only the most valuable frames over a weak link (Challenge 8). All three run on the **Axelera Embedded 110m (Metis AIPU, 1 GB)**, which is available on site. Tagline: *"Eyes closed, ears open. We nap so you don't have to."* As of Sat 12:05 the code kit is written and **passes 15/15 tests** on the real kit data, and real model training is starting on the team's M4 Max MacBook. **No real accuracy numbers exist yet.**

---

## 2. Team, hardware, roles

| Role | Person | Person's machine | Owns |
|---|---|---|---|
| **P1** | **Alejandro** (team lead, GitHub owner) | **MacBook Pro 16" M4 Max, 48 GB** (the only fast machine; Apple GPU = MPS, not CUDA) | Ch 6 detection: RF-DETR Nano training on tiles, tiled inference, COCO output |
| **P2** | **Salo** | laptop | Ch 6 tracking: `detect_video.py` + `track.py`, MOT output |
| **P3** | **Viktoriia** | laptop | Ch 9 only: `train_cv.py`, `predict_consistent.py` (Ch 8 dropped Sat 13:55) |
| **P4** | **Dana** | **the team's one Windows laptop** (WSL2 + Ubuntu 22.04) | Axelera/Metis compile and benchmarks, integration, submission checks, demo video, pitch |

**Hardware facts [verified]:**
- The **Voyager SDK installs on Linux (Ubuntu 22.04+); Windows only via WSL2; macOS isn't listed.** No Mac can compile for Metis. Compile on the Axelera host in the Spectrum Room, or on P4's WSL2 if the compiler-only install works without a card [unconfirmed].
- **RF-DETR 1.12.1 auto-detects Apple MPS** and supports MPS training (gather-based grid_sample, because MPS lacks its backward). The kit accepts rfdetr ≥ 1.4.0.
- The **Ch 8/9 kit trainers default to CUDA-else-CPU**, so pass `--device mps` on the Mac.
- Check Python is native: `python -c "import platform; print(platform.machine())"` must print `arm64`.
- **Fallback for detector training:** free Colab or Kaggle NVIDIA GPU, if the 1-epoch timing on the Mac says a full tiled run can't finish by about 18:00.
- **Ch 6 edge score = end-to-end throughput:** the host-side decoder and tracker must be timed **on the Axelera host**, not the Mac.

---

## 3. What the team was given (sources)

### Event resources (organisers' list) [verified, from the team lead]
1. **3D printers** from Druk Army (3D Print Army). Not used by us.
2. **Fire Point equipment for Challenge 2** (hall near entrance); the listed "equipment" only repeats the Ch 2 specs. Not used.
3. **Equipment for the Axelera challenges (06–09) in the Spectrum Room, upstairs.** This is what we use.
4. **Featherless AI**: $25 LLM API credits, promo code `EDTH26`, OpenAI-compatible base URL `https://api.featherless.ai/v1`, one use per account. It's an LLM API, **not** training GPUs. Optional coding help only, never inside the pipeline.
5. **Alta Ares** sits in the ground-floor restaurant, has RC drones (for their Search & Destroy challenge), and published a GitHub repo `AA-OpenProjects/AA-Hackathon-EDTH-202611`.

### The 9 challenges (full detail in `docs/4_ALL_CHALLENGES_reference.md`)
| # | Name (sponsor) | Teams at registration | Our verdict |
|---|---|---|---|
| 1 | Voice-to-Map (Lysk) | 9 | Not chosen: crowded, no scorer |
| 2 | "Keep It Level" stabilised antenna platform (Fire Point) | 2 | Not chosen: needs controls and CAD skills |
| 3 | Search & Destroy: target re-acquisition + autonomous strike guidance (Alta Ares) | 5 | Not chosen. **Claude won't help develop autonomous strike guidance.** |
| 4 | Fleet Orchestration (Alta Ares) | 7 | Not entered; appears only as a "next step" in the pitch |
| 5 | Jet-drone interceptor (Frontier Defence Labs) | 0 | Infeasible (needs jet-engine facilities) |
| **6** | **Small Aircraft Detection & Tracking (Piotr Gryko / Axelera)** | 4 | **CHOSEN** |
| 7 | RGB + Thermal detection & tracking (Piotr Gryko) | 1 | Not chosen (outside our chain) |
| **8** | **Edge Video Triage (Piotr Gryko)** | 1 | **CHOSEN** |
| **9** | **Acoustic-to-Visual Drone Detection (Piotr Gryko)** | 3 | **CHOSEN** |

**Data:** one pCloud link for Ch 6–9 (from the challenge pages). The team lead uploaded the three kits: `challenge-6-cuas-tracking_tar.gz`, `challenge-8-edge-triage_tar.gz`, `challenge-9-acoustic-visual-cuas_tar.gz`.

**Kit numbering trap [verified]:** inside the kits, **Ch 6 = "Challenge 1"**, **Ch 8 = "Challenge 4"**, **Ch 9 = "Challenge 5"**.

### Facts about the challenges we didn't choose (in case it comes up)
- **Ch 4 [verified from the repo]:**
  - Hand-in is a `planner.py`, **Python standard library only**, 60 s per scenario, invalid plan = 0.
  - Live board on 8 public scenarios; deadline **Sun 22 Nov 12:00**.
  - The repo has **no link-loss part**, although the PDF mentioned one.
  - Score = points − λ·energy − μ·effectors.
- **Ch 3 [verified from the repo]:** a PyBullet sim, 30 Hz stick commands, hidden terrains, jury-judged.
- **Madrid hook [verified via news reports]:** **3 Feb 2020, Madrid-Barajas** closed around 12:30–14:30 after pilots reported a drone. 26 flights were rerouted, and the Civil Guard later found no proof the drone existed.

---

## 4. Decision log (chronological, with reasons)

1. **Fri 21:50, first look at Ch 6.** Explained the metrics (mAP, small-object AP, MOTA, IDF1, continuity) and the edge constraint.
2. **Idea: combine Ch 4 + Ch 6.** Initially suggested a story link. **Corrected after reading the Ch 4 brief:** the planner's only input is a scenario JSON, so there's no technical link; the challenges are scored separately.
3. **Reviewed all 9 challenges** with resources. Ranked them; teammates liked 4, 6 and 9. Advised against three separate submissions. Proposed 6 + 9, then **6 + 8 + 9** ("hear → look → send"), all same sponsor, low competition.
4. **Name:** the team lead proposed "Siesta Defense System" with "We nap so you don't have to". Claude suggested dropping "Defense System" (it detects and confirms; it doesn't defend), adding "Eyes closed, ears open" (answers "what if it sleeps through an attack?": only the camera naps), and using the **Barajas 2020** story as the hook and close ("drone confirmed" or "reopen the runway").
5. **Kits analysed (Sat 10:40).** Key discoveries:
   - The Ch 9 kit itself recommends the acoustic → RF-DETR cascade.
   - Tiny drones (7–10 px in full-HD video).
   - Weak baselines (Ch 6 MOTA −0.013; Ch 8 Spearman −0.157).
   - Ch 8 labels are clip-constant.
   - Ch 9 wake == drone_buzz.
   - The MOT scorer penalises a misplaced box twice.
6. **Probes run:** motion finds tiny drones but makes clutter (motion-only MOTA −0.56). The physics sound features are weak (LOO F1 0.31), so they were **cut**.
7. **Scope review (Sat 11:00):** a teammate's Claude said the plan was too broad. Resolved by priority ranking: **Ch 6 (2 people) > Ch 9 > Ch 8**, Ch 8 limited to clip pooling + ranking loss + Metis benchmark.
8. **Hardware decided (11:03–11:12):** M4 Max = P1; Windows = P4 (WSL2); questions to ask the Axelera team listed.
9. **Code kit built (11:38–11:50)** and **full test suite run (12:05): 15/15 PASS.** The suite found that motion windows cover the 7 px drone in seq 4 only 19% of the time. **Decision:** default raised from 6 to 12 windows; **use FULL mode for scored submissions**, motion mode for the efficiency story.
10. **Working method:** GitHub repo `TheAleZap/EU_DEFENSE_HACKATON` (see `README.md` "Team workflow"). Branches `p1-detect`, `p2-track`, `p3-audio-triage`, `p4-metis-pitch`.
11. **Names and scope (Sat 13:55, team lead):** P1 Alejandro, P2 Salo, P3 Viktoriia, P4 Dana. **Ch 8 dropped as a submission**: P3 does Ch 9 only; Ch 8 code stays in the repo as pitch material ("send" step of the story). Axelera hardware not yet available: nobody waits for it; only Metis compile/benchmarks are blocked.

---

## 5. The Siesta approach (ideas A–I) [our view, partly verified]

| Idea | Where | What | Evidence |
|---|---|---|---|
| A. Zoom where it matters | Ch 6 detection | Train and infer on **native-resolution 384 tiles** | 7 px drones vanish when 1920 px frames are resized to 384 [verified]. Tiled set built and loader-checked [test] |
| B. Peek before you stare | Ch 6 video | Motion proposals → detector only on windows + periodic full sweep | About 48% fewer detector calls [test]; coverage caveat for seq 4 [test] |
| C. Scorer-aware output | Ch 6 tracking | Emit only confirmed tracks; fill short gaps; silence beats a wrong box | Synthetic: MOTA −0.86 → +0.56 avg [test, synthetic] |
| D. Two thresholds | Ch 6 | Low threshold for COCO mAP, strict for MOT | From scorer code [verified] |
| E. Rank clips, not frames | Ch 8 | Pool priority per `video_id` | Labels clip-constant [verified]; synthetic capture@1% 0.40 → 0.90 [test, synthetic] |
| F. Train for order | Ch 8 | RankNet pairwise loss + augmentation | Baseline used MSE, no augmentation [verified] |
| G. Consistent outputs | Ch 9 | wake == (class == drone_buzz) via fused probability | Identity holds 56/56 [verified] |
| H. Augment + CV threshold | Ch 9 | Time shift, gain, masks, background mixing; 7-fold CV on 56 clips | Kit picks the threshold on 8 clips [verified] |
| I. One chip, three models | All | Compile all three on Metis; cost ledger (latency, memory, INT8 drop) | 15–20% of each proposed score is Metis efficiency [verified]. FP32 sizes 103 / 89 / 3.6 / 3.6 MB; should fit 1 GB [estimate] |

**Deliberately not doing:**
- Hard-coding "one object per sequence" (valid sequences have exactly 1 object; test may differ).
- Tuning on test.
- Calling camera-off time "energy" (the kit's `energy_savings_pct` key isn't measured energy).
- Claiming Ch 6 detections feed Ch 8 (Ch 8 is mostly ground-level battlefield footage).
- Physics sound features.
- Appearance re-ID for tiny targets.

---

## 6. Key kit facts (full list in `docs/2_TEAM_BRIEF_plan_roles_checkpoints.md` Section 2) [verified]

**Ch 6**
- Classes `0 drone_air, 1 airplane, 2 helicopter`.
- Train 859 (14 clips) / valid 422 / test 328; 200 calibration images.
- 8 tracking sequences: valid seq 1–4 with GT, test seq 5–8.
- COCO `category_id` is 0-based; MOT `class_id` and `frame` are 1-based; top 100 per image.
- MOT matching is IoU ≥ 0.5, class-agnostic.
- ONNX outputs must be read by name (`dets`, `labels`).
- The baseline runs CPU ONNX Runtime at 40 ms/frame; no trained weights are shipped.
- The backbone-only export (`features [1,256,24,24]`) is the Metis starting point.
- Proposed judging: Detection 40 / Tracking 40 / Edge 20.

**Ch 8**
- 947 frames / 282 clips; train 698 / valid 138 / test 111.
- Test budgets k = 1 (1%) and 6 (5%); ties broken pessimistically.
- JSONL `{"id", "triage_keep", "priority_score"}`.
- Proposed judging: capture 55 / Spearman 25 / Metis 20 (bonus for clip dedup and 100% on-chip).

**Ch 9**
- 62 clips; train 48 / valid 8 / test 6.
- JSONL `{"id", "wake_triggered", "sound_class"}`.
- The trainer stores the wake threshold inside the ONNX.
- Proposed judging: wake F1 40 / accuracy 25 / camera-off 20 / Metis 15.

**Metis compile [verified, Axelera docs]**
- `axcompile -i model.onnx --imageset <calib> --transform preprocess.py --input-shape 1,3,H,W -o out/`.
- `--transform` loads `get_preprocess_transform(image)`, which takes a PIL image or array and returns a `torch.Tensor`.
- Output: `compiled_model/` + `compilation_report.json` (with a `status` field).
- The CNNs have normalisation inside the graph; RF-DETR needs ImageNet normalisation.

---

## 7. Code kit map (repo layout)

```
ch6_detection_tracking/build_tiled_dataset.py   P1  native 384 crops + originals + negatives, RF-DETR folder layout
ch6_detection_tracking/tiling.py                     shared: kit decode import, tile grid, Detector, class-wise NMS
ch6_detection_tracking/tiled_detect.py          P1  COCO preds for valid/test (tiles + whole-frame pass)
ch6_detection_tracking/detect_video.py          P2  per-frame detections, modes full | motion (default 12 windows)
ch6_detection_tracking/track.py                 P2  tracker + output policy -> MOT files (offline / --online / --kit for camera comp)
ch6_detection_tracking/synth_test.py            P2  synthetic tracker mechanics test
ch8_edge_triage/train_rank.py            P3  kit model + RankNet + augmentation; selection on clip-pooled valid
ch8_edge_triage/pool_clips.py            P3  clip pooling for valid/test JSONL
ch9_acoustic_wake/train_cv.py              P3  augmentation + 7-fold CV threshold + fused rule; exports ONNX with metadata
ch9_acoustic_wake/predict_consistent.py    P3  consistent JSONL predictions
metis_axelera_compile/preprocess_cnn.py     P4  axcompile transform for Ch 8/9 (224, no normalisation)
metis_axelera_compile/preprocess_rfdetr.py  P4  axcompile transform for Ch 6 (384, ImageNet normalisation)
check_submissions_before_upload.py        ALL pre-upload checker (must print PASS)
tests/run_all_tests.py      ALL 15 tests; rerun with real models via --c6-onnx/--c8-onnx/--c9-onnx
early_probes/motion_probe.py, early_probes/harmonic_probe.py   early probes (Section 8)
```
Exact commands are in `README.md`. Extra dependency: `opencv-python`.

---

## 8. Every measurement so far (none are real model scores)

| Measurement | Result | Type |
|---|---|---|
| Kit baselines (from kits, 1–2 epoch smoke) | Ch 6 mAP@50 0.447, small_AP50 0.506, MOTA −0.013, IDF1 0.248; Ch 8 capture@1% 0, @5% 0.286, Spearman −0.157; Ch 9 wake F1 0.80, accuracy 0.50 | [verified, kit] |
| Motion probe, seq 1–4 | target found in 99/150, 68/72, 145/150, 124/150 frames; 34/14/1151/197 candidates per frame; motion-only MOTA −0.56 | [test] |
| Physics sound features | LOO wake F1 0.31, so cut | [test] |
| Full test suite | 15/15 PASS (see `docs/3_TEST_REPORT_results_and_coverage.md`) | [test] |
| Motion-window coverage (12 windows) | seq 1: 65%, seq 2: 97%, seq 3: 95%, **seq 4: 19%** | [test] |
| Detector calls per frame | full 13.3, motion 6.9 (about 48% fewer) | [test, dummy model] |
| Tracker policy | MOTA −0.86 → +0.56, IDF1 0.37 → 0.76 (avg of 3) | [test, **synthetic**] |
| Ch 8 clip pooling | capture@1% 0.40 → 0.90, @5% 0.80 → 0.99, Spearman 0.55 → 0.69 | [test, **synthetic**] |
| Metis preprocessing vs kit | max diff 0.0 on 448 images | [test] |
| Non-pretrained Ch 9 model | collapses to "always wake" | [test]: **confirm the pretrained download** |
| Ch 6 RF-DETR Nano, 1 epoch, untiled, M4 Max MPS (Sat 12:34) | 114 s/epoch wall incl. val; valid mAP50 0.455, mAP50_95 0.280, small_AP50 0.576; FP32 CPU ORT 26 ms/img | [verified, kit scorer] |
| Ch 6 RF-DETR Nano, kit baseline 20 ep (early stop at 16), untiled, M4 Max MPS (Sat 12:52) | 16 min; valid mAP50 0.595, mAP50_95 0.392, small_AP50 0.734 | [verified, kit scorer] |
| Same model, tiled inference (tiled_detect.py) | valid mAP50 0.464, small_AP50 0.691, 13.9 calls/img: worse than untiled | [verified, kit scorer] |
| Same model, kit MOT predictor | valid MOTA −0.215, IDF1 0.157 | [verified, kit scorer] |
| Ch 6 v0 submission (Sat 13:20) | `submissions/v0/` COCO test + MOT seq 5–8 from the 20-epoch model; check_submissions PASS | [test] |
| Ch 6 tiled dataset build (Sat 12:35) | train 3266 imgs / 2544 boxes, valid 1590 / 1182, 5 s; matches test counts | [test] |

---

## 9. Plan from here (Sat 12:05 onwards)

**Gate 0 (target 12:00, finish now):** organiser answers, installs, data checks, `v0` baselines. See `docs/2_TEAM_BRIEF_plan_roles_checkpoints.md` 0a.

| When | Must be true |
|---|---|
| Sat 13:00 | Valid `v0` submissions for Ch 6 (COCO + MOT), Ch 8, Ch 9; Ch 9 CNN compile attempted on Metis |
| Sat 14:00, cut line | RF-DETR backbone compiles, or stop and report CPU INT8 |
| Sat ~14:00 | Tiled RF-DETR training launched on the M4 Max |
| Sat 19:00 | Tracker beats baseline on valid (leave-one-sequence-out); Ch 8/9 improved; cascade demo on ≥ 1 clip; `v1` |
| Sun 11:00 | Final `v2` test files, checker PASS, submitted early |
| Sun 12:00 | Feature freeze; demo video recorded; afternoon pitch rehearsals |

**Next actions per person:**
- **P1:** 1-epoch timing on MPS → build tiled set → train → compare tiled vs untiled on valid.
- **P2:** run `detect_video.py --mode full` with P1's model → `track.py` → tune with leave-one-sequence-out.
- **P3:** `train_cv.py` (check the pretrained weights load) → `train_rank.py` → pooling.
- **P4:** Axelera room answers → compile the Ch 9 model first.

---

## 10. Open questions [unconfirmed] (owner P4 unless stated)
1. Can one team submit to Ch 6, 8 and 9? How is a combined project judged and pitched?
2. Deadlines and upload method for test predictions (Ch 6/8/9).
3. Are the kits' proposed judging weights final? What's the hardware measurement protocol?
4. Axelera room: device count, host OS/CPU, SDK version, booking, SSH, running our own Python, power measurement, compiler-only install on WSL2.
5. Pitch length and preselection time.
6. Optional: permission to film/record an Alta Ares RC drone for a real-drone demo (only under organiser safety rules).

---

## 11. Rules every teammate and every Claude follows
1. Never tune on test. Calibration folders are for INT8 only.
2. Only kit evaluators produce reportable numbers; label each with model, FP32/INT8 and hardware.
3. Keep a valid tagged submission at all times; run `check_submissions_before_upload.py` before every upload; submit early.
4. Don't overclaim: say "camera-off time", not energy; small test sets are noisy; synthetic results are not scores.
5. No secrets in the repo; no datasets or model weights in git.
6. **Claude won't assist with Challenge 3's autonomous strike guidance.** Siesta is detection, confirmation and data triage.

---

## 12. Glossary
- **MOTA**: 1 − (misses + false positives + ID switches) / ground-truth boxes.
- **IDF1**: how consistently each object keeps one ID.
- **Continuity**: share of ground-truth frames covered.
- **small_AP**: AP on boxes < 32×32 px.
- **Capture@k%**: high-priority frames among the top k% (budget-normalised).
- **MPS**: Apple GPU backend.
- **Metis / Embedded 110m**: Axelera's AI chip on an M.2 card, 1 GB.
- **Voyager SDK / axcompile**: Axelera's toolchain.
- **INT8 calibration**: setting quantisation ranges using sample images.
- **Host tail**: the code that runs on the host CPU after the chip (decoder, tracker).
- **LOO / leave-one-sequence-out**: tune on 3 sequences, test on the 4th, rotate.
