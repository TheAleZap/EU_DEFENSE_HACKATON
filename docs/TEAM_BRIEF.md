# SIESTA: "Eyes closed, ears open." Team Master Plan (EDTH Paris 2026)

> **Full shared context of the team lead's chat: `TEAM_MEMORY.md`. Read it first.**
>
> **How to use this file (each teammate):** upload this file to Claude (or save it as `CLAUDE.md` in the repo root), then say **"I am P_ — help me with my part."** Read **Section 0a (Gate 0 checklist)**, **Section 10 (decision history)** and **your role in Section 6** first. Companion reference: `EDTH_ALL_CHALLENGES.md`. Roles: **P1 = Ch 6 detection (M4 Max Mac)**, **P2 = Ch 6 tracking**, **P3 = Ch 9 then Ch 8**, **P4 = Metis + integration + pitch (Windows laptop)**.
>
> **Instructions for Claude reading this:** this is the team's agreed plan. Stay inside the assigned role and the rules in Section 9. Facts marked **[verified]** were checked directly against the official starter kits and data. Facts marked **[unconfirmed]** were not. Never invent metrics, formats or kit behaviour. If something isn't here or in the kit, say so. Prefer simple, validated, working solutions. Keep a valid submission at all times.

---

## 0. Open questions (fill in as answers arrive)

| Question | Answer |
|---|---|
| Do we have a GPU (which one)? | **MacBook Pro 16" (Nov 2024), Apple M4 Max, 48 GB unified memory.** It's Apple's GPU (MPS), not NVIDIA/CUDA. **[verified in code]** RF-DETR 1.12.1 auto-detects MPS and supports MPS training (gather-based grid_sample, since MPS lacks its backward); the kit accepts rfdetr ≥ 1.4.0. The Ch 8/9 trainers default to CUDA-else-CPU, so **pass `--device mps`**. **Check first:** `python -c "import platform; print(platform.machine())"` must print `arm64` (not `x86_64`/Rosetta). **Speed unknown until measured:** time 1 epoch; if the full tiled run can't finish by about 18:00, train the detector on a free cloud NVIDIA GPU (Google Colab / Kaggle free tiers) and keep the Mac for the tracker and Ch 8/9. GPU priority rules are in Section 6. |
| Can one team submit to Ch 6, Ch 8 and Ch 9? How are combined projects judged and pitched? | _TBD (ask Piotr Gryko)_ |
| Deadlines and submission method for Ch 6/8/9 test predictions | _TBD_ |
| Are the "proposed judging weights" in the kits final? | _TBD_ |
| What Axelera hardware and SDK version are in the Spectrum Room? How is device time booked? | _TBD (P4)_ |
| Pitch length; preselection time | _TBD_ |
| **Axelera room:** number of Embedded 110m devices; host CPU/OS; access method (booking, SSH); time limits | _TBD (P4)_ |
| **Axelera room:** Voyager SDK version (kit docs reference v1.8); can `axcompile` run on the host or only on a separate machine? | _TBD (P4)_ |
| **Axelera room:** host has Python ≥ 3.11, `uv`, `ffmpeg`, OpenCV, internet for pip? Can we run our own Python (decoder + tracker) on that host for end-to-end timing? Can we SSH in from our laptops? | _TBD (P4)_ |
| **Axelera room:** does the compiler-only install (`axelera-devkit`) work on a laptop without a Metis card, so we can compile before our device slot? | _TBD (P4)_ |
| **Axelera room:** known-working RF-DETR / backbone compile on Metis? Tips for the GridSample deformable-attention tail? | _TBD (P4)_ |
| **Axelera room:** any way to measure device power? (Ch 8 scores power; otherwise report latency and memory only) | _TBD (P4)_ |
| **Alta Ares (ground-floor restaurant):** may we film and record one of their RC drones for a sensing demo, and under what safety conditions? | _TBD (only with organiser permission)_ |

---

## 0a. Gate 0: readiness checklist (target Sat 12:00, before real work)

Tick every box. A box that can't be ticked gets an owner and a fallback; it never gets silently skipped.

**A. Confirm with organisers (P4 asks; answers go in Section 0)**
- [ ] One team may submit to **Ch 6, Ch 8 and Ch 9**, and we know how a combined project is judged and pitched. *If not allowed:* Ch 6 + Ch 9 are the core; Ch 8 becomes pitch-only.
- [ ] **Deadline and upload method** for test predictions of each challenge.
- [ ] Whether the kits' **"proposed" judging weights** are final, plus the **hardware measurement protocol** (the kits say organisers must publish these).
- [ ] **Pitch length** and **preselection time**.
- [ ] **Axelera access:** device count, host OS, SDK version, booking, SSH, whether we can run our own Python on the host, power measurement.
- [ ] *(Optional)* Alta Ares: permission and safety conditions to film or record an RC drone.

**B. Team and machines**
- [ ] Roles fixed: **P1 = M4 Max MacBook (Ch 6 detector)**, **P4 = the Windows laptop (Metis, integration, pitch)**, P2 = tracker, P3 = Ch 9 + Ch 8.
- [ ] Every teammate posts chip / RAM / OS in the team chat (Section 3b).
- [ ] Private Git repo created; data, model weights and `.env` are **git-ignored**; folder per challenge.
- [ ] Shared experiment log created (row format in Section 9).

**C. Software on every machine**
- [ ] Python ≥ 3.11, `uv`, `git`, `ffmpeg` installed.
- [ ] Each needed kit installed: `cd starter-kit && uv venv && uv pip install -e .`
- [ ] **Mac (P1):** `python -c "import platform; print(platform.machine())"` prints `arm64`; RF-DETR pretrained weights downloaded (first training run does this, so do it on good Wi-Fi).
- [ ] **Windows (P4):** WSL2 with **Ubuntu 22.04** installed (`wsl --install -d Ubuntu-22.04` in an admin PowerShell, then reboot). This needs admin rights and virtualisation enabled; **start it now**, it can take a while. *If blocked* (managed laptop): P4 works directly on the Axelera host over SSH or at the desk.

**D. Data integrity (each owner checks their kit)**
- [ ] Ch 6: train 859 / valid 422 / test 328 images; 200 calibration; `tracking_eval/seq_0001–0008` with GT only in 0001–0004.
- [ ] Ch 8: train 698 / valid 138 / test 111 frames; 200 calibration; test rows have `video_id`.
- [ ] Ch 9: train 48 / valid 8 / test 6 clips (wav + mp4 + png); 48 calibration spectrograms.
- [ ] Each kit's `viewer.py` opens.

**E. Baselines run end to end (the `v0` safety net)**
- [ ] Ch 6: train → export → `predict detect` → `evaluate detect`, and `predict mot` → `evaluate mot --split valid`. Also generate **test** files (`preds.json` on test images; `pred_mot/seq_0005–0008.txt`).
- [ ] Ch 8: train → predict → `evaluate triage`; test JSONL generated.
- [ ] Ch 9: train → predict → `evaluate acoustic`; test JSONL generated.
- [ ] All `v0` files saved and tagged.

**F. Metis**
- [ ] Device slot booked; `axcompile --help` works on the device host (or WSL2).
- [ ] `preprocess.py` written for the Ch 9 CNN (raw [0,1] RGB CHW float32; **no** ImageNet normalisation, since it's in the graph).
- [ ] First compile attempt of the Ch 9 model done (success or written error).

**G. Everyone has read Section 9 (rules) and Section 2 "pitfalls" below.**

### Known pitfalls (every Claude must respect these) [verified from kit code/docs]
- Kit numbering: Ch 6 = kit "Challenge 1", Ch 8 = "Challenge 4", Ch 9 = "Challenge 5".
- Ch 6 COCO `category_id` is **0-based**; MOT `class_id` is **1-based**; MOT `frame` is **1-based**; no duplicate (frame, track_id) rows; at most top 100 detections per image.
- Ch 6 ONNX outputs must be read **by name** (`dets`, `labels`), not position (same shape).
- Ch 6 MOT scorer: **IoU ≥ 0.5**, class-agnostic; a misplaced box costs FP + FN.
- RF-DETR expects **ImageNet-normalised** input; the Ch 8/9 CNNs have normalisation **inside** the graph (feed raw [0,1]).
- Ch 8: ranking uses `priority_score` only, ties broken **pessimistically**; budgets k = max(1, round(N × budget)), so **test k = 1 (1%) and 6 (5%)**.
- Ch 9: the trainer stores the wake threshold **inside the ONNX**; `wake_triggered` must equal `sound_class == "drone_buzz"` (true for every labelled clip).
- Test labels are absent: evaluators **can't** score test. Never try to "check" test.
- Ch 8/9 trainers default to CUDA else CPU: pass `--device mps` on Apple silicon.
- Voyager SDK: Linux (Ubuntu 22.04+) or Windows **via WSL2** only; **not macOS**.
- Ch 6 test predictions: `predict_onnx_baseline.py detect --split test` works (the test COCO file lists all 328 images with 0 annotations). `predict ... mot` writes **all 8** sequences (valid + test) by default.
- `train_rfdetr.py` expects a dataset folder with **`train/`, `valid/` and `test/`**, each with `_annotations.coco.json`. Keep that structure when building the tiled dataset, and **don't use `--run-test`** (test has no labels).

---

## 1. The project in one minute

**Siesta** is an edge sensor node that finds small drones while spending as little compute, attention and bandwidth as possible. It **naps until there's real trouble**:

1. **Listen** (Challenge 9). A tiny acoustic model is always on. Camera off unless it hears a drone. *Eyes closed, ears open.*
2. **Look** (Challenge 6). When woken, the camera detects and tracks aircraft. Even here it **"peeks before it stares"**: cheap motion detection points to where to look, and the detector examines only those spots closely.
3. **Send** (Challenge 8). Over a weak link, only the top 1% or 5% of frames go out, ranked by priority. **It ranks clips, not frames**, matching how the labels were made.

All three run on the **Axelera Embedded 110m** (Metis chip, 1 GB memory), measured on real hardware in the Spectrum Room.

**Tagline:** "We nap so you don't have to." **Opening hook [verified via news reports]:** on 3 Feb 2020 a drone sighting stopped all flights at Madrid-Barajas between about 12:30 and 14:30, siesta time. 26 flights were rerouted, and the Civil Guard later found no proof the drone existed. Siesta's answer: confirm it (track + evidence frames) or rule it out ("reopen the runway").

**Sponsor endorsement [verified]:** the Ch 9 kit's own "Ideas to Win" lists a **two-stage cascade**: wake with the acoustic model, then verify on the paired video with the Ch 6 RF-DETR detector. The kit also notes drone rotors have **narrowband harmonic signatures**, and that **9 training clips have a drone that's heard but not seen** ("the ears catch what the eyes miss").

**Naming trap [verified]:** inside the kits the numbering differs. **Ch 6 = "Challenge 1"** (`challenge-1-cuas-tracking`), **Ch 8 = "Challenge 4"** (`challenge-4-edge-triage`), **Ch 9 = "Challenge 5"** (`challenge-5-acoustic-visual-cuas`). Always say which numbering you mean.

---

## 2. Verified facts from the kits

### Ch 6: detection + tracking (kit "Challenge 1")
- **Classes (0-based):** `0 drone_air`, `1 airplane`, `2 helicopter`.
- **Data:** train 859 images (from **14 clips only**), valid 422 (8 clips), test 328 (7 clips, labels held out). Negatives in all splits. Splits are leak-free by clip. 200 calibration images (train-only, 384×384).
- **Small targets dominate [verified]:** 270 of 387 valid boxes are smaller than 32×32 px; median box area is about 308 px².
- **Tracking sequences:** 8 MP4s at 30 fps. **Valid = seq_0001–0004** (GT available), **test = seq_0005–0008** (GT held out). **Each valid sequence contains exactly one object, present in every frame [verified].**

  | seq | resolution | target | median box |
  |---|---|---|---|
  | 0001 | 720×1278 | helicopter | 211×110 px |
  | 0002 | 352×640 | airplane | 32×24 px |
  | 0003 | 1920×1080 | drone | **10×11 px** |
  | 0004 | 1080×1920 | drone | **7×7 px** |

  Test sequences may differ (e.g. multiple objects), so **never hard-code "one object".**
- **Recommended model:** RF-DETR **Nano at 384×384**. The kit ships **no trained weights [verified]**, so we train it ourselves (`train_rfdetr.py`, which downloads pretrained weights). **YOLO/ultralytics are not used in this challenge.**
- **Why the baseline fails on tiny drones [our analysis]:** resizing a 1920 px frame to 384 shrinks a 7 px drone to about 1.4 px. **Native-resolution tiles/crops are essential.**
- **Baseline (1-epoch smoke test, CPU ONNX):** mAP@50 0.447, mAP@50:95 0.293, small_AP50 0.506. Tracking **MOTA −0.013, IDF1 0.248**, continuity 0.432, 6 ID switches. Baseline tracker: IoU matching at 0.25, score threshold 0.25.
- **Scorer facts [verified from code]:**
  - Tracking is class-agnostic, with a **per-frame IoU ≥ 0.5** match.
  - MOTA = 1 − (FN + FP + IDSW) / GT.
  - **A box in the wrong place costs 2 (a false positive and a missed object); outputting nothing costs 1.**
  - IDF1 also counts every predicted row.
  - Continuity = share of GT frames covered.
- **Formats [verified]:**
  - `preds.json`: list of `{"image_id": int, "category_id": 0-based int, "bbox": [x,y,w,h], "score": 0..1}`, top 100 per image.
  - `pred_mot/seq_XXXX.txt`: rows `frame(1-based),track_id,x,y,w,h,conf,class_id(1-based),1.0`, with no duplicate (frame, track_id).
- **Proposed judging weights:** Detection 40% (mAP@50:95, small_AP@50) · Tracking 40% (MOTA, IDF1, continuity) · Edge efficiency 20% (Metis throughput, memory fit < 1 GB).
- **Metis reality [verified]:** full RF-DETR uses a GridSample deformable-attention tail. The kit's **backbone-only export** (`features [1,256,24,24]`) is the Metis starting point; the **decoder must run on the host and be wired up by us**. No compiled Metis baseline exists. ONNX outputs must be **indexed by name** (`dets`, `labels`), because both have shape `[1,300,4]`.

### Ch 8: edge triage (kit "Challenge 4")
- **Data:** 947 frames / 282 clips; train 698 (212 clips), valid 138 (41), test 111 (29). Mostly **ground-level battlefield footage** (train perspective: 622 ground, 42 from_drone, 28 other, 6 of_drone). Scene tags: armored_vehicle, vehicle, person, drone (117 train frames), explosion_smoke, trench_building. RGB/night; one thermal frame in valid.
- **Labels are clip-level [verified]:** every frame in a clip has identical keep/priority labels (0 of 41 valid clips differ), and **test rows include `video_id` [verified]**.
- **Scoring [verified from code]:**
  - Rank by `priority_score`; ties are broken pessimistically.
  - Budget k = max(1, round(N × budget)). **On test (N = 111): k = 1 for the 1% budget and k = 6 for 5%**, so the 1% score is effectively all-or-nothing.
  - "High" = keep and priority ≥ 0.7.
  - Spearman rank correlation is also reported.
- **Baseline (2-epoch smoke):** capture@1% **0.0**, capture@5% 0.286, **Spearman −0.157** (worse than random). The trainer uses MSE on priority with **no augmentation** [verified].
- **Format:** JSONL `{"id": str, "triage_keep": bool, "priority_score": float}`, one line per GT id.
- **Proposed weights:** budget capture 55% · Spearman 25% · Metis INT8 throughput/power/footprint 20% (**bonus for 100% on-chip execution and for temporal clip dedup**).

### Ch 9: acoustic wake (kit "Challenge 5")
- **Data:** 62 five-second clips (16 kHz WAV + synced 30 fps MP4 + 224×224 spectrogram PNG). Train 48 (15 drone), valid 8 (2 drone), test 6. 48 calibration spectrograms (train-only).
- **Labels [verified]:** `has_drone_audio == (sound_class == "drone_buzz")` for **every** train and valid clip.
- **Baseline (2-epoch smoke):** wake F1 0.80, class accuracy 0.50. The trainer has **no augmentation** [verified] and supports MobileNetV3-Small or ResNet-18. The wake threshold is chosen on valid and stored in the ONNX model.
- **Format:** JSONL `{"id": str, "wake_triggered": bool, "sound_class": "drone_buzz"|"vehicle_mechanical"|"explosion"|"ambient_outdoor"}`.
- **Scorer:** wake F1, class accuracy, optical duty cycle. Its key `energy_savings_pct` is **not measured energy**; call it "camera-off time".
- **Proposed weights:** wake F1 40% · class accuracy 25% · camera-off proxy 20% · Metis latency/footprint 15%.

### Environment (all three kits)
- Python ≥ 3.11, `uv` (`uv venv && uv pip install -e .` in each `starter-kit/`). Ch 6 tracking needs `ffmpeg`.
- **Metis compile:** Voyager SDK `axcompile -i <model.onnx>` plus a `--transform preprocess.py` **we write** (implementing `get_preprocess_transform(image)`: RGB resize, float32 CHW, [0,1] scaling, and **ImageNet normalisation only for RF-DETR**, since the CNNs already include it in the graph). Calibrate only with each kit's calibration folder.

---

## 3. What we tested ourselves [verified, Sat morning]

1. **Motion proposals on Ch 6 valid sequences.** Camera-motion-compensated three-frame differencing (`scripts/motion_probe.py`) finds the target in **99/150, 68/72, 145/150 and 124/150 frames** (seq 1–4). That includes the 7–10 px drones.
   - It also produces **about 34, 14, 1,151 and 197 candidates per frame**: moving trees and compression noise.
   - Its boxes are poor on tiny targets (median IoU 0.13 and 0.29 on seq 3 and 4).
   - A naive motion-only tracker scores **MOTA −0.56, IDF1 0.17**, worse than the baseline.
   - **Conclusion:** motion is an excellent *proposal* source but **must be confirmed and re-boxed by the detector**. Never ship motion alone.
3. **Full test suite (Sat 12:05): 15/15 PASS.** Details in `TEST_REPORT.md`. New finding: motion windows contain the 7 px drone of seq 4 in only **19%** of non-sweep frames (12 windows; 1% with 6). Seq 2: 97%, seq 3: 95%. **Decision: use `detect_video.py --mode full` for scored MOT files; use motion mode (default now 12 windows) for the efficiency story** (about 48% fewer detector calls).
2. **Physics-only drone-sound check on Ch 9.** Harmonicity/peakiness features (`scripts/harmonic_probe.py`) point the right way but alone reach only **leave-one-out wake F1 0.31** on train+valid.
   - **Conclusion:** use them at most as an **auxiliary** feature. The learned model stays primary.

---

## 3b. Hardware plan (decided Sat 11:10)

**Key constraint [verified, Axelera docs]:** the Voyager SDK installs on **Linux (Ubuntu 22.04+)**, and on Windows only through **WSL2**. **macOS isn't listed.** So **no Mac can compile for Metis.** Compilation and benchmarking happen on the **Axelera host in the Spectrum Room** (or a teammate's Linux/WSL2 laptop, if the compiler-only `axelera-devkit` install works without a Metis card, which the docs leave unclear).

| Machine | Who | Used for | Fallback |
|---|---|---|---|
| **M4 Max MacBook Pro, 48 GB** | **P1 only** | Ch 6 RF-DETR Nano training on MPS; tiled inference; ONNX export | Free cloud NVIDIA GPU (Colab / Kaggle) for detector training if 1-epoch timing says MPS is too slow |
| P2 laptop | P2 | Tracker development (CPU is enough: ≤ 150-frame sequences); `ffmpeg`, OpenCV; uses P1's latest ONNX | Any laptop |
| P3 laptop | P3 | Ch 9 + Ch 8 training and cross-validation (tiny CNNs: CPU is fine; `--device mps` if Apple silicon) | Any laptop |
| **The team's Windows laptop → P4** | P4 | **WSL2 + Ubuntu 22.04** for the Voyager SDK route (the only Windows path Axelera documents); SSH to the Axelera host; optional compiler-only install; eval scripts; demo editing; slides | If WSL2 can't be enabled (admin/virtualisation blocked), work directly on the Axelera host |
| **Axelera Embedded 110m + its host PC** (Spectrum Room) | P4 (+P1 for Ch 6) | `axcompile`, INT8 calibration, on-device accuracy, latency, memory. **Also runs our host-side tail** (Ch 6 decoder + tracker), because Ch 6's edge score is *end-to-end* throughput, so the host-tail time must be measured on that host, not on the Mac | Report CPU ONNX Runtime numbers, clearly labelled |
| Phones | P4 | Demo capture (video + audio, 30 fps) if a real-drone recording is allowed; hotspot backup for internet | Recorded dataset demo |

**Bring or get today:**
- USB stick or SSD (≥ 16 GB) to move data and models to the Axelera host if the network is slow.
- USB-C→HDMI adapter for the pitch.
- Chargers and a power strip.
- Headphones, to listen to Ch 9 clips in the kit viewer.

**Not needed:** 3D printers, Fire Point equipment, the Alta Ares simulators. Featherless is optional, as a coding assistant only, never inside the pipeline.

**Memory fit [estimate, verify on device]:** Ch 6 FP32 ONNX is 102.7 MB (full) / 89.0 MB (backbone); INT8 should be roughly a quarter of that. Ch 8 and Ch 9 are about 3.6 MB each. All three should fit in the 1 GB card memory at once, which is part of the "one chip, three models" claim, so **measure it**.

**Bottlenecks to manage:**
1. **One fast machine:** the M4 Max is reserved for Ch 6 training. Everyone else works on CPU.
2. **Shared Metis devices:** book slots early, and arrive with ONNX + `preprocess.py` + calibration folder ready.
3. **Venue Wi-Fi:** install PyTorch, rfdetr and the RF-DETR pretrained weights **now** on every machine; use a phone hotspot if slow.
4. **Compile risk:** CNNs first; the RF-DETR backbone has its own cut line at 14:00.

**Every teammate, report now in the team chat:** laptop chip, RAM, OS (macOS / Windows / Linux), and whether Python 3.11+ and `ffmpeg` are installed.

---

## 4. The approach: what's unusual, and why it's feasible

| # | Idea | Where | Why it works (evidence) |
|---|---|---|---|
| A | **Zoom where it matters:** train and infer on native-resolution 384 px tiles/crops instead of shrinking whole frames | Ch 6 detection | Tiny boxes vanish when downscaled 3–5× (Sections 2–3). Tiles keep 7 px drones at 7 px. |
| B | **Peek before you stare:** motion proposals → detector crops around them → tracker. A slow full-frame tile sweep still runs so motionless targets aren't missed. | Ch 6 tracking | Motion finds the targets (Section 3), the detector removes clutter and fixes boxes, and compute is spent only where something moved. That's Siesta inside Stage 2 too. |
| C | **Scorer-aware output policy:** only emit a box when the track is confirmed and well localised; fill short gaps only between confident detections | Ch 6 tracking | The scorer charges 2 for a misplaced box and 1 for silence. Baseline MOTA is negative because of false positives. |
| D | **Two thresholds:** low for COCO detections (mAP rewards recall), strict for MOT output | Ch 6 | Each metric rewards a different behaviour (scorer code). |
| E | **Rank clips, not frames:** pool predictions across each `video_id`, plus a tiny per-frame tie-breaker | Ch 8 | Labels are clip-constant and test rows carry `video_id` (verified). The kit offers a bonus for temporal clip dedup. |
| F | **Train for order:** a pairwise/listwise ranking loss on priority (keeping the other heads as auxiliary), image augmentation, clip-grouped cross-validation | Ch 8 | Eval is ranking-based. Baseline Spearman is −0.16, using MSE with no augmentation. |
| G | **One decision, consistent outputs:** `wake_triggered = (sound_class == "drone_buzz")`, from one calibrated model, so wake and class never contradict | Ch 9 | Verified label identity on all 56 labelled clips. |
| H | **Sound-mixing augmentation + cross-validation:** mix drone audio over ambience at varying loudness, time shifts, gain, spectrogram masking; pick settings by CV over train+valid; compare MobileNetV3-S vs ResNet-18 | Ch 9 | The trainer has no augmentation. 8 valid clips (2 positive) overfit easily, as the kit itself warns. |
| I | **One chip, three models, one cost ledger:** compile the Ch 9 and Ch 8 CNNs first (easy, fully on-chip), then the Ch 6 backbone with a host decoder. Measure latency, memory and INT8 accuracy drop for each. | All | 15–20% of each challenge's proposed score is Metis efficiency, and there are no compiled baselines, so few teams will have these numbers. |

### Priority ranking and scope (decided Sat 11:00 after review)
- **Priority 1, Ch 6 (two people, P1 + P2):** win it outright. Most teams, most points, biggest headroom. Tiled detection (A) + motion proposals confirmed by the detector (B) + scorer-aware output policy (C/D).
- **Priority 2, Ch 9 (P3, first):** fully on-chip, and the sponsor's own recommended cascade. Ideas G + H.
- **Priority 3, Ch 8 (P3, second):** cheap, 1 competing team. **Only** clip pooling (E) + ranking loss (F) + a Metis benchmark. Nothing more.
- **Cut:** physics sound features (tested weak); second seeds and extra-long Ch 6 runs unless the GPU is idle; Ch 6 host decoder if the backbone doesn't compile by 14:00.
- **Ambitious demo (optional, needs organiser permission and their safety rules):** film and record a **real drone** at the venue (Alta Ares has RC drones) and run Siesta on it on the Metis chip. Fallback: the recorded cascade on Ch 9 paired clips.

**Deliberately not doing:** hard-coding one object per sequence, tuning on test, claiming energy savings, claiming Ch 6 detections feed Ch 8 ranking (Ch 8 is ground battlefield footage), and appearance re-identification for tiny targets.

---

## 5. Architecture

```
mic (5 s windows) ─► [Ch 9 wake model on Metis] ─ no ─► camera stays off (log camera-off time)
                                     └ yes ─► camera on
camera frames ─► [motion proposals, host] ─► crops ─► [Ch 6 RF-DETR: backbone on Metis + host decoder]
                 [slow full-frame tile sweep] ─┘                 │
                                                                 ▼
                                   [tracker: camera-motion comp + Kalman + confirm/suppress + gap fill]
                                                                 │
                                   track messages (always, tiny) ┼─► operator / map
frames ─► [Ch 8 triage model on Metis] ─► clip-pooled priority ─► top 1% / 5% uplink
```

**Scoring:** each challenge is scored separately (Ch 6 COCO + MOT, Ch 8 JSONL, Ch 9 JSONL). The cascade is the **demo and pitch layer**, built on Ch 9's paired video.

**Track message contract (for the demo):** `{frame, track_id, class, bbox_xywh, confidence, timestamp}`.

---

## 6. Roles

**GPU priority (we have one GPU resource; adjust if we have more):**
1. P1's Ch 6 detector training comes first.
2. P3's Ch 8/9 cross-validation runs go in the gaps. These small CNNs also train on a laptop CPU if the GPU is busy.
3. Nobody leaves an idle job holding the GPU. Log every run in the experiment log.

### P1: Ch 6 detection ("zoom where it matters")
1. Set up the environment on the GPU machine and run the kit's quickstart end to end: `train_rfdetr.py --model nano --epochs 20` → export → predict → evaluate. That gives the **`v0` safety submission with a properly trained baseline** (not the 1-epoch smoke model). Record time per epoch and GPU memory use. Start with the kit defaults (batch 8, grad-accum 2); if memory runs out, halve the batch and double grad-accum.
2. **Build a tiled training set:** native-resolution 384×384 crops around every box (with jitter) plus random negative crops from negatives and backgrounds. Keep COCO format, so `train_rfdetr.py` works unchanged.
3. **Train RF-DETR Nano on the crops** (the kit's recommended deployment model; keep Nano as the deliverable). With the GPU, run a clean comparison, all scored with the kit evaluator on valid:
   - (a) baseline, whole frames;
   - (b) tiles;
   - (c) tiles + more epochs (e.g. 40–60);
   - if time allows, (d) a second seed of the best setting.
   Keep whichever wins. Launch (b) as soon as the tiled set exists.
4. **Tiled inference** for COCO output: overlapping 384 tiles at native resolution plus one downscaled whole-frame pass for large aircraft. Merge tiles with class-wise NMS. Keep top 100 per image with a **low** score threshold (idea D).
5. Evaluate with the kit's `evaluate_submissions.py detect` only.
6. **Hand to P2:** a function `detect(crop_or_frame) → [(cls, score, box)]`. **Hand to P4:** the exported ONNX (full + backbone).

### P2: Ch 6 tracking ("peek before you stare" + scorer-aware policy)
1. Start from `scripts/motion_probe.py` (motion proposals with camera-motion compensation).
2. **Proposal → confirm:** run P1's detector on crops around motion candidates (merge overlapping candidates; cap the count per frame by motion strength) **plus** a periodic full-frame tile sweep.
3. **Tracker:** camera-motion-compensated Kalman filter; center-distance association for tiny boxes (IoU is unreliable at 7 px); confirm after N consistent hits; delete after M misses.
4. **Output policy (idea C):** emit only confirmed tracks with confident boxes. Smooth box size (keep the detector's box, not the motion blob). Interpolate short gaps only between confident detections. Use a strict threshold.
5. **Tune with leave-one-sequence-out over the 4 valid sequences.** Report per-sequence numbers, never just the best one.
6. Evaluate only with `evaluate_submissions.py mot --split valid`. **Write test MOT files for seq_0005–0008** with the same settings.

### P3: Ch 9 and Ch 8 (the two small CNNs)
**Ch 9:**
1. Baseline run; log wake F1, accuracy, duty cycle.
2. **Augmentation (idea H):** waveform mixing (drone + ambience/vehicle at random SNR), time shift, gain, then regenerate log-mel spectrograms **exactly like the kit's PNGs** (check the kit's spectrogram parameters; if unclear, augment the PNGs with time/frequency masking instead).
3. **Consistency (idea G):** a single decision drives both outputs.
4. Select settings with **cross-validation over all 56 labelled clips**. Then set the final threshold, document it, and store it in the ONNX the way the starter does. (Physics features are cut; see Section 4.)

**Ch 8:**
1. Baseline run.
2. **Ranking loss** on priority (pairwise or ListNet) plus the existing auxiliary heads.
3. Image augmentation (flips, crops, colour jitter, night-like darkening).
4. **Clip-grouped CV.**
5. **Clip pooling at inference (idea E):** mean or max of frame scores per `video_id`, plus a tiny frame-level tie-breaker. Compare pooled vs. unpooled on CV and keep the winner.
6. Remember k = 1 at the 1% budget on test: make sure the single top slot is the most confident clip.

**Hand to P4:** both ONNX files (static shapes, opset 17, as the starters export). On the Mac, run both trainers with `--device mps` (or the CPU if the GPU is busy with Ch 6).

### P4: Metis, integration, pitch (safety owner)
1. **Spectrum Room:** confirm the hardware and SDK version, book device time, and fill in Section 0.
2. **Compile order:** Ch 9 CNN → Ch 8 CNN → Ch 6 backbone (host decoder separately).
3. Write `preprocess.py` per model (Section 2 rules). INT8-calibrate with each kit's calibration folder **only**.
4. **Cost ledger:** for every model, record FP32 vs INT8 accuracy on valid, device latency, throughput and memory. **Label every number with where it ran** (Metis vs. laptop CPU ONNX Runtime).
5. **Format checks:** run each kit's evaluator before every submission. Keep tagged versions `v0/v1/v2`.
6. **Cascade demo** on Ch 9 paired clips: audio wake timeline → camera on → P1/P2 boxes with IDs → P3 triage strip with an uplink-bytes counter. Record it as a video.
7. Own the slides and rehearsals.

---

## 7. Checkpoints

| When | Must be true |
|---|---|
| **Sat 12:00, Gate 0** | Every box in Section 0a ticked, or given an owner and a fallback. |
| **Sat 13:00** | Valid baseline submissions for **all three** (Ch 6 COCO + MOT, Ch 8 JSONL, Ch 9 JSONL), scored with the kit evaluators and tagged `v0`. Ch 9 CNN compiled on Metis (or a written blocker). |
| **Sat 14:00, cut line** | If the Ch 6 backbone won't compile, stop and report Ch 6 on CPU ONNX Runtime INT8. Ch 8/9 stay the on-chip proof. |
| **Sat 19:00** | Tracker v1 beats baseline MOTA/IDF1 under leave-one-out. Ch 8/9 improved on CV. Cascade runs on ≥ 1 clip. Submissions `v1`. |
| Sat afternoon | Tiled detector run (b) launched on the GPU by about 14:00. Results by early evening feed P2's tracker. |
| Sat night | Best detector setting trained longer or with a second seed on the GPU. P3's final Ch 8/9 models trained. Pitch outline drafted. |
| **Sun 11:00** | Final test predictions `v2`, re-validated, submitted early (once deadlines are confirmed). |
| **Sun 12:00** | Feature freeze. Demo video recorded. Afternoon: slides and 3 timed rehearsals. |

**If something has to give:** reduce scope, never validity. The order is: Ch 6 Metis host decoder → Ch 6 extra training runs → Ch 8 ranking loss (keep pooling). **Never** drop a submission.

---

## 8. Pitch (Siesta)

1. **Hook:** Barajas, 3 Feb 2020. Two hours closed at siesta time, 26 flights rerouted, a drone nobody could prove existed.
2. **Problem:** small drones are hard to see, and watching for them costs power, attention and bandwidth.
3. **Siesta:** eyes closed, ears open. Listen, then wake and peek, then stare, then send only what matters.
4. **Demo video.**
5. **Results vs. baseline**, per challenge, with the counts behind them (small validation sets stated honestly).
6. **One chip, three models:** the cost ledger measured on the Axelera device.
7. **What's different:** tiles for tiny drones; motion proposals confirmed by the detector; a scorer-aware tracker; clip-level ranking; consistent wake/class decisions.
8. **Back to Barajas:** "drone confirmed, here's the track and three frames", or "no drone, reopen the runway".
9. **Next:** confirmed tracks feed a C2 / fleet planner (Challenge 4 link); field data that labels itself; dual use (airports, prisons, infrastructure).

**Tone:** the humour lives in the name, the tagline and the hook. Everything after slide 3 is rigorous. Don't call it a "defense system" (it detects and confirms). Lean on the Barajas story more than the siesta cliché unless a Spanish teammate presents.

---

## 9. Rules (non-negotiable)

1. Never tune on test. Use valid / cross-validation only.
2. Calibration folders are for INT8 calibration only.
3. Only the kits' own evaluators produce reportable numbers.
4. Always keep a valid, tagged submission. Validate formats before uploading. Submit early.
5. Every performance number states model, precision (FP32/INT8) and hardware (Metis vs CPU).
6. No overclaiming: "camera-off time", not energy. Small test sets mean noisy scores, and we say so.
7. Don't hard-code dataset quirks (e.g. one object per sequence).
8. No secrets in the repo (Featherless and other keys in environment variables).
9. **Experiment log row:** `time | owner | challenge | change | config/commit | metric (eval script) | notes`.

---

## 10. Decision history and context (so every Claude knows what the team lead's chat knows)

- **Event:** EDTH Paris, 20–22 Nov 2026. Team of 4. About 24 hours of work from Sat 11:00. Pitch Sunday evening if preselected. Goal: win, as ambitiously as is feasible.
- **All 9 challenges were reviewed** (full detail in `EDTH_ALL_CHALLENGES.md`). Outcome:
  - **Chosen: Ch 6 + Ch 8 + Ch 9 = "Siesta"** (all Piotr Gryko / Axelera). They chain naturally (hear → look → send), and the Ch 9 kit itself recommends the acoustic → RF-DETR cascade. Low competition (4 / 1 / 3 teams at registration). Axelera hardware on site.
  - **Ch 4 (Alta Ares Fleet Orchestration):** not entered. No technical link (its only input is a scenario JSON); its kit is Python-stdlib-only with a Sunday 12:00 hand-in. Appears only as a "next step" in the pitch.
  - **Ch 3 (Alta Ares Search & Destroy):** not entered. It's autonomous strike guidance, which the team's assistant (Claude) won't help develop.
  - **Ch 5 (jet interceptor):** infeasible (needs jet-engine facilities).
  - **Ch 1 (Voice-to-Map):** crowded (9 teams), no scorer.
  - **Ch 2 (stabilised antenna):** needs controls and CAD skills.
  - **Ch 7 (RGB + thermal):** good, but outside our chain.
- **Name and story:** "Siesta", with "Eyes closed, ears open" and "We nap so you don't have to". **Madrid hook:** the Madrid-Barajas drone closure on 3 Feb 2020 (~12:30–14:30, 26 flights rerouted, drone never confirmed). Don't call it a "defense system"; it detects and confirms.
- **Hardware decided:** M4 Max MacBook Pro 48 GB = the training machine (P1); the one Windows laptop = P4 (WSL2 for the Voyager SDK); the Axelera host in the Spectrum Room for compile and benchmarks; free Colab/Kaggle GPU as backup for detector training.
- **Tests run on Saturday morning (Section 3):** motion proposals find the tiny Ch 6 drones but need detector confirmation (motion-only MOTA −0.56). The physics-only Ch 9 sound features are weak (LOO F1 0.31), so they were **cut**.
- **Corrections made along the way:** an early idea to merge Ch 4 with Ch 6 was dropped (scored separately, no shared input). The Ch 4 repo turned out to be stdlib-only with no link-loss part. The physics wake check was downgraded after testing.
- **Scope review (Sat 11:00):** a teammate's Claude flagged the plan as too broad. Resolution: **Ch 6 is priority 1 (2 people)**, Ch 9 priority 2, Ch 8 priority 3 (only clip pooling + ranking loss + Metis benchmark). Extras cut (Section 4).
- **Files in the team project:** `EDTH_TEAM_BRIEF.md` (this file), `EDTH_ALL_CHALLENGES.md` (all-challenge reference), `scripts/motion_probe.py`, `scripts/harmonic_probe.py`.
- **Code kit (Sat 11:50), `siesta/` in the project and `siesta_code.zip`; start with `siesta/README.md`:**
  - **P1:** `c6/build_tiled_dataset.py`, `c6/tiled_detect.py` (+ `c6/tiling.py`)
  - **P2:** `c6/detect_video.py` (full / motion modes), `c6/track.py`, `c6/synth_test.py`
  - **P3:** `c9/train_cv.py`, `c9/predict_consistent.py`, `c8/train_rank.py`, `c8/pool_clips.py`
  - **P4:** `metis/preprocess_cnn.py`, `metis/preprocess_rfdetr.py`
  - **Everyone:** `check_submissions.py`, which must print PASS before any upload.

  All of it was run end to end against the real kit data and **scored with the kit evaluators**. The test environment had no pretrained weights, so its model scores are meaningless; only the mechanics were verified. Extra dependency: `opencv-python`.
