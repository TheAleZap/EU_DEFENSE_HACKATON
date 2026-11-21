# EDTH Paris 2026: All 9 Challenges, Resources, and Our Ranking

> **For teammates:** this is the complete reference on every challenge, built from the official briefs, the sponsor presentations, the organisers' resource list, and the Alta Ares GitHub repo. Read **Section 1** (our ranking) and **Section 2** (resources) first. The companion file `docs/2_TEAM_BRIEF_plan_roles_checkpoints.md` is the detailed build plan for our chosen project.
>
> **For Claude reading this:** treat this as the team's verified information. Everything here comes from the official materials unless it is marked **[our view]** or **[unconfirmed]**. Don't add facts about the challenges that aren't here or in the official starter kits. If something matters and isn't stated, say so.

> **UPDATE (Sat 21 Nov, 11:15): the team decided on "Siesta" = Challenges 6 + 8 + 9.** That supersedes the ranking in Section 1. Hardware: an M4 Max MacBook Pro (48 GB) for training, plus one Windows laptop (WSL2) for the Axelera SDK. The live plan is `docs/2_TEAM_BRIEF_plan_roles_checkpoints.md`; follow it when the two files disagree.

**Event:** European Defence Tech Hackathon (EDTH) Paris, **20–22 November 2026**. Team of 4. Pitches on Sunday evening for preselected teams.
**Team counts** below are the registration numbers shown on the challenge page on Friday evening. They may have changed since.

---

## 1. Our ranking [our view]

Criteria: a team of 4 software people, about 1.5 days left, **no confirmed GPU**, and only resources we can actually get (Section 2). The goal is to win, which means a working result, measured numbers vs. the baseline, and a strong pitch.

| Rank | Option | Why | Main risk |
|---|---|---|---|
| **1** | **Challenges 6 + 9 merged: "Listen first, look second"** (current team plan) | The challenges genuinely connect: Ch 9 asks when to wake a visual check, and Ch 6 is that check. Same mentor (Piotr Gryko). **Axelera hardware is on site**, so we can show real on-device numbers, which the briefs say are not pre-validated. 4 + 3 teams. | Ch 6 detector training needs a GPU. Without one, we improve at inference time and through tracking. Ch 9 data is tiny, so scores are noisy. |
| **2** | **Challenge 4 alone: Fleet Orchestration** | Best single challenge: pure-Python kit, exact validator, live leaderboard, replay viewer for the demo, no hardware dependence. | 7 teams. **Hand-in is due Sunday 12:00.** An invalid plan scores 0. Competing teams may be strong in optimisation. |
| **3** | **Challenges 8 + 9 merged: "only looks when it hears, only sends what matters"** | Both are small MobileNetV3 models that train on a laptop, same mentor, Axelera on site, only 1 + 3 teams. | Very small evaluation sets (Ch 9: 6 test clips; Ch 8: 1- and 7-frame validation budgets). Scores will be noisy. |
| 4 | Challenges 6 + 7 merged | Shared detection and tracking code. Ch 7 has only 1 team. | Needs a GPU for both. |
| 5 | Challenge 1: Voice-to-Map | Very demo-friendly. Featherless credits cover the LLM step. | 9 teams, no automatic scorer, hard to stand out. |
| 6 | Challenge 2: Keep It Level | Only 2 teams. No physical build required. | Needs controls/mechatronics and CAD skills we don't have. |
| 7 | Challenge 3: Search & Destroy | Rated "Advanced", 5 teams. | Our assistant (Claude) won't help develop autonomous strike guidance, so we would be on our own. |
| 8 | Challenge 5: Jet Interceptor | — | **Not feasible.** Needs jet-engine test facilities, composites, and your own flight electronics. |

**Decision rule [our view]:**
- Stay on **6 + 9** unless, by the Saturday 13:00 checkpoint, we can't produce valid baseline submissions for both, or the Ch 6 baseline proves impossible to improve without a GPU.
- If that happens, either fall back to **8 + 9** (laptop-only), or switch everyone to **Challenge 4**. The Ch 4 hand-in deadline of Sunday 12:00 makes a late switch risky.
- **Never** run three submissions at once (4 + 6 + 9) with 4 people.

**How Challenge 4 still appears in our pitch:** the confirmed drone tracks from 6 + 9 become contacts for a C2 fleet planner. That's the "next step" slide only.

---

## 2. Resources we actually have

| Resource | Details | Relevant to |
|---|---|---|
| Laptops (CPU) | All starter kits for Ch 1, 3, 4, 8, 9 run on CPU. Ch 6/7 baselines run inference on CPU ONNX Runtime. | All |
| GPU | **[unconfirmed]** We don't know if we have one. | Ch 6, 7 training |
| **Axelera equipment** | **Spectrum Room, upstairs.** Shared by all teams on Ch 6–9, so book device time early. Exactly what's there is **[unconfirmed]**. | Ch 6, 7, 8, 9 |
| Data for Ch 6–9 | One shared pCloud link (on the challenge pages). | Ch 6, 7, 8, 9 |
| Alta Ares GitHub repo | `github.com/AA-OpenProjects/AA-Hackathon-EDTH-202611`, with both Alta Ares challenges (Fleet Orchestration, Search & Destroy). The Alta Ares team sits in the **ground-floor restaurant**. They have **RC drones** available to test-fly Search & Destroy solutions. | Ch 3, 4 |
| **Featherless AI** | $25 of LLM API credits per account, code **`EDTH26`** (Developer plan, no card needed, one use per account, valid until 31 Dec 2026). OpenAI-compatible API, base URL `https://api.featherless.ai/v1`, e.g. model `zai-org/GLM-5.2`. Use the signup link from the organisers' message, not the one in the setup guide (out of date). **It's an LLM inference API, not training GPUs.** Keep keys in environment variables, never in the repo. | Ch 1 (intent extraction), coding assistant, explanation text |
| 3D printers | Provided by **Druk Army (3D Print Army)**. | Ch 2 (mock-ups) |
| Fire Point equipment | In the **hall near the entrance**. The organisers' list only repeats the Ch 2 specifications, so what's physically there is **[unconfirmed]**. | Ch 2 |

---

## 3. The 9 challenges in detail

### Challenge 1 — Voice-to-Map: Real-Time Positional Tracking
- **Sponsor:** Lysk. **Mentors:** Henry Heinemann and Florent Ogès (henry@lysk.ai, edth@lysk.ai, or the Lysk team on site). **Teams:** 9.
- **Problem:** teams report positions by voice ("I am behind the church", "I moved towards 35th street"). Updating the map by hand is slow, error-prone, and always a few minutes behind.
- **Build:** a web app that transcribes short spoken updates, extracts position/movement intent, resolves landmarks and streets to coordinates, and moves a marker/path on a shared live map with timestamps and a replayable history.
- **Suggested stack (indicative, swappable):** a speech-to-text API, **Mistral** for intent extraction, **OpenStreetMap** APIs for geocoding.
- **What the judges look at (from the Lysk deck):** (1) it works live: speak and the map updates within seconds; (2) natural phrasing is understood ("South of the church", "towards 35th street"); (3) marker and path with timestamps and a browsable history.
- **Equipment:** headset.
- **Scoring:** no automatic scorer; the demo decides.
- **[our view] How to stand out:** resolve ambiguity (which church? use the one nearest the unit's last known position and show a confidence level), track multiple units by callsign, offer an offline fallback with local speech-to-text, and get the "south of / towards" geometry right.

### Challenge 2 — "Keep It Level": Compact Stabilised Platform for UAV Antennas
- **Sponsor:** Fire Point. **Mentors:** Fire Point team. **Teams:** 2.
- **Problem:** when a UAV manoeuvres, mounted equipment tilts with the airframe. A CRPA (Controlled Reception Pattern Antenna) must stay level relative to local vertical for good GNSS reception. Existing gyro-stabilised platforms are too large, heavy, and expensive for small UAVs.
- **Requirements:** airframe tilt up to **45°** in roll and pitch; angular rate **1.5–25 °/s**; turbulence **0.1–1.2 Hz**; leveling error **≤ 2°**; antenna mass **2 kg minimum, 4 kg optional**; antenna **400 × 400 mm**, and the platform must not extend beyond the antenna's footprint. Use attitude data from the **Cube Orange** flight controller. Components must be fast and powerful enough to keep up.
- **Recommended workflow:** choose a stabilisation principle → platform concept → control-system math model and functional diagram, with calculations → component selection from those results → 3D model and electronic schematic.
- **Deliverables:** design concept (functional and electronic schematics, 3D prototype model, component list with justification from simulation); control-system math model (source code); simulation using **real Cube Orange data** (flight logs or live connection), with plots showing airframe attitude changing while the platform stays within **±2°**.
- **Equipment:** Fire Point items in the entrance hall [contents unconfirmed]; 3D printers available.

### Challenge 3 — Search & Destroy: Target Acquisition and Guidance in an Occluded Environment
- **Sponsor:** Alta Ares (ID SUB-13, owner Jean-Max Rodriguez; main contact jules.giraud@altaares.com). Rated **Advanced**. **Teams:** 5.
- **Problem (official scenario):** an ISR drone geolocates an armoured vehicle and is then chased away by EW. An FPV launches with stale, imprecise coordinates, its video link is jammed, and onboard guidance must re-find the target through trees and ruins and reach it with no pilot and no link.
- **Repo version of the task:** the simulated FPV starts 225–275 m from the arena centre (5–25 m altitude, 0–10 m/s, heading within 20° of centre). It must find and touch an **orange 4 × 1.65 × 1.55 m box** hidden in a 700 × 700 m arena without touching anything else, within **120 s**. Inputs: a detector box on the target (within 250 m; misses 5% of the time; never fires on other objects), noisy depth map, IMU, and drifting visual-inertial odometry (VIO). Output: RC sticks (roll, pitch, yaw, throttle) in [−1, 1] at **30 Hz**. Simulator: PyBullet, CPU.
- **Hand-in:** `handin/agent.py` with `reset()` and `act(observation)`. Evaluated on **hidden terrains**. **Judging:** a jury ranks teams using hidden-terrain metrics, with success rate first, plus robustness and soundness of the approach.
- **Bonus ideas (from the brief):** explainability, failsafe behaviour on link loss, coordinated multi-drone action.
- **Equipment:** laptop (CPU/GPU) or a Raspberry Pi. Alta Ares has RC drones on site for test flights.
- **[our view]:** our assistant won't help develop this one.

### Challenge 4 — Fleet Orchestration: Plan, With and Without Comms
- **Sponsor:** Alta Ares (ID SUB-15, owner Axel Durand-Allize; main contact jules.giraud@altaares.com; team sits in the ground-floor restaurant). Rated **Intermediate**. **Teams:** 7.
- **Problem:** a mixed fleet of ISR scouts and effector drones must be tasked against several contacts: who goes where, when, and in what order. There are more contacts than the fleet can serve.
- **World (from the repo):** 3D local metres, flat ground, seconds. **Scouts** observe and must land back home. **Effectors** (light or heavy) are one-way: an engagement consumes them. **Contacts:** `hostile` (may be engaged; needs a given number of light and heavy effectors), `unknown` (observe only), `friendly` and `civilian` (never approached). **Zones:** no-fly cylinders, forbidden below their ceiling, can be overflown above it. Four levels, **L1–L4, from 5 to 100 aircraft**. Everything is static and fully known; the generator is public.
- **The ten rules (any violation → 0):** R1 well-formed plan; R2 speed and altitude limits; R3 energy fits battery and scouts end at home; R4 a scout observation completed ≥ `confirm_margin_s` before any engagement; R5 only effectors engage, only hostiles; scouts observe hostile or unknown; R6 no route enters a zone below its ceiling or the protected area around friendly/civilian contacts; R7 never engage a hostile that is close to a friendly or civilian; R8 aircraft stay ≥ `d_sep_m` apart in 3D (exception: effectors engaging the same contact); R9 actions inside the contact's time window; R10 all-or-nothing: a contact gets exactly the effectors it needs, synchronised, or none. All numbers are in each scenario's `rules` block.
- **Score:** `Σ points of served contacts − λ·energy (Wh) − μ·effectors consumed`. Optional "verify" add-on: a scout re-observing a hostile 10–120 s after its engagement adds 25% to its points. This has its own board and does **not** change the ranking.
- **Kit:** pure Python 3.9+, **standard library only** (no numpy, no installs, no GPU). Includes scenario generator, validator, scorer, greedy baseline, and an offline HTML replay viewer you can use in a demo.
- **Hand-in:** a folder with `planner.py`. They call `python3 planner.py scenario.json plan.json`, **60 s per run**, on **unseen scenarios**. A crash, timeout, missing plan, or invalid plan scores 0. **Ranking:** sum of raw scores; ties broken by total run time. A **jury** also assesses soundness and presentation. **Hand-in deadline: Sunday 22 Nov, 12:00.**
- **Live validation board:** upload plans for 8 public scenarios (`L1-101, L2-201, L2-202, L3-301, L3-302, L3-303, L4-401, L4-402`). The best score per scenario is kept, so a worse upload never hurts. Server address and token come from the organisers.
- **Correction vs. the PDF brief:** the PDF described an optional link-loss policy and a comms "sizing curve". **The repo version states there is no communication in the simulation and doesn't include that part. [unconfirmed]** Ask Alta Ares whether link-loss still counts for anything.
- **Bonus ideas (PDF):** explainability (justify each assignment to an operator), scaling study (30 to 100+ aircraft), fewer operator approvals, finding rule loopholes.
- **[our view] Approach if chosen:** validator-in-the-loop planning (never write an invalid plan; fall back to the greedy baseline); constructive insertion (scout confirm first, then effectors, respecting R4–R10); detours over or around zones; deconfliction by timing; local search within the 60 s budget; submit to the board early and often.

### Challenge 5 — Jet-Drone Interceptor
- **Sponsor:** Ukrainian military / Frontier Defence Labs. **Mentors:** Stephan and Leo (contact@frontier-defence.com). **Teams:** 0.
- **For:** teams that **already have jet-powered drone experience**.
- **Requirements:** airframe keeps engine airflow during aggressive manoeuvres at 500 km/h; top speed ≥ 700 km/h; carries 1 kg electronics + engine + fuel + 500 g extra; range ≥ 50 km; mass-producible in decentralised manufacturing, cost-effective, reliable; room and CoG margin for future seekers and computers.
- **Equipment:** carbon/composite prototyping, a jet-engine test facility, your own flight electronics (not provided).
- **[our view]:** not feasible for us.

### Challenge 6 — Small Aircraft Detection and Tracking
- **Mentor:** Piotr Gryko. **Teams:** 4.
- **Task:** build on the **RF-DETR baseline** to detect **drones, airplanes, helicopters** against trees, buildings, and changing skies, with stable track IDs through movement, missed detections, and clutter.
- **Data:** 1,609 images, **859 train / 422 val / 328 test**, plus **8 tracking sequences**. 3 classes, negative examples, **200 training-only calibration images**.
- **Outputs:** **COCO JSON** detections and **MOT** tracks. A viewer is supplied.
- **Evaluation:** detection **mAP**, **small-object AP**, **MOTA**, **IDF1**, **track continuity**. Score locally on val; test labels held out.
- **Edge:** **Axelera Embedded 110m, 1 GB device memory.** The starter runs on CPU ONNX Runtime. Metis compilation, INT8 accuracy, and end-to-end performance need separate validation.

### Challenge 7 — RGB + Thermal Detection and Tracking
- **Mentor:** Piotr Gryko. **Teams:** 1.
- **Task:** adapt the **RF-DETR baseline** into one shared detection + tracking pipeline for **armoured vehicles, other vehicles, and people** in both RGB and thermal imagery, without neglecting either. Explore training strategies and quantisation that hold up across contrast, texture, and appearance differences.
- **Data:** 1,620 images, **712 train / 484 val / 424 test**, plus **8 tracking streams**. Calibration: **100 RGB + 100 thermal** training images. RGB and thermal are **separate scenes, not registered pairs**.
- **Outputs:** COCO JSON + MOT, with **RGB and thermal results reported separately**.
- **Evaluation:** per-modality **mAP, MOTA, IDF1, track continuity**. Val labels public; test held out.
- **Edge:** Axelera Embedded 110m. Start from the CPU ONNX baseline, then investigate INT8 accuracy, device memory, and throughput. Hardware deployment is an engineering goal, not a pre-validated feature.

### Challenge 8 — Edge Video Triage
- **Mentor:** Piotr Gryko. **Teams:** 1.
- **Task:** distil cloud-generated teacher labels into a compact edge model that **ranks frames by priority** and selects useful content within **1% and 5% upload budgets**. The student has five outputs: keep/discard, priority, modality, perspective, scene tags. Rank by priority score.
- **Data:** **947 frames from 282 clips**, **698 train / 138 val / 111 test**, plus 200 training-only calibration images. Labels come from clip-level teacher assessments. **No thermal training examples** in this release.
- **Starter:** **MobileNetV3-Small**, static ONNX export. Checkpoint selection and inference use the same priority score and tie-breaking rules.
- **Outputs:** **JSONL** per frame with frame ID, keep decision, and priority score.
- **Evaluation:** priority **rank correlation** and **capture within the 1% and 5% budgets**. The legacy `recall_at_*` metrics mean budget-normalised capture, not conventional recall. **The validation budgets contain only 1 and 7 frames**, so interpret results with care.
- **Edge:** Axelera Embedded 110m; validate deployment separately.

### Challenge 9 — Acoustic-to-Visual Drone Detection
- **Mentor:** Piotr Gryko. **Teams:** 3.
- **Task:** a lightweight acoustic model that, for each **5-second audio window**, decides whether to **trigger a visual check** and classifies the sound as **drone buzz, vehicle/mechanical, explosion, or outdoor ambience**.
- **Data:** **62 paired clips, 48 train / 8 val / 6 test.** Each has 16 kHz mono audio, **synchronised video**, and a precomputed spectrogram. **48 calibration spectrograms** from training data.
- **Starter:** **MobileNetV3-Small**, static ONNX export. The trainer stores its validation-selected wake threshold inside the model, and inference uses it automatically.
- **Outputs:** **JSONL** per clip with clip ID, wake decision, and sound class.
- **Evaluation:** **wake F1**, **classification accuracy**, **fraction of windows without an optical wake**.
- **Scope:** optical inactivity is a resource-use proxy, **not measured energy savings**. Visual confirmation is optional. **Bearing, camera slew, and event-onset prediction are not scored.** Small pilot dataset; test labels held out.
- **Edge:** Axelera Embedded 110m.

---

## 4. Which challenges can be combined

**Scoring:** every challenge is scored separately. A "merge" means separate deliverables plus one shared story and pitch. **Before committing to a merge, confirm with the organisers that one team can submit to more than one challenge [unconfirmed].**

| Combination | How they connect | Verdict [our view] |
|---|---|---|
| **6 + 9** | Ch 9 decides when to wake a visual check; Ch 6 is that check. Ch 9's synced video lets us demo the cascade. Same mentor, Axelera on site. | **Best merge (our plan).** |
| **8 + 9** | Microphone wakes the camera; triage decides which frames to upload. Both laptop-trainable. | Best fallback if there's no GPU. |
| **6 + 7** | One detection and tracking stack for air and ground, day and night. | Needs a GPU. |
| 6 + 8 + 9 | Full edge sensing chain. | Too much for 4 people in 1.5 days. |
| 3 + 4 | Same sponsor. Ch 4 plans the mission; Ch 3 handles the final approach. | We won't pursue it (Ch 3). |
| 4 + 6 / 4 + 9 | **No technical link.** Ch 4's only input is its scenario JSON, and its contacts are ground targets. | Pitch narrative only. |
| 1 + 4 | Voice reports as an operator interface. | Weak link, crowded track. |
| 2, 5 | Hardware builds. | Standalone. |

---

## 5. Open questions to answer Saturday morning
1. Do we have a GPU? (decides the Ch 6 approach)
2. Can one team submit to both Ch 6 and Ch 9? How are merged projects pitched? Pitch length? When is preselection announced? (ask Piotr Gryko / organisers)
3. Submission deadline and method for Ch 6 and Ch 9. (Ch 4's is known: Sunday 12:00.)
4. What Axelera equipment is in the Spectrum Room, and how is device time booked?
5. If we ever switch to Ch 4: does the link-loss part from the PDF still count? Where are the board address and token?

## 6. Rules that apply whatever we pick
1. Always keep a **valid** submission saved, and submit early.
2. Never tune on test data. Use calibration images only for INT8 calibration.
3. One shared eval script. Only measured numbers go on slides, each labelled with where it was measured (device vs. laptop CPU, FP32 vs. INT8).
4. Hold the checkpoints: **Sat 13:00** (valid baseline submissions), **Sat 19:00** (integration working), **Sun 12:00** (feature freeze + recorded demo video).
5. Use a recorded demo in the pitch, not a live one.
