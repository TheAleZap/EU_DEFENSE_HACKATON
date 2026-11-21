# Siesta: Listen, Look, Send (EDTH Paris 2026)

**Start here:** `00_START_HERE.md`. Then `docs/1_TEAM_MEMORY_full_shared_context.md` (full shared context), `docs/2_TEAM_BRIEF_plan_roles_checkpoints.md` (plan and roles), `docs/3_TEST_REPORT_results_and_coverage.md` (15/15 tests + coverage matrix).

## Project summary (for mentors and organisers)

**Siesta: eyes closed, ears open.** A team of 4 on the Axelera track (Piotr Gryko), entering **Challenge 6** and **Challenge 9**.

### Our idea
On 3 February 2020, Madrid-Barajas airport closed for about two hours because pilots reported a drone. 26 flights were rerouted, and afterwards nobody could prove the drone had ever been there. The hard part wasn't stopping a drone. It was answering one question fast and with evidence: *is there really a drone, and where is it?*

Small drones are hard to answer that question about. In the challenge data a drone can be just 7 pixels wide in a full-HD frame. Finding something that small means running cameras and AI nonstop, which costs power, bandwidth and an operator's attention, and almost all of that watching finds nothing.

Our idea is to stop staring and start listening. **Siesta** is built from two of the Axelera challenges, joined into one sensor node:

- **Challenge 9 gives it ears.** A small audio model listens to the sky all the time. As long as it hears nothing like a drone, the camera stays asleep. That's the "siesta".
- **Challenge 6 gives it eyes.** When the audio model hears rotors, the camera wakes up. A detector finds the tiny aircraft and a tracker follows each one with a stable ID. The node only reports tracks it's confident about, because a wrong alarm is worse than staying quiet.

Both run on one small Axelera Metis chip at the edge, so the node can live on a mast or a rooftop and send only a few bytes per track. It doesn't stream video. The operator gets a clear answer: *drone confirmed, here's the track*, or *nothing there, reopen the runway*.

We didn't invent the "listen, then look" pairing. The Challenge 9 kit itself suggests handing audio detections to the Challenge 6 detector, and 9 of its training clips have a drone you can hear but can't see. What we add is making the two halves work as one system, tuning each part to how it's actually scored, and measuring what it costs on the real chip.

**Where it could go next:** confirmed tracks could feed a command-and-control map or a drone fleet planner (Challenge 4). A third step could send only the most important frames over a weak radio link (Challenge 8, which we parked to stay focused). The same node would fit airports, prisons and power plants. It detects and confirms; a human decides what happens next.

### How it works
A low-power sensor node for spotting small drones. A small audio model listens all the time while the camera stays off. When it hears rotors, it wakes the camera. An RF-DETR detector then finds aircraft as small as 7 pixels in full-HD video, and a tracker follows each one with a stable ID. Only short track messages leave the node, never a video stream. The goal is to run both models on a single Axelera Metis chip.

```
 mic, always on ──► [ audio wake model · Metis ] ── no drone ──► camera stays off
                                │
                           drone heard
                                ▼
 camera wakes ──► [ RF-DETR detector · Metis ] ──► [ tracker: confirmed tracks, one ID each ]
                                                                │
                                                                ▼
                                                 track message ──► operator / map
```

### What we're entering
| Challenge | What it asks | Our approach | Who |
|---|---|---|---|
| **6. Small aircraft detection and tracking** (priority 1) | Detections (COCO) and tracks (MOT) for tiny drones, airplanes and helicopters; efficiency on Metis | Properly trained RF-DETR Nano, plus a tracker built around the scorer: report only confirmed tracks and fill short gaps | Alejandro (detection), Salo (tracking) |
| **9. Acoustic-to-visual drone detection** (priority 2) | Per 5-second clip: wake the camera or not, plus a sound class; camera-off time | Audio augmentation, a wake threshold chosen by cross-validation, wake and class always consistent, then a demo that hands off to the Ch 6 detector | Viktoriia |
| **Metis, integration, demo** | Compile and benchmark on the chip; the cascade demo | Compile the audio model first, then the detector backbone with the decoder on the host | Dana |

We also looked at Challenge 8 (edge video triage) and dropped it to stay focused.

### Where we are (Sat 21 Nov, 16:15)
| Part | Status | Result (validation split, organisers' scorer) |
|---|---|---|
| Ch 6 detection | Submission file ready, passes our format checker | mAP@50 **0.595** against the kit reference 0.447; small-object AP@50 **0.734** against 0.506 (RF-DETR Nano, FP32, MacBook M4 Max) |
| Ch 6 tracking | **Our tracker now beats the baseline** (v1 submission, checker passes) | MOTA **+0.247** against the kit reference −0.013; IDF1 **0.448** against 0.248; 3 ID switches. Default settings, not tuned on validation |
| Ch 6 detection, bigger model | RF-DETR Small (512 px input) trained and scored. **Rejected: Nano stays** | Detection is mixed: mAP@50:95 0.416 against Nano's 0.392, but small-object AP@50 0.702 against 0.734 and mAP@50 0.565 against 0.595. Tracking is clearly worse: MOTA +0.167 against +0.247, 14 ID switches against 3. Also 2× slower (53 against 26 ms per image, FP32, CPU) |
| Ch 9, big model | **AudioSet-pretrained audio model** (v2 candidate, checker passes) | Leave-one-clip-out over 56 clips: class accuracy **0.73** (kit-style model 0.30), wake F1 **0.53** (0.50), camera on 23% of the time. Wake and class never contradict |
| Ch 9, on-chip model | **New: the kit's small CNN, trained with the extra data and taught by the AudioSet model** (v3 candidate, checker passes) | Leave-one-clip-out over 56 clips, 3 training seeds: wake F1 **0.59** on average (0.47–0.67), class accuracy **0.59** (0.55–0.61), camera on 39%. The same CNN without the extra data: 0.48, 0.34, camera on 96%. Small enough for Metis (MobileNetV3-Small) |
| Metis | **Blocked: no hardware access yet** | None |
| Code | Full pipeline for every part; 15/15 tests pass on the real kit data | n/a |

Tried and rejected: training and inferring on native-resolution tiles to help with tiny drones. It gave no gain on validation, so we kept the simpler model. Running the tracker on tiled detections was also why our first tracker scored worse than the kit's (MOTA −0.32). Switching to one whole-frame pass per frame fixed it, and it's 25 times cheaper.

Still hard: two validation sequences have drones only 7–10 pixels wide. Whole-frame detection can't see them. Tiled detection at native resolution does find the 7 px drone in 82% of frames, but with about as many confident false boxes on fixed scenery. Adding only tiny tile tracks raises validation MOTA to +0.31, but that depends on a size cut-off tuned to one sequence, so we don't ship it. For now the tracker reports nothing on those two sequences rather than false alarms.

Ch 9 finding: the AudioSet model often hears the "drone" clips as explosions, thunder or music. The drone is usually buried under louder sounds, which is why every model finds wake detection hard. The AudioSet model is an 86M-parameter transformer (114 ms per 5 s clip on the laptop CPU), much bigger than the kit's 3.6 MB CNN, and we haven't tried compiling it for Metis.

### New Ch 9 data, and how the plan changes
**What we got (Sat 15:30):** `audio_export`, 90 real-world clips, mostly social-media and combat footage: drones such as the Shahed passing overhead, FPV strikes, gunfire, artillery, vehicles, helicopters, speech. Each clip has a 16 kHz mono `audio.wav` and a `video.mp4`. 89 also have an AI-written `analysis.json` with a clip-level `has_drone_audio` flag: **38 clips with drone audio (23 min), 51 without (36 min)**. Clips run 5 s to 3.5 min (median 25 s). The kit's own docs say Ch 9 "needs more independent recordings", and this is exactly that.

**What it is good for, and its limits:**
- **Quantity:** about 700 five-second windows, against the kit's 56 labelled clips. That's far more training material.
- **Hard negatives:** the 51 no-drone clips are full of the sounds our models confuse with drones: engines, explosions, gunfire, rotors.
- **Labels are coarse and machine-written.** They say *whether* a clip contains drone audio, not *when*, and not the kit's 4 sound classes. A 5-second window from a "drone" clip may be just speech. So: **no-drone clips give clean negatives; drone clips give noisy positives.**
- **Different recordings.** Phone audio with music and speech, while the kit's audio is "clean field microphone". The extra data could teach the wrong things, so we have to prove it helps on the kit's data.

**How the plan changes:**
1. **The extra data is used for training only, never for scoring.** Every Ch 9 number we report stays leave-one-clip-out on the kit's 56 labelled clips, with the extra clips added to the training side of each split. We keep the extra data only if it improves that number.
2. **Wake detection gets the extra data; the 4-class sound label stays on kit data.** The extra labels only cover drone or not, so we train a drone/no-drone model on kit plus extra windows, and keep the 4-class model on kit clips. Combined rule: wake if the drone model says so, sound class = `drone_buzz` if wake, otherwise the best non-drone class. Wake and class still never contradict.
3. **Two routes, run in parallel:**
   - **AudioSet route (Alejandro's Mac, now):** add the extra windows to `audioset_probe.py`. Quick, and likely the best accuracy.
   - **On-chip route (Viktoriia):** turn the extra windows into kit-style spectrogram PNGs and retrain the kit's small CNN. This is the model that compiles for Metis, so it's the one that turns accuracy into the edge story. It needs the kit's spectrogram settings reverse-engineered first, since they aren't documented.
4. **Better demo material:** real drone footage with synchronised audio, ideal for showing "hear it, then look" in the pitch video.
5. **New rules question:** are we allowed to train on this extra data? The kit data alone stays our fallback (v1 Ch 9 file).

The data is about 1 GB, so it stays out of git, like the kits. Keep it at `~/edth/extra/audio_export` (or pass its path).

**What we found (16:15, all leave-one-clip-out on the kit's 56 clips):**
- **On-chip route: the extra data is a big win.** We skipped reverse-engineering the kit's PNGs. Instead `ch9_acoustic_wake/distill_cnn.py` makes its own log-mel spectrograms (16 kHz, 128 mel bands, 224×224) for both the kit clips and up to 6 windows per extra clip. It trains the kit's MobileNetV3-Small on them. The AudioSet model acts as a teacher, giving soft 4-class labels for every window, so the coarse drone/no-drone labels stop being a problem. Results: class accuracy jumps from 0.34 to 0.59, wake F1 from 0.48 to 0.59, and camera-on time falls from 96% to 39%. Training for 20 epochs works best (12 epochs: F1 0.58; 30 epochs: 0.44). Single runs vary by ±0.1, so treat 0.59 as the honest number, not the best seed's 0.67.
- **AudioSet route: the extra data doesn't help.** The separate drone/no-drone model from plan step 2 above made wake F1 worse (0.46). Adding the extra windows to the 4-class model at low weight gave F1 0.55 and accuracy 0.75, within noise of the kit-only 0.53 and 0.73. v2 stays kit-only.
- **So the two candidates are:** v2, the AudioSet model (better class accuracy, 86M parameters, Metis untested), and v3, the kit-sized CNN (better wake F1 and camera-on time, built for the chip). v3 has the better edge story. Both need the organisers to allow the extra data and/or the AudioSet model (v3 uses AudioSet as a teacher during training only). If either is refused, v1 (the kit baseline) is the fallback.

### Plan to Sunday
| By | Goal |
|---|---|
| Sat 19:00 | ~~Our tracker beats the kit's MOTA on validation~~ (done 15:00). ~~Ch 9 beats its baseline~~ (done 15:30, pending the rules question below). ~~Ch 9 extra data tested on both routes~~ (done 16:10: helps the on-chip CNN). ~~RF-DETR Small scored~~ (done 16:05: rejected). At least one model compiled on Metis. Cascade demo working on one real drone clip. |
| Sun 11:00 | Final test predictions checked and submitted. |
| Sun 12:00 | Feature freeze, demo video recorded, pitch rehearsals. |

### What we need from you
**Hardware (most urgent):**
1. Access to an Axelera Metis / Embedded 110m: when, where (Spectrum Room?), and how do we book time?
2. The host machine: which OS, CPU and Voyager SDK version? Can we SSH in and run our own Python? We need to time the host-side decoder and tracker on it.
3. Can the compiler (`axcompile`) run without a card, for example on a Linux laptop or WSL2? Then we could compile before we get device time.
4. Is there any way to measure power on the board?
5. Has anyone compiled RF-DETR (it uses deformable attention) for Metis? Is there a known recipe? Our plan is the backbone on the chip and the decoder on the host.

**Rules and logistics:**

6. Can one team submit to Ch 6 and Ch 9 and pitch them as one project?
7. What are the deadlines, and how do we upload test predictions?
8. Are the "proposed" judging weights in the kits final? What's the protocol for hardware measurements?
9. How long is the pitch, and when is preselection?
10. For Ch 9, may we use an audio model pretrained on a public dataset (AudioSet), the way the kit uses an ImageNet-pretrained CNN? And does a bigger model on Metis count against us if it still fits in 1 GB?
11. May we train Ch 9 on extra real-world recordings beyond the kit (we evaluate only on kit data)? And may a big pretrained model act as a teacher during training, if the submitted model is the kit's own small CNN?

**Feedback we'd value:**

12. Is "listen first, then look" the right story for the judges, or should we pitch Ch 6 on its own?
13. Any advice on tracking drones only 7 pixels wide, when the detector also fires on fixed scenery?

## Team and current plan (updated Sat 21 Nov, 16:15)

| Role | Who | Branch | Work |
|---|---|---|---|
| **P1** | **Alejandro** (M4 Max MacBook) | `p1-detect` | Challenge 6 detection |
| **P2** | **Salo** | `p2-track` | Challenge 6 tracking |
| **P3** | **Viktoriia** | `p3-audio-triage` | Challenge 9 only |
| **P4** | **Dana** (Windows laptop) | `p4-metis-pitch` | Axelera prep, submission checks, pitch |

**Scope:** Challenge 6 is priority 1, Challenge 9 is priority 2. **Challenge 8 is dropped** as a submission; its code stays in the repo as pitch material for the "send" step.

### Status
| Challenge | Status | Still to do |
|---|---|---|
| 6, detection | v0 done: valid mAP@50 0.595, small_AP50 0.734 (20-epoch RF-DETR Nano, kit scorer). Tiled model: no gain. **RF-DETR Small (512 px): scored and rejected** (small_AP50 0.702, tracking MOTA +0.167, 2× slower). Nano stays | INT8 check of the Nano backbone against FP32 on valid, for the efficiency score |
| 6, tracking | **v1 done** (`submissions/v1/`, checker PASS): valid MOTA **+0.247**, IDF1 **0.448**, 3 ID switches (kit scorer). Whole-frame detections + `track.py` defaults | Tuned settings reach +0.283 in-sample, but leave-one-out says the tuning doesn't generalise, so defaults ship. Tiny-drone hybrid tested (+0.314) but too fragile to ship |
| 9, acoustic wake | v1 = kit baseline. **v2 candidate**, AudioSet model + logistic regression (`audioset_probe.py`): wake F1 0.533, class acc 0.732, camera on 0.23. **v3 candidate**, kit CNN trained on kit + extra data with the AudioSet model as teacher (`distill_cnn.py`): wake F1 0.59, class acc 0.59, camera on 0.39 (average of 3 seeds). All leave-one-clip-out on 56 clips; both checker PASS | Organisers must confirm extra data and pretrained teacher are allowed. Average several seeds to steady v3. Compile v3 on Metis |
| Axelera chip | **Hardware not available yet** | Prepare so compile can run the moment it arrives |
| Pitch and demo | Mentor summary written (see "Project summary" above) | Slides, story, demo video |

### What each person does now
- **Alejandro (P1), on the Mac:** (1) INT8 check: score the exported Nano backbone in INT8 against FP32 on valid, so the efficiency claim has numbers. (2) Help Viktoriia steady the v3 CNN: train 5 seeds and average their outputs (the ensemble's own leave-one-clip-out number is the one to report). (3) Run the Ch 6 detector on the `audio_export` drone videos for Salo's demo.
- **Salo (P2):** pull `main` first, and use `detect_video.py --mode whole` from now on (RF-DETR Small was tested and tracks worse; stay on Nano). Tasks: (1) **demo clip**: pick 2–3 `audio_export` videos where a drone is visible (the Shahed clips, e.g. `787248`), run the v1 detector and tracker on them, and render the boxes and track IDs into a video; ask Alejandro for the data and model by AirDrop. (2) Per-sequence error analysis of v1 for the pitch: where the points are lost. (3) `track.py --online` numbers for an honest real-time demo.
- **Viktoriia (P3):** own the **on-chip Ch 9 route, now v3**. The spectrogram reverse-engineering is no longer needed: `ch9_acoustic_wake/distill_cnn.py` makes its own spectrograms and already trains the kit CNN on the extra data (wake F1 0.59, class acc 0.59; see "What we found"). Pull `main` and: (1) rerun it on your machine to confirm the numbers (command in the Ch 9 section below; needs the extra data and two logit caches from Alejandro by AirDrop). (2) Try to reduce the seed noise: average 3–5 seeds, or vary windows per clip (`--extra-max-windows`) and extra weight; keep only changes that improve the leave-one-clip-out numbers over several seeds. (3) Export calibration spectrograms for Dana: the chip must see the **same** spectrograms as `distill_cnn.py` makes, not the kit PNGs.
- **Dana (P4):** (1) ask the organisers the rules questions 10–11 above: AudioSet-pretrained model, extra recordings, and a big model as teacher only. (2) Metis: compile the **v3 CNN** (`audio_wake_cnn.onnx`, AirDrop from Alejandro) first as soon as hardware appears, with Viktoriia's calibration spectrograms. (3) Pitch: build the "hear it, then look" storyboard around one real-drone `audio_export` clip with Salo. The story now has numbers: the camera is on only 39% of the time with the on-chip model, against 96% for the same model without our data. (4) Keep running `check_submissions_before_upload.py` on each new submission folder.

### Hardware hold
Nobody waits for the Axelera hardware. Training, scoring and submission files all run on laptops. Only Metis compile and on-chip speed measurements are blocked. Cut line 14:00: if the chip isn't working, report CPU ONNX Runtime numbers, clearly labelled.

### Getting started (Salo, Viktoriia, Dana)
```bash
git clone https://github.com/TheAleZap/EU_DEFENSE_HACKATON.git   # re-clone if you cloned before 12:45
cd EU_DEFENSE_HACKATON
git checkout -b p2-track            # or p3-audio-triage / p4-metis-pitch
git push -u origin p2-track
```
Unpack the kits outside the repo (e.g. `~/edth/kits/`), run `uv venv && uv pip install -e .` in each needed kit's `starter-kit/`, then open Claude in the repo and say "I am P2" (or P3 / P4).

### Using branches
```bash
# while working: commit small and often, only your own files
git add <your files>
git commit -m "P2: tracker tuned on seq 1-3, MOTA 0.31 on seq 4"
git push

# get teammates' latest work into your branch
git fetch origin
git merge origin/main
```
At each checkpoint (19:00, Sun 11:00) open a pull request from your branch into `main`; one teammate takes a quick look and merges it. `main` must always have working code and submission files that pass `check_submissions_before_upload.py`.

## Code kit. Built and tested Sat 21 Nov

Every script was run end to end in a test environment against the real kit data and **scored with the kits' own evaluators**. That environment had no internet access for pretrained weights and only 2 CPUs, so the **models trained there are only plumbing tests; their scores mean nothing.** Real numbers come from your machines.

Paths below: `C6=<...>/challenge-1-cuas-tracking`, `C8=<...>/challenge-4-edge-triage`, `C9=<...>/challenge-5-acoustic-visual-cuas`.
Extra dependency beyond the kits: `pip install opencv-python` (for motion and video reading).

## Ch 6, detection (P1)
```bash
# 1. Build the native-resolution tiled training set (idea A). ~1 min.
python ch6_detection_tracking/build_tiled_dataset.py --src $C6/dataset --out tiled_dataset
#    tested: train 859 -> 3266 images (2544 boxes), valid 422 -> 1590 (1182 boxes); test copied
# 2. Train with the KIT trainer on it (Mac: RF-DETR auto-uses MPS)
cd $C6/starter-kit && python train_rfdetr.py --dataset-dir /path/to/tiled_dataset --model nano --epochs 20 --output-dir runs/c6_tiled
python export_rfdetr_axelera.py --model nano --num-classes 3 --weights runs/c6_tiled/checkpoint_best_total.pth --calibration-dir ../calibration
# 3. Tiled COCO inference (every native 384 tile + one whole-frame pass, class-wise NMS, top 100)
python ch6_detection_tracking/tiled_detect.py --kit $C6 --onnx exports/rfdetr/full/inference_model.onnx --split valid --out preds_valid.json
python $C6/starter-kit/evaluate_submissions.py detect --gt $C6/dataset/valid/_annotations.coco.json --preds preds_valid.json
python ch6_detection_tracking/tiled_detect.py --kit $C6 --onnx .../inference_model.onnx --split test --out preds_test.json
```
**Compare against the kit's untiled prediction of the same model.** Keep tiling only if mAP / small_AP improve on valid.

## Ch 6, tracking (P2)
**v1 recipe (valid MOTA +0.247, IDF1 0.448, kit scorer; 33 s for all 8 sequences on the M4 Max):**
```bash
python ch6_detection_tracking/detect_video.py --kit $C6 --onnx $C6/starter-kit/exports/c6_base20/full/inference_model.onnx --mode whole --out dets_whole.json
python ch6_detection_tracking/track.py --dets dets_whole.json --out-dir pred_mot --kit $C6        # default settings
python $C6/starter-kit/evaluate_submissions.py mot --gt-dir $C6/tracking_eval --pred-dir pred_mot --split valid
```
Use `--mode whole` (one whole-frame pass, like the kit's baseline) for scored files. With the 20-epoch model, tiled `--mode full` detections score far worse (MOTA −0.32 after tuning) and cost 25 calls per frame. `--mode motion` is only for the efficiency story (motion windows cover the 7 px drone in seq 4 only 19% of frames). `track.py --merge sum|max` merges the same box reported under several classes; it helps when settings are tuned, but not at the defaults.
Tune `--high-thr --confirm-score --min-hits --min-len --max-gap --gate` with **leave-one-sequence-out** over seq_0001–0004:
```bash
# grid over track.py settings, scored by the kit MOT scorer; prints HELD-OUT per-sequence numbers (report these)
python ch6_detection_tracking/tune_tracker.py --dets dets_whole.json --kit $C6        # --quick for a small grid, --no-cam to skip camera compensation
#    ends by printing the track.py command with the settings that are best on all 4 valid sequences (use it for test seq 5-8)
```
`ch6_detection_tracking/synth_test.py` is the mechanics test (synthetic detections from GT, **not a real score**): policy OFF gives MOTA about −0.85 / IDF1 0.37; policy ON gives MOTA +0.49 to +0.61 / IDF1 0.72 to 0.80.
`track.py` writes files for all sequences in the detections JSON, including test seq_0005–0008.

## Ch 9 (P3)
**v2 candidate (AudioSet model; leave-one-clip-out wake F1 0.533, class acc 0.732 on 56 clips):**
```bash
pip install scikit-learn transformers          # into the Ch 9 kit venv; downloads the 350 MB AST model once
python ch9_acoustic_wake/audioset_probe.py --challenge-dir $C9 --out c9_test.jsonl
#    --extra ~/edth/extra/audio_export adds the extra clips to training (no gain on this route)
```
**v3 candidate (on-chip kit CNN, trained on kit + extra data with the AudioSet model as teacher; leave-one-clip-out wake F1 0.59, class acc 0.59, average of 3 seeds; ~3 min on the M4 Max):**
```bash
# once: AudioSet logits for kit clips and extra windows (teacher labels), written as JSON caches
python ch9_acoustic_wake/audioset_probe.py --challenge-dir $C9 --out /tmp/ignore.jsonl --extra ~/edth/extra/audio_export \
  --cache runs/c9_audioset_logits.json --extra-cache runs/c9_audioset_extra_logits_w6.json
python ch9_acoustic_wake/distill_cnn.py --challenge-dir $C9 --extra ~/edth/extra/audio_export \
  --kit-logits runs/c9_audioset_logits.json --extra-logits runs/c9_audioset_extra_logits_w6.json \
  --epochs 20 --seed 42 --device mps --out runs/c9_distill_e20
#    writes audio_wake_cnn.onnx, c9_test.jsonl and cv_report.json (out-of-fold numbers: report THESE). --no-extra for the kit-only comparison
```
Earlier CNN route:
```bash
python ch9_acoustic_wake/train_cv.py --challenge-dir $C9 --device mps --out runs/c9_cv        # 7-fold CV on 56 clips + final model
#    prints OUT-OF-FOLD wake F1 / class acc / camera-on: report THESE (valid clips are inside CV)
python ch9_acoustic_wake/predict_consistent.py --challenge-dir $C9 --onnx runs/c9_cv/audio_wake_cnn.onnx --split test --out c9_test.jsonl
```
The fused rule makes `wake_triggered == (sound_class == "drone_buzz")` always. Compare the OOF numbers with the kit trainer's valid numbers.
**Warning:** without pretrained weights the model collapses to "always wake" (we saw this). Make sure the pretrained weights download.

## Ch 8 (dropped as a submission; reference only)
```bash
python ch8_edge_triage/train_rank.py --kit $C8 --device mps --out runs/c8_rank               # ranking loss + augmentation
cd $C8/starter-kit && python predict_onnx_baseline.py triage --onnx runs/c8_rank/triage_distill.onnx --dataset-dir ../dataset --split valid --out raw_valid.jsonl
python ch8_edge_triage/pool_clips.py --kit $C8 --split valid --preds raw_valid.jsonl --out pooled_valid.jsonl   # rank clips, not frames
python $C8/starter-kit/evaluate_submissions.py triage --gt $C8/dataset/valid/labels.jsonl --preds pooled_valid.jsonl
```
Synthetic check (noisy frame scores, kit scorer, 20 seeds): pooling took capture@1% from 0.35 to 1.00, capture@5% from 0.81 to 0.98, and Spearman from 0.58 to 0.70. The real gain depends on the model; always compare pooled vs unpooled on valid.

## Metis (P4, on the Axelera Linux host or WSL2)
`metis_axelera_compile/preprocess_cnn.py` (Ch 8/9, no normalisation) and `metis_axelera_compile/preprocess_rfdetr.py` (Ch 6, ImageNet normalisation) were verified to give **identical tensors** to the kits' inference preprocessing.
```bash
axcompile -i audio_wake_cnn.onnx --imageset $C9/calibration_spectrograms --transform metis_axelera_compile/preprocess_cnn.py --input-shape 1,3,224,224 -o out_c9/
axcompile -i triage_distill.onnx --imageset $C8/calibration --transform metis_axelera_compile/preprocess_cnn.py --input-shape 1,3,224,224 -o out_c8/
axcompile -i exports/rfdetr/backbone/inference_model.onnx --imageset $C6/calibration --transform metis_axelera_compile/preprocess_rfdetr.py --input-shape 1,3,384,384 -o out_c6_backbone/
```
If the compiler reports a rank mismatch, use `--transform <file>:get_preprocess_transform_batched`. Check `axcompile --help`.
Output: `compiled_model/` (manifest.json, model.json, ...) and `compilation_report.json` (has a `status`).

## Before EVERY upload
```bash
python check_submissions_before_upload.py --c6-kit $C6 --c6-preds preds_test.json --c6-mot pred_mot \
  --c8-kit $C8 --c8-preds c8_test.jsonl --c9-kit $C9 --c9-preds c9_test.jsonl
```
It must print **PASS**. It checks ids, 0/1-based classes, frames, duplicates, value ranges, missing sequence files, and Ch 9 consistency.

## Run the test suite (rerun with REAL models as soon as you have them)
```bash
python tests/run_all_tests.py --c6 $C6 --c8 $C8 --c9 $C9 --work /tmp/siesta_tests \
  --c9-onnx runs/c9_cv/audio_wake_cnn.onnx [--c6-onnx .../inference_model.onnx] [--c8-onnx runs/c8_rank/triage_distill.onnx]
```

## Team workflow (GitHub)
- Repo `TheAleZap/EU_DEFENSE_HACKATON`; all 4 teammates are collaborators. `main` must always run and pass `tests/run_all_tests.py`.
- **One branch per person:** `p1-detect` (Alejandro), `p2-track` (Salo), `p3-audio-triage` (Viktoriia), `p4-metis-pitch` (Dana). Small commits; merge to `main` via pull request (or a fast merge after one teammate's look) at each checkpoint (13:00, 19:00, Sun 11:00).
- **Never commit** kits, datasets, images, videos, `runs/`, `exports/`, model weights (`*.pth`, `*.pt`, `*.onnx`), compiled Metis output, or `.env`/API keys (all in `.gitignore`). Share models via a shared drive or AirDrop/USB; write their path and checksum in `EXPERIMENTS_LOG.md`.
- **Do commit:** code, configs, `EXPERIMENTS_LOG.md` rows, `docs/` updates, and **small submission files** in `submissions/vN/` (COCO JSON, MOT txt, JSONL) after `check_submissions_before_upload.py` prints PASS. Tag them: `git tag v1-submitted`.
- **Kits location:** unpack each kit OUTSIDE the repo (e.g. `~/edth/kits/`) and pass paths with `--kit`, `--c6`, etc.
- **Claude Code:** `CLAUDE.md` makes Claude read `docs/1_TEAM_MEMORY_full_shared_context.md` first. Update `docs/2_TEAM_BRIEF_plan_roles_checkpoints.md` Section 0 when answers arrive.
