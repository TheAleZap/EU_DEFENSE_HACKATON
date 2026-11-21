# Siesta: Listen, Look, Send (EDTH Paris 2026)

**Start here:** `00_START_HERE.md`. Then `docs/1_TEAM_MEMORY_full_shared_context.md` (full shared context), `docs/2_TEAM_BRIEF_plan_roles_checkpoints.md` (plan and roles), `docs/3_TEST_REPORT_results_and_coverage.md` (15/15 tests + coverage matrix).

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
```bash
# 1. Per-frame detections on all 8 sequences. Use FULL for scored files; MOTION ("peek before you stare", idea B)
#    for the efficiency story (tested: motion windows cover the 7 px drone in seq 4 only 19% of frames)
python ch6_detection_tracking/detect_video.py --kit $C6 --onnx .../inference_model.onnx --mode motion --out dets_motion.json
python ch6_detection_tracking/detect_video.py --kit $C6 --onnx .../inference_model.onnx --mode full   --out dets_full.json
#    prints model calls per frame (tested: full 13.3 avg / motion 6.9 avg over 8 sequences, about 48% fewer)
# 2. Track + scorer-aware output policy (idea C); --kit enables camera-motion compensation
python ch6_detection_tracking/track.py --dets dets_motion.json --out-dir pred_mot --kit $C6
python $C6/starter-kit/evaluate_submissions.py mot --gt-dir $C6/tracking_eval --pred-dir pred_mot --split valid
```
Tune `--high-thr --confirm-score --min-hits --min-len --max-gap --gate` with **leave-one-sequence-out** over seq_0001–0004:
```bash
# grid over track.py settings, scored by the kit MOT scorer; prints HELD-OUT per-sequence numbers (report these)
python ch6_detection_tracking/tune_tracker.py --dets dets_full.json --kit $C6        # --quick for a small grid, --no-cam to skip camera compensation
#    ends by printing the track.py command with the settings that are best on all 4 valid sequences (use it for test seq 5-8)
```
`ch6_detection_tracking/synth_test.py` is the mechanics test (synthetic detections from GT, **not a real score**): policy OFF gives MOTA about −0.85 / IDF1 0.37; policy ON gives MOTA +0.49 to +0.61 / IDF1 0.72 to 0.80.
`track.py` writes files for all sequences in the detections JSON, including test seq_0005–0008.

## Ch 9 (P3)
```bash
python ch9_acoustic_wake/train_cv.py --challenge-dir $C9 --device mps --out runs/c9_cv        # 7-fold CV on 56 clips + final model
#    prints OUT-OF-FOLD wake F1 / class acc / camera-on: report THESE (valid clips are inside CV)
python ch9_acoustic_wake/predict_consistent.py --challenge-dir $C9 --onnx runs/c9_cv/audio_wake_cnn.onnx --split test --out c9_test.jsonl
```
The fused rule makes `wake_triggered == (sound_class == "drone_buzz")` always. Compare the OOF numbers with the kit trainer's valid numbers.
**Warning:** without pretrained weights the model collapses to "always wake" (we saw this). Make sure the pretrained weights download.

## Ch 8 (P3)
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
- **Private repo**; all 4 teammates are collaborators. `main` must always run and pass `tests/run_all_tests.py`.
- **One branch per person:** `p1-detect`, `p2-track`, `p3-audio-triage`, `p4-metis-pitch`. Small commits; merge to `main` via pull request (or a fast merge after one teammate's look) at each checkpoint (13:00, 19:00, Sun 11:00).
- **Never commit** kits, datasets, images, videos, `runs/`, `exports/`, model weights (`*.pth`, `*.pt`, `*.onnx`), compiled Metis output, or `.env`/API keys (all in `.gitignore`). Share models via a shared drive or AirDrop/USB; write their path and checksum in `EXPERIMENTS_LOG.md`.
- **Do commit:** code, configs, `EXPERIMENTS_LOG.md` rows, `docs/` updates, and **small submission files** in `submissions/vN/` (COCO JSON, MOT txt, JSONL) after `check_submissions_before_upload.py` prints PASS. Tag them: `git tag v1-submitted`.
- **Kits location:** unpack each kit OUTSIDE the repo (e.g. `~/edth/kits/`) and pass paths with `--kit`, `--c6`, etc.
- **Claude Code:** `CLAUDE.md` makes Claude read `docs/1_TEAM_MEMORY_full_shared_context.md` first. Update `docs/2_TEAM_BRIEF_plan_roles_checkpoints.md` Section 0 when answers arrive.
