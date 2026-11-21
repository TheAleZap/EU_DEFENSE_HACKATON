# Siesta test report and requirements coverage (Sat 21 Nov, 12:05)

**Result: 15 / 15 tests passed.** Run with `python tests/run_all_tests.py ...` (full log in `TEST_REPORT.json`).

All tests ran against the **real kit data** and used the **kits' own scorers**. The environment had no internet for pretrained weights and 2 CPUs, so the detector used in the Ch 6 tests is a **dummy ONNX with the exact RF-DETR inputs and outputs**, and the Ch 8/9 models are **1-epoch, non-pretrained plumbing models**. Those tests prove **mechanics and formats, not accuracy**. Rerun the suite with real models via `--c6-onnx/--c8-onnx/--c9-onnx`.

## Test results

| ID | What it checks | Result | Key numbers |
|---|---|---|---|
| C6-1 | Tiled dataset: structure, every original image and box kept, boxes inside images, native scale, negatives empty, test copied unchanged | PASS | train 859 → 3266 images, 842 → 2544 boxes; valid 422 → 1590, 387 → 1182; median box area unchanged (1485 / 308 px²), so **no downscaling** |
| C6-2 | Tile grid covers every pixel of all 14 image/video sizes; tiles ≤ 384 | PASS | max 24 tiles per image |
| C6-3 | Class-wise NMS: keeps different classes, removes duplicates, caps at top 100 | PASS | |
| C6-4 | `tiled_detect.py` on **all** 422 valid + 328 test images; kit COCO scorer accepts | PASS | 9.4 model calls per image on average |
| C6-5 | **Motion windows vs real target position** (real videos, no detector) | PASS (informative) | Target inside a motion window: seq 1: 65%, seq 2: 97%, seq 3: 95% (12 windows), **seq 4 (7 px drone): 19%** (12 windows) / 1% (6). **Decision: default raised to 12; use FULL mode for scored submissions; motion mode for the efficiency story** (carry-over windows around previous detections add coverage once a target is found). |
| C6-6 | `detect_video.py` full + motion on **all 8** sequences → `track.py` (offline + camera compensation, and online) → kit MOT scorer | PASS | detector calls per frame: full 13.3 avg (25 on 1920×1080), motion 6.9 avg (about 10 on 1920×1080), so **about 48% fewer calls** |
| C6-7 | Tracker output policy on synthetic detections (3 seeds), kit MOT scorer (**synthetic, not a real score**) | PASS | MOTA −0.81/−0.92/−0.86 → **+0.61/+0.58/+0.49**; IDF1 0.41/0.35/0.34 → **0.80/0.75/0.72**; ID switches 10–11 → 0–3 |
| C8-1 | `pool_clips.py`: ids and order preserved, scores in [0,1], clip frames ranked together; kit scorer; synthetic noisy predictions (10 seeds) | PASS | capture@1% 0.40 → **0.90**, capture@5% 0.80 → **0.99**, Spearman 0.55 → **0.69** |
| C8-2 | Test-split prediction with a trained ONNX → pooling → test file | PASS | |
| C9-1 | `predict_consistent.py`: wake == (class == drone_buzz) on every clip; kit scorer accepts | PASS | |
| C9-2 | The label identity behind the rule holds on all 56 labelled clips | PASS | 56 / 56 |
| M-1 | Metis `preprocess_*.py` vs kit inference preprocessing on **all 448 calibration images** | PASS | max difference **0.0** |
| CHK-1 | `check_submissions.py` passes the complete test set (Ch 6 COCO + seq 5–8 MOT, Ch 8, Ch 9) | PASS | |
| CHK-2 | Checker catches 6 injected mistakes | PASS | missing MOT file, 0-based MOT class, duplicate (frame,id), COCO class 3, missing Ch 8 id, invalid Ch 9 class: **all caught** |
| RF-1 | RF-DETR's **own training loader** (`build_roboflow_from_coco`) loads the tiled dataset | PASS | tiled train 3266 / val 1590; same label ids {0,1,2} as the kit's original dataset |

## Requirements coverage (what each challenge asks → where it's covered → what's still open)

### Ch 6: detection + tracking (judging proposal: detection 40%, tracking 40%, edge 20%)
| Requirement | Covered by | Status |
|---|---|---|
| Build on RF-DETR (Nano 384) | kit `train_rfdetr.py` on `tiled_dataset` (RF-1 verified loader) | **To do on the Mac:** actual training |
| Improve small-object detection (small_AP@50) | idea A: native tiles in training (C6-1) and inference (C6-2, C6-4) | Mechanics verified; **gain must be measured on valid vs kit baseline** |
| COCO JSON detections, valid + test | `tiled_detect.py` (C6-4), checker (CHK) | Verified format |
| Stable tracks through motion, misses, clutter (MOTA, IDF1, continuity) | `track.py`: camera compensation, centre-distance gating, two-stage association, confirmation, gap fill, scorer-aware output (C6-6, C6-7) | Mechanics verified (synthetic); **tune on real detections with leave-one-sequence-out** |
| MOT files for all 8 sequences | `detect_video.py` + `track.py` (C6-6), checker | Verified format |
| Edge: Metis compile, INT8, end-to-end throughput, < 1 GB | `metis/preprocess_rfdetr.py` (M-1), backbone compile command | **Open:** needs Voyager SDK + device; host-tail timing must run on the Axelera host |
| Efficiency story | motion mode: about 48% fewer detector calls (C6-6) | Verified mechanics; coverage caveat in C6-5 |

### Ch 8: edge triage (proposal: capture 55%, Spearman 25%, Metis 20% + bonus for clip dedup and 100% on-chip)
| Requirement | Covered by | Status |
|---|---|---|
| Distil teacher labels into a compact model | `c8/train_rank.py` (kit model + ranking loss + augmentation) | Smoke-tested; **real training on the team machine** |
| Capture within 1% / 5% budgets, Spearman | ranking loss + `pool_clips.py` (C8-1) | Synthetic gain verified; **measure on valid** |
| Temporal clip dedup (bonus) | `pool_clips.py` ranks clips, not frames | Implemented and tested |
| JSONL output, exact ids | kit predictor + pooling (C8-2), checker | Verified |
| Metis INT8, throughput, power, footprint | `metis/preprocess_cnn.py` (M-1) | **Open:** compile and measure; power only if a meter or tool exists |

### Ch 9: acoustic wake (proposal: wake F1 40%, class accuracy 25%, camera-off proxy 20%, Metis 15%)
| Requirement | Covered by | Status |
|---|---|---|
| Wake decision + 4-class sound label per 5 s clip | `c9/train_cv.py` + `predict_consistent.py` | Smoke-tested; **real training with pretrained weights** |
| Robust threshold despite 8 valid clips | 7-fold CV on 56 clips, threshold from out-of-fold predictions | Implemented |
| Wake/class never contradict | fused rule (C9-1, C9-2) | Verified |
| Camera-off proxy | reported by kit scorer; utility F1 − 0.15 × camera-on, same as kit | Implemented |
| JSONL output, exact ids | `predict_consistent.py`, checker | Verified |
| Metis wake latency and footprint | `metis/preprocess_cnn.py` (M-1) | **Open:** compile and measure |
| Optional cascade to visual check | demo layer (P4) on Ch 9 paired videos | **Open:** build the demo video |

## Known limits and risks (be honest in the pitch)
1. No real accuracy numbers exist yet; everything above is mechanics.
2. Motion windows miss the 7 px drone in seq 4 most of the time, so **don't rely on motion-only detection for scored files.**
3. Metis compilation of RF-DETR is unverified (deformable attention). Cut line Sat 14:00, as in the brief.
4. The Ch 8/9 validation and test sets are tiny (Ch 8 test budgets k = 1 and 6; Ch 9 has 6 test clips), so scores are noisy.
5. Without pretrained weights, the Ch 9 model collapses to "always wake" (observed). **Confirm the weights download.**
