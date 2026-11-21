# Experiment log (one row per run; numbers only from the kit evaluators)

| time | owner | challenge | change | config / commit | metric (eval script, split) | model path + checksum | notes |
|---|---|---|---|---|---|---|---|
| Sat 12:05 | Claude | all | code kit test suite | tests/run_all_tests.py | 15/15 PASS (mechanics only) | n/a (dummy / smoke models) | see docs/3_TEST_REPORT_results_and_coverage.md |
| Sat 12:34 | P1 | 6 | kit baseline, 1 epoch (timing run) | RF-DETR Nano, kit defaults (bs 8, accum 2), untiled, M4 Max MPS | mAP50 0.4554, mAP50_95 0.2801, small_AP50 0.5764 (evaluate_submissions.py detect, valid); FP32 ONNX, CPU ORT 26 ms/img median on M4 Max | P1 Mac `~/edth/kits/challenge-1-cuas-tracking/starter-kit/exports/c6_1ep/full/inference_model.onnx`, sha256 c729834532fdbd50… | 1 epoch = 114 s wall incl. val; for P2 plumbing only |
