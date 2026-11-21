"""Siesta full test suite. Verifies the code kit against the REAL kit data and the kits' OWN scorers.

What it proves:  the code runs, produces valid files for every required output, and the
                 mechanisms (tiling, motion windows, tracker policy, clip pooling, consistency,
                 Metis preprocessing, submission checks) do what the plan claims.
What it can't prove: real accuracy. That needs models trained with pretrained weights on the team's
                 machines, and Metis compilation, which needs the Voyager SDK and hardware.

Usage:
  python tests/run_all_tests.py --c6 <challenge-1-cuas-tracking> --c8 <challenge-4-edge-triage> \
      --c9 <challenge-5-acoustic-visual-cuas> --work /tmp/siesta_tests [--c8-onnx X --c9-onnx Y]
Writes <work>/TEST_REPORT.json and prints PASS/FAIL per test.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "c6"))
PY = sys.executable
RESULTS = []


def test(name):
    def deco(fn):
        def run(*a, **k):
            t0 = time.time()
            try:
                info = fn(*a, **k) or {}
                RESULTS.append({"test": name, "status": "PASS", "seconds": round(time.time() - t0, 1), **info})
                print(f"PASS  {name}  {json.dumps(info)[:300]}")
            except Exception as e:  # noqa: BLE001
                RESULTS.append({"test": name, "status": "FAIL", "error": f"{type(e).__name__}: {e}",
                                "trace": traceback.format_exc()[-1500:]})
                print(f"FAIL  {name}  {type(e).__name__}: {e}")
        return run
    return deco


def sh(args, **kw):
    r = subprocess.run([str(a) for a in args], capture_output=True, text=True, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(map(str, args))}\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
    return r.stdout


def last_json(out: str):
    return json.loads(out[out.index("{"):out.rindex("}") + 1])


def make_dummy_rfdetr(path: Path):
    """ONNX with the kit RF-DETR I/O (input [1,3,384,384] -> dets/labels [1,300,4]), outputs depend on input."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    rng = np.random.default_rng(0)
    dets = np.concatenate([rng.uniform(0.1, 0.9, (300, 2)), rng.uniform(0.02, 0.2, (300, 2))], 1).astype(np.float32)[None]
    labels = rng.normal(-4, 2, (1, 300, 4)).astype(np.float32)
    nodes = [helper.make_node("ReduceMean", ["input"], ["m"], keepdims=1),
             helper.make_node("Reshape", ["m", "shp"], ["m3"]),
             helper.make_node("Mul", ["m3", "k"], ["z"]),
             helper.make_node("Add", ["cd", "z"], ["dets"]), helper.make_node("Add", ["cl", "z"], ["labels"])]
    inits = [numpy_helper.from_array(np.array([1, 1, 1], np.int64), "shp"),
             numpy_helper.from_array(np.full((1, 1, 1), 0.01, np.float32), "k"),
             numpy_helper.from_array(dets, "cd"), numpy_helper.from_array(labels, "cl")]
    g = helper.make_graph(nodes, "dummy", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 384, 384])],
                          [helper.make_tensor_value_info("dets", TensorProto.FLOAT, [1, 300, 4]),
                           helper.make_tensor_value_info("labels", TensorProto.FLOAT, [1, 300, 4])], inits)
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, path)


# ---------------------------------------------------------------- Ch 6 tests
@test("C6-1 tiled dataset: structure, counts, every original box kept, boxes inside images, native scale")
def t_tiled(c6: Path, work: Path):
    import statistics as st
    from PIL import Image
    out = work / "tiled_dataset"
    sh([PY, ROOT / "c6/build_tiled_dataset.py", "--src", c6 / "dataset", "--out", out])
    info = {}
    for split in ("train", "valid"):
        src = json.loads((c6 / "dataset" / split / "_annotations.coco.json").read_text())
        dst = json.loads((out / split / "_annotations.coco.json").read_text())
        assert dst["categories"] == src["categories"], "categories changed"
        files = {im["file_name"] for im in dst["images"]}
        for im in src["images"]:
            assert Path(im["file_name"]).stem + ".jpg" in files, f"original {im['file_name']} missing"
        dims = {im["id"]: (im["width"], im["height"]) for im in dst["images"]}
        for a in dst["annotations"]:
            x, y, w, h = a["bbox"]
            W, H = dims[a["image_id"]]
            assert w > 0 and h > 0 and x >= -0.01 and y >= -0.01 and x + w <= W + 0.01 and y + h <= H + 0.01, f"box outside image {a}"
        sample = random.Random(0).sample(dst["images"], 25)
        for im in sample:
            with Image.open(out / split / im["file_name"]) as p:
                assert p.size == (im["width"], im["height"]), "size mismatch"
        crop_ids = {im["id"] for im in dst["images"] if "_p" in im["file_name"]}
        orig_area = st.median([a["bbox"][2] * a["bbox"][3] for a in src["annotations"]])
        crop_area = st.median([a["bbox"][2] * a["bbox"][3] for a in dst["annotations"] if a["image_id"] in crop_ids])
        assert 0.7 * orig_area <= crop_area <= 1.3 * orig_area, "crop boxes are not at native scale"
        neg = [im for im in dst["images"] if "_n" in im["file_name"]]
        neg_ids = {im["id"] for im in neg}
        assert not any(a["image_id"] in neg_ids for a in dst["annotations"]), "negative crop has a box"
        info[split] = {"images": len(dst["images"]), "boxes": len(dst["annotations"]), "orig_boxes": len(src["annotations"]),
                       "neg_crops": len(neg), "median_box_area_orig": round(orig_area, 1),
                       "median_box_area_crops": round(crop_area, 1)}
    t_src = json.loads((c6 / "dataset/test/_annotations.coco.json").read_text())
    t_dst = json.loads((out / "test/_annotations.coco.json").read_text())
    assert t_src == t_dst and len(list((out / "test").glob("*.jpg"))) == len(t_src["images"]), "test copy differs"
    info["test_split"] = "copied unchanged"
    return info


@test("C6-2 tile grid covers every pixel of every valid/test image size, tiles <= 384")
def t_grid(c6: Path):
    import numpy as np
    from tiling import tile_grid
    sizes = set()
    for split in ("valid", "test"):
        for im in json.loads((c6 / "dataset" / split / "_annotations.coco.json").read_text())["images"]:
            sizes.add((im["width"], im["height"]))
    for m in json.loads((c6 / "tracking_eval/sequences_manifest.json").read_text())["sequences"]:
        sizes.add((m["width"], m["height"]))
    worst = 0
    for W, H in sizes:
        cov = np.zeros((H, W), bool)
        tiles = tile_grid(W, H)
        for x, y, w, h in tiles:
            assert w <= 384 and h <= 384 and x >= 0 and y >= 0 and x + w <= W and y + h <= H
            cov[y:y + h, x:x + w] = True
        assert cov.all(), f"uncovered pixels at {W}x{H}"
        worst = max(worst, len(tiles))
    return {"image_sizes_checked": len(sizes), "max_tiles_per_image": worst}


@test("C6-3 class-wise NMS keeps distinct classes, removes duplicates, caps top-k")
def t_nms():
    from tiling import classwise_nms
    d = [(0, 0.9, [10, 10, 10, 10]), (0, 0.8, [11, 11, 10, 10]), (1, 0.7, [10, 10, 10, 10]), (0, 0.6, [100, 100, 5, 5])]
    k = classwise_nms(d)
    assert [x[1] for x in k] == [0.9, 0.7, 0.6], k
    many = [(0, i / 1000, [i * 20, 0, 10, 10]) for i in range(300)]
    assert len(classwise_nms(many, top_k=100)) == 100
    return {"cases": 2}


@test("C6-4 tiled_detect on ALL valid + test images -> kit COCO scorer accepts; checker accepts test file")
def t_detect(c6: Path, work: Path, onnx: Path):
    sh([PY, ROOT / "c6/tiled_detect.py", "--kit", c6, "--onnx", onnx, "--split", "valid", "--out", work / "preds_valid.json"])
    s = last_json(sh([PY, c6 / "starter-kit/evaluate_submissions.py", "detect", "--gt",
                      c6 / "dataset/valid/_annotations.coco.json", "--preds", work / "preds_valid.json"]))
    o = last_json(sh([PY, ROOT / "c6/tiled_detect.py", "--kit", c6, "--onnx", onnx, "--split", "test", "--out",
                      work / "sub/preds_test.json"]))
    assert o["images"] == 328
    return {"valid_scored_keys": sorted(s)[:6], "test_images": o["images"], "calls_per_image": o["calls_per_image"]}


@test("C6-5 'peek before you stare': do motion windows actually cover the real target? (real videos, no detector)")
def t_motion_cover(c6: Path):
    import cv2
    from detect_video import motion_candidates, pick_windows, read_video
    out = {}
    for seq in ("seq_0001", "seq_0002", "seq_0003", "seq_0004"):
        frames = read_video(c6 / "tracking_eval" / seq / "video.mp4")
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
        H, W = grays[0].shape
        gt = {int(l.split(",")[0]): list(map(float, l.split(",")[2:6]))
              for l in (c6 / "tracking_eval" / seq / "gt/gt.txt").read_text().splitlines()}
        res = {}
        for mw in (6, 12):
            hit = n = 0
            for t in range(1, len(grays) - 1):
                if t % 15 == 0:            # sweep frames cover everything by construction
                    continue
                wins = pick_windows(motion_candidates(grays[t - 1], grays[t], grays[t + 1]), W, H, mw)
                b = gt.get(t + 1)
                if not b:
                    continue
                cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
                n += 1
                hit += any(x <= cx <= x + w and y <= cy <= y + h for x, y, w, h in wins)
            res[f"max_windows_{mw}"] = round(hit / max(1, n), 3)
        out[seq] = res
    return {"fraction_of_non_sweep_frames_where_target_is_inside_a_motion_window": out,
            "note": "motion windows alone (no carry-over from previous detections); carry-over only adds coverage"}


@test("C6-6 detect_video (full + motion) on ALL 8 sequences -> track.py -> kit MOT scorer + checker PASS")
def t_video(c6: Path, work: Path, onnx: Path):
    stats = {}
    for mode in ("full", "motion"):
        sh([PY, ROOT / "c6/detect_video.py", "--kit", c6, "--onnx", onnx, "--mode", mode, "--out", work / f"dets_{mode}.json"])
        d = json.loads((work / f"dets_{mode}.json").read_text())
        assert all(f"seq_000{i}" in d for i in range(1, 9)), "missing sequences"
        stats[mode] = {k: v["model_calls_per_frame"] for k, v in d["_stats"].items() if k.startswith("seq")}
    for variant, extra in (("offline_cam", ["--kit", c6]), ("online", ["--online"])):
        out = work / f"pred_mot_{variant}"
        sh([PY, ROOT / "c6/track.py", "--dets", work / "dets_motion.json", "--out-dir", out, *extra])
        assert len(list(out.glob("seq_*.txt"))) == 8
        last_json(sh([PY, c6 / "starter-kit/evaluate_submissions.py", "mot", "--gt-dir", c6 / "tracking_eval",
                      "--pred-dir", out, "--split", "valid"]))
    (work / "sub").mkdir(exist_ok=True)
    sh(["cp", "-r", work / "pred_mot_offline_cam", work / "sub/pred_mot"])
    calls = {m: round(sum(v.values()) / len(v), 2) for m, v in stats.items()}
    return {"mean_model_calls_per_frame": calls, "per_sequence": stats}


@test("C6-7 tracker output policy on synthetic detections (3 seeds), kit MOT scorer: NOT a real score")
def t_synth(c6: Path, work: Path):
    out = sh([PY, ROOT / "c6/synth_test.py", "--kit", c6, "--work", work / "synth"])
    rows = [json.loads(l.split(": ", 1)[1]) for l in out.strip().splitlines() if l.startswith("seed")]
    for r in rows:
        assert r["policy_on"]["MOTA"] > r["policy_off"]["MOTA"] and r["policy_on"]["IDF1"] > r["policy_off"]["IDF1"]
    return {"seeds": rows}


# ---------------------------------------------------------------- Ch 8 tests
@test("C8-1 pool_clips: ids/order preserved, scores in [0,1], clip-constant ranking, kit scorer accepts, synthetic gain")
def t_pool(c8: Path, work: Path):
    import statistics as st
    rows = [json.loads(l) for l in (c8 / "dataset/valid/labels.jsonl").read_text().splitlines()]
    gains = {"frame": [], "pooled": []}
    for seed in range(10):
        r = random.Random(seed)
        p = work / "c8p.jsonl"
        p.write_text("\n".join(json.dumps({"id": x["id"], "triage_keep": r.random() < 0.3,
                                           "priority_score": round(min(1, max(0, x["priority_score"] + r.gauss(0, 0.25))), 6)})
                               for x in rows) + "\n")
        q = work / "c8q.jsonl"
        sh([PY, ROOT / "c8/pool_clips.py", "--kit", c8, "--split", "valid", "--preds", p, "--out", q])
        P = [json.loads(l) for l in p.read_text().splitlines()]
        Q = [json.loads(l) for l in q.read_text().splitlines()]
        assert [a["id"] for a in P] == [b["id"] for b in Q]
        assert all(0 <= b["priority_score"] <= 1 and isinstance(b["triage_keep"], bool) for b in Q)
        by = {}
        for x, b in zip(rows, Q):
            by.setdefault(x["video_id"], []).append(b["priority_score"])
        assert all(max(v) - min(v) <= 1.01e-3 for v in by.values()), "frames of a clip not kept together"
        for name, f in (("frame", p), ("pooled", q)):
            gains[name].append(last_json(sh([PY, c8 / "starter-kit/evaluate_submissions.py", "triage", "--gt",
                                             c8 / "dataset/valid/labels.jsonl", "--preds", f])))
    mean = {n: {k: round(st.mean(d[k] for d in v), 3) for k in ("recall_at_1pct", "recall_at_5pct", "priority_spearman")}
            for n, v in gains.items()}
    return {"synthetic_noise_sigma_0.25_10_seeds": mean}


@test("C8-2 test-split pooling + checker on Ch 8 test file")
def t_c8_test(c8: Path, work: Path, onnx: Path | None):
    raw = work / "c8_test_raw.jsonl"
    if onnx:
        sh([PY, c8 / "starter-kit/predict_onnx_baseline.py", "triage", "--onnx", onnx, "--dataset-dir", c8 / "dataset",
            "--split", "test", "--out", raw])
    else:
        rows = [json.loads(l) for l in (c8 / "dataset/test/labels.jsonl").read_text().splitlines()]
        raw.write_text("\n".join(json.dumps({"id": x["id"], "triage_keep": False, "priority_score": random.random()}) for x in rows) + "\n")
    sh([PY, ROOT / "c8/pool_clips.py", "--kit", c8, "--split", "test", "--preds", raw, "--out", work / "sub/c8_test.jsonl"])
    return {"used_trained_onnx": bool(onnx)}


# ---------------------------------------------------------------- Ch 9 tests
@test("C9-1 predict_consistent: wake == (class == drone_buzz) on every clip; kit scorer accepts valid; test file written")
def t_c9(c9: Path, work: Path, onnx: Path):
    for split, out in (("valid", work / "c9_valid.jsonl"), ("test", work / "sub/c9_test.jsonl")):
        sh([PY, ROOT / "c9/predict_consistent.py", "--challenge-dir", c9, "--onnx", onnx, "--split", split, "--out", out])
        for l in out.read_text().splitlines():
            r = json.loads(l)
            assert r["wake_triggered"] == (r["sound_class"] == "drone_buzz"), r
    s = last_json(sh([PY, c9 / "starter-kit/evaluate_submissions.py", "acoustic", "--gt", c9 / "dataset/valid/labels.jsonl",
                      "--preds", work / "c9_valid.jsonl"]))
    return {"scorer_keys": sorted(s)}


@test("C9-2 label identity used by the consistency rule holds on all 56 labelled clips")
def t_c9_labels(c9: Path):
    n = 0
    for split in ("train", "valid"):
        for l in (c9 / "dataset" / split / "labels.jsonl").read_text().splitlines():
            r = json.loads(l)
            assert r["has_drone_audio"] == (r["sound_class"] == "drone_buzz")
            n += 1
    return {"clips": n}


# ---------------------------------------------------------------- Metis + checker
@test("M-1 Metis preprocess callbacks == kit inference preprocessing (all calibration images)")
def t_pre(c6: Path, c8: Path, c9: Path):
    import importlib.util
    import numpy as np
    from PIL import Image
    from tiling import load_kit

    def mod(p):
        spec = importlib.util.spec_from_file_location(p.stem, p)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    cnn, rf = mod(ROOT / "metis/preprocess_cnn.py"), mod(ROOT / "metis/preprocess_rfdetr.py")
    kit = load_kit(c6 / "starter-kit")
    worst, n = 0.0, 0
    for f in sorted((c6 / "calibration").glob("*.jpg")):
        with Image.open(f) as im:
            worst = max(worst, float(np.abs(kit._preprocess_imagenet(im, 384)[0] - rf.get_preprocess_transform(im).numpy()).max()))
        n += 1
    for folder in (c8 / "calibration", c9 / "calibration_spectrograms"):
        for f in sorted(folder.iterdir()):
            with Image.open(f) as im:
                worst = max(worst, float(np.abs(kit._preprocess_unit_rgb(im, 224)[0] - cnn.get_preprocess_transform(im).numpy()).max()))
            n += 1
    assert worst < 1e-6
    assert tuple(cnn.get_preprocess_transform_batched(Image.new("RGB", (10, 10))).shape) == (1, 3, 224, 224)
    return {"images_compared": n, "max_abs_diff": worst}


@test("CHK-1 check_submissions PASSES on the complete test set (Ch 6 COCO + 4 MOT, Ch 8, Ch 9)")
def t_chk_ok(c6, c8, c9, work):
    out = sh([PY, ROOT / "check_submissions.py", "--c6-kit", c6, "--c6-preds", work / "sub/preds_test.json",
              "--c6-mot", work / "sub/pred_mot", "--c8-kit", c8, "--c8-preds", work / "sub/c8_test.jsonl",
              "--c9-kit", c9, "--c9-preds", work / "sub/c9_test.jsonl"])
    assert "PASS" in out
    return {}


@test("CHK-2 check_submissions FAILS on 6 injected mistakes (each must be caught)")
def t_chk_bad(c6, c8, c9, work):
    import shutil
    caught = {}
    base = work / "sub"

    def run(**over):
        args = {"--c6-preds": base / "preds_test.json", "--c6-mot": base / "pred_mot",
                "--c8-preds": base / "c8_test.jsonl", "--c9-preds": base / "c9_test.jsonl", **over}
        r = subprocess.run([PY, ROOT / "check_submissions.py", "--c6-kit", c6, "--c8-kit", c8, "--c9-kit", c9,
                            *[str(x) for kv in args.items() for x in kv]], capture_output=True, text=True)
        return r.returncode != 0
    bad = work / "bad"
    if bad.exists():
        shutil.rmtree(bad)
    bad.mkdir()
    # 1 missing MOT file
    shutil.copytree(base / "pred_mot", bad / "mot1"); (bad / "mot1/seq_0007.txt").unlink()
    caught["missing_mot_sequence"] = run(**{"--c6-mot": bad / "mot1"})
    # 2 0-based class in MOT
    shutil.copytree(base / "pred_mot", bad / "mot2")
    f = bad / "mot2/seq_0005.txt"
    L = f.read_text().splitlines()
    if L:
        r = L[0].split(","); r[7] = "0"; L[0] = ",".join(r)
    else:
        L = ["1,1,1,1,5,5,0.9,0,1.0"]
    f.write_text("\n".join(L) + "\n")
    caught["mot_class_0_based"] = run(**{"--c6-mot": bad / "mot2"})
    # 3 duplicate (frame, id)
    shutil.copytree(base / "pred_mot", bad / "mot3")
    f = bad / "mot3/seq_0006.txt"
    f.write_text(f.read_text() + "1,999,1,1,5,5,0.9,1,1.0\n1,999,2,2,5,5,0.9,1,1.0\n")
    caught["mot_duplicate_row"] = run(**{"--c6-mot": bad / "mot3"})
    # 4 COCO 1-based category
    P = json.loads((base / "preds_test.json").read_text())
    P[0]["category_id"] = 3
    (bad / "p.json").write_text(json.dumps(P))
    caught["coco_category_3"] = run(**{"--c6-preds": bad / "p.json"})
    # 5 Ch 8 missing id
    L = (base / "c8_test.jsonl").read_text().splitlines()[1:]
    (bad / "c8.jsonl").write_text("\n".join(L) + "\n")
    caught["c8_missing_id"] = run(**{"--c8-preds": bad / "c8.jsonl"})
    # 6 Ch 9 invalid class
    L = (base / "c9_test.jsonl").read_text().splitlines()
    r = json.loads(L[0]); r["sound_class"] = "drone"; L[0] = json.dumps(r)
    (bad / "c9.jsonl").write_text("\n".join(L) + "\n")
    caught["c9_bad_class"] = run(**{"--c9-preds": bad / "c9.jsonl"})
    assert all(caught.values()), caught
    return {"caught": caught}


@test("RF-1 RF-DETR's own COCO loader accepts the tiled dataset (train + valid)")
def t_rfdetr(work: Path):
    try:
        import rfdetr  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"rfdetr not importable here ({e}); run on the Mac instead") from e
    from types import SimpleNamespace
    from rfdetr.datasets.coco import build_roboflow_from_coco  # the loader RF-DETR training uses
    out = {}
    for name, root in (("tiled", work / "tiled_dataset"), ("kit_original", Path(work.args_c6) / "dataset")):
        args = SimpleNamespace(dataset_dir=str(root), square_resize_div_64=True, segmentation_head=False,
                               multi_scale=False, expanded_scales=False, patch_size=16, num_windows=2)
        res = {}
        for split in ("train", "val"):
            ds = build_roboflow_from_coco(split, args, 384)
            labels = set()
            for i in range(0, len(ds), max(1, len(ds) // 40)):
                _, tgt = ds[i]
                labels |= set(int(v) for v in tgt["labels"].tolist())
            res[split] = {"len": len(ds), "label_ids_seen": sorted(labels)}
        out[name] = res
    assert out["tiled"]["train"]["label_ids_seen"] == out["kit_original"]["train"]["label_ids_seen"], out
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c6", type=Path, required=True)
    ap.add_argument("--c8", type=Path, required=True)
    ap.add_argument("--c9", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--c6-onnx", type=Path, default=None, help="trained full RF-DETR ONNX (default: dummy for plumbing)")
    ap.add_argument("--c8-onnx", type=Path, default=None)
    ap.add_argument("--c9-onnx", type=Path, required=True)
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    (a.work / "sub").mkdir(exist_ok=True)
    type(a.work)  # Path
    globals()["_C6"] = a.c6
    a.work = Path(a.work)
    setattr(Path, "args_c6", a.c6)       # lets RF-1 find the kit dataset without changing signatures
    onnx6 = a.c6_onnx
    if onnx6 is None:
        onnx6 = a.work / "dummy_rfdetr.onnx"
        make_dummy_rfdetr(onnx6)
    t_tiled(a.c6, a.work); t_grid(a.c6); t_nms()
    t_detect(a.c6, a.work, onnx6); t_motion_cover(a.c6); t_video(a.c6, a.work, onnx6); t_synth(a.c6, a.work)
    t_pool(a.c8, a.work); t_c8_test(a.c8, a.work, a.c8_onnx)
    t_c9(a.c9, a.work, a.c9_onnx); t_c9_labels(a.c9)
    t_pre(a.c6, a.c8, a.c9)
    t_chk_ok(a.c6, a.c8, a.c9, a.work); t_chk_bad(a.c6, a.c8, a.c9, a.work)
    t_rfdetr(a.work)
    (a.work / "TEST_REPORT.json").write_text(json.dumps(RESULTS, indent=2))
    n_fail = sum(r["status"] == "FAIL" for r in RESULTS)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} tests passed. Report: {a.work / 'TEST_REPORT.json'}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
