"""Sort the audio_export clips by whether they contain drones, using their AI-written analysis.json.

Groups:
  drone_audio   has_drone_audio is true (the drone can be heard)
  drone_video   a drone is in the video (objects, subjects, events, "of_drone" view, drone footage type)
                or the clip was filmed from a drone, but its audio is not flagged as drone
  no_drone      everything else: combat, shooting, tanks, speech

Writes <out>/clips.csv (one row per clip, with the reason) and id lists used by
distill_cnn.py --extra-ids:
  ids_all.txt               every labelled clip
  ids_drone_any.txt         drone_audio + drone_video          ("only videos that contain drones")
  ids_drone_audio.txt       drone_audio only
  ids_audio_plus_clean.txt  drone_audio + no_drone              (drops the ambiguous silent-drone clips)

  python ch9_acoustic_wake/select_drone_clips.py --extra ~/edth/extra/audio_export --out ~/edth/runs/c9_selection
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

DRONE = re.compile(r"\b(drone|uav|fpv|quadcopter|shahed|geran|lancet|kamikaze|loitering munition|multicopter|octocopter)", re.I)
DRONE_TYPES = {"drone_strike", "surveillance_drone"}


def drone_evidence(a: dict) -> list[str]:
    why = []
    if any(DRONE.search(str(o)) for o in a.get("objects_detected") or []):
        why.append("object")
    if any(DRONE.search(str(s.get("type", "")) + " " + str(s.get("description", ""))) for s in a.get("subjects") or []):
        why.append("subject")
    if any(DRONE.search(str(e.get("description", ""))) for e in a.get("events") or []):
        why.append("event")
    if a.get("footage_perspective") == "of_drone":
        why.append("of_drone")
    if a.get("footage_perspective") == "from_drone":
        why.append("from_drone")
    if a.get("footage_type") in DRONE_TYPES:
        why.append(a["footage_type"])
    return why


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extra", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for d in sorted(p for p in args.extra.iterdir() if (p / "analysis.json").exists() and (p / "audio.wav").exists()):
        a = json.loads((d / "analysis.json").read_text())
        why = drone_evidence(a)
        group = "drone_audio" if a.get("has_drone_audio") else ("drone_video" if why else "no_drone")
        rows.append({"id": d.name, "group": group, "has_drone_audio": bool(a.get("has_drone_audio")),
                     "footage_type": a.get("footage_type"), "perspective": a.get("footage_perspective"),
                     "evidence": "+".join(why), "audio": (a.get("audio_description") or "")[:120]})

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "clips.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    sets = {
        "all": [r for r in rows],
        "drone_any": [r for r in rows if r["group"] != "no_drone"],
        "drone_audio": [r for r in rows if r["group"] == "drone_audio"],
        "audio_plus_clean": [r for r in rows if r["group"] != "drone_video"],
    }
    for name, rs in sets.items():
        (args.out / f"ids_{name}.txt").write_text("".join(r["id"] + "\n" for r in rs))
    for g in ("drone_audio", "drone_video", "no_drone"):
        print(f"{g:12s} {sum(r['group'] == g for r in rows)}")
    for name, rs in sets.items():
        print(f"ids_{name}.txt: {len(rs)} clips")


if __name__ == "__main__":
    main()
