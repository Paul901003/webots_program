#!/usr/bin/env python3
"""Select final spread-out viewpoints from validated_viewpoints.json."""

import json
import math
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
FOUR_VIEW_DIR = REPO_ROOT / "controllers" / "ycb_supervisor_capture"
sys.path.insert(0, str(FOUR_VIEW_DIR))

import candidate_viewpoint_config as planner_config  # noqa: E402


VALIDATED_PATH = FOUR_VIEW_DIR / "validated_viewpoints.json"
SELECTED_PATH = FOUR_VIEW_DIR / "selected_viewpoints.json"
TARGET_M = planner_config.OBJECT_CENTER_M
NUM_OUTPUT_POSES = planner_config.NUM_OUTPUT_POSES
ROLL_SELECTION_PENALTY = 0.05


def norm(vector):
    length = math.sqrt(sum(value * value for value in vector))
    if length < 1e-12:
        return [0.0, 0.0, 0.0]
    return [value / length for value in vector]


def subtract(a, b):
    return [a[i] - b[i] for i in range(3)]


def dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def angle_between(a, b):
    na = norm(a)
    nb = norm(b)
    return math.degrees(math.acos(max(-1.0, min(1.0, dot(na, nb)))))


def candidate_position(record):
    meta = record.get("meta", {})
    position = meta.get("camera_position_m")
    if isinstance(position, list) and len(position) == 3:
        return [float(value) for value in position]
    ray = record.get("ray", {})
    position = ray.get("ray_origin_m")
    if isinstance(position, list) and len(position) == 3:
        return [float(value) for value in position]
    return None


def candidate_roll_error(record):
    ray = record.get("ray", {})
    if isinstance(ray.get("roll_err_deg"), (int, float)):
        return float(ray["roll_err_deg"])
    meta = record.get("meta", {})
    if isinstance(meta.get("roll_err_deg"), (int, float)):
        return float(meta["roll_err_deg"])
    return 180.0


def select_spread_viewpoints(validated, count):
    pool = [record for record in validated if candidate_position(record) is not None]
    selected = []
    selected_dirs = []
    while pool and len(selected) < count:
        if not selected_dirs:
            best_index = min(range(len(pool)), key=lambda i: candidate_roll_error(pool[i]))
        else:
            best_index = max(
                range(len(pool)),
                key=lambda i: min(
                    angle_between(
                        subtract(candidate_position(pool[i]), TARGET_M),
                        selected_dir,
                    )
                    for selected_dir in selected_dirs
                ) - ROLL_SELECTION_PENALTY * candidate_roll_error(pool[i]),
            )
        record = pool.pop(best_index)
        selected.append(record)
        selected_dirs.append(subtract(candidate_position(record), TARGET_M))
    return selected


def print_camera_poses(selected):
    print("CAMERA_POSES = {")
    for index, record in enumerate(selected, start=1):
        rounded = [round(float(value), 2) for value in record["joint_deg"]]
        position = candidate_position(record)
        delta = subtract(position, TARGET_M)
        dist = math.sqrt(dot(delta, delta))
        el = math.degrees(math.asin(max(-1.0, min(1.0, delta[2] / max(dist, 1e-12)))))
        az = math.degrees(math.atan2(delta[1], delta[0]))
        ray_miss_mm = record["ray"]["ray_miss_m"] * 1000.0
        roll_err = candidate_roll_error(record)
        print(
            f"    {index}: {{\"joint_deg\": {rounded}}},  "
            f"# source_id={record['id']} el={el:.0f} az={az:.0f} "
            f"ray_miss={ray_miss_mm:.1f}mm roll={roll_err:.1f}deg"
        )
    print("}")


def main():
    data = json.loads(VALIDATED_PATH.read_text(encoding="utf-8"))
    validated = data.get("validated", [])
    selected = select_spread_viewpoints(validated, NUM_OUTPUT_POSES)
    result = {
        "source": str(VALIDATED_PATH),
        "target_m": TARGET_M,
        "requested_count": NUM_OUTPUT_POSES,
        "validated_count": len(validated),
        "selected_count": len(selected),
        "selected": selected,
    }
    SELECTED_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Selected {len(selected)}/{len(validated)} validated poses.")
    print_camera_poses(selected)
    print(f"Wrote {SELECTED_PATH}")


if __name__ == "__main__":
    main()
