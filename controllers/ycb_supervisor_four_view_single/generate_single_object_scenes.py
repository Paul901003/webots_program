#!/usr/bin/env python3
"""Generate single_scene_plan.json for ycb_supervisor_four_view_single.

Reads viewpoints from data/viewpoints/selected_viewpoints.json and creates
one scene per object from TARGET_OBJECTS (or ALL_OBJECTS if TARGET_OBJECTS
is empty).
"""

import json
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(CURRENT_DIR))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import ALL_OBJECTS, MASS_TABLE, TARGET_OBJECTS  # noqa: E402

SELECTED_VIEWPOINTS_PATH = os.path.join(REPO_ROOT, "data", "viewpoints", "selected_viewpoints.json")
SCENE_PLAN_PATH          = os.path.join(REPO_ROOT, "data", "scene_plans", "single_scene_plan.json")


def load_viewpoints():
    with open(SELECTED_VIEWPOINTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    selected = data.get("selected", [])
    if not selected:
        raise ValueError(f"selected_viewpoints.json 中沒有 selected 欄位: {SELECTED_VIEWPOINTS_PATH}")
    return [
        {"id": i + 1, "joint_deg": record["joint_deg"]}
        for i, record in enumerate(selected)
    ]


def get_object_pool():
    pool = TARGET_OBJECTS[:] if TARGET_OBJECTS else ALL_OBJECTS[:]
    return [name for name in pool if name in MASS_TABLE]


def main():
    viewpoints = load_viewpoints()
    object_pool = get_object_pool()
    n_views = len(viewpoints)

    scenes = [
        {
            "scene_name": f"{name}_{n_views}views",
            "objects": [{"name": name}],
            "viewpoints": viewpoints,
        }
        for name in object_pool
    ]

    plan = {"scenes": scenes}
    os.makedirs(os.path.dirname(SCENE_PLAN_PATH), exist_ok=True)
    with open(SCENE_PLAN_PATH, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    print(f"生成 {len(scenes)} 個場景（每場景 {n_views} 個視角）")
    print(f"輸出: {SCENE_PLAN_PATH}")


if __name__ == "__main__":
    main()
