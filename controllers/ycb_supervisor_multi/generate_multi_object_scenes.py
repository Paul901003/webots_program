import itertools
import json
import os
import random
import sys
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import (  # noqa: E402
    ALL_OBJECTS,
    MASS_TABLE,
    MULTI_MIN_APPEARANCES,
    MULTI_OBJECT_COUNT,
    MULTI_SCENE_FILE,
    MULTI_SCENE_RANDOM_SEED,
    TARGET_OBJECTS,
)

SCENE_PLAN_PATH = os.path.join(CURRENT_DIR, MULTI_SCENE_FILE)


def get_capture_object_pool():
    return TARGET_OBJECTS[:] if TARGET_OBJECTS else ALL_OBJECTS[:]


def build_multi_object_scenes(object_pool, group_size, min_appearances, rng):
    if len(object_pool) < group_size:
        raise ValueError(
            f"物體數量不足，無法每個場景放 {group_size} 個物體，目前只有 {len(object_pool)} 個"
        )

    combos = list(itertools.combinations(sorted(object_pool), group_size))
    appearance_counts = {name: 0 for name in object_pool}
    scenes = []
    used_combos = set()

    while min(appearance_counts.values()) < min_appearances:
        best_combo = None
        best_score = None

        for combo in combos:
            if combo in used_combos:
                continue

            deficits = [max(0, min_appearances - appearance_counts[name]) for name in combo]
            score = (
                sum(deficits),
                min(deficits),
                -sum(appearance_counts[name] for name in combo),
                rng.random(),
            )
            if best_score is None or score > best_score:
                best_score = score
                best_combo = combo

        if best_combo is None or best_score[0] <= 0:
            break

        scenes.append(list(best_combo))
        used_combos.add(best_combo)
        for name in best_combo:
            appearance_counts[name] += 1

    return scenes, appearance_counts


def validate_scene_objects(scenes):
    for scene_index, scene in enumerate(scenes, start=1):
        if not isinstance(scene, list) or not scene:
            raise ValueError(f"場景 {scene_index} 格式錯誤")
        for object_name in scene:
            if object_name not in MASS_TABLE:
                raise ValueError(f"場景 {scene_index} 含有未知物體: {object_name}")


def save_scene_plan(scene_plan_path, scenes, metadata):
    payload = {
        "metadata": metadata,
        "scenes": scenes,
    }
    with open(scene_plan_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def build_timestamped_scene_plan_path(scene_plan_path):
    directory = os.path.dirname(scene_plan_path)
    filename = os.path.basename(scene_plan_path)
    stem, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(directory, f"{stem}_{timestamp}{ext}")


def main():
    object_pool = get_capture_object_pool()
    rng = random.Random(MULTI_SCENE_RANDOM_SEED)
    scenes, appearance_counts = build_multi_object_scenes(
        object_pool=object_pool,
        group_size=MULTI_OBJECT_COUNT,
        min_appearances=MULTI_MIN_APPEARANCES,
        rng=rng,
    )
    validate_scene_objects(scenes)

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "object_pool_size": len(object_pool),
        "multi_object_count": MULTI_OBJECT_COUNT,
        "multi_min_appearances": MULTI_MIN_APPEARANCES,
        "random_seed": MULTI_SCENE_RANDOM_SEED,
        "scene_count": len(scenes),
        "appearance_counts": appearance_counts,
        "target_objects": TARGET_OBJECTS[:],
    }
    output_path = build_timestamped_scene_plan_path(SCENE_PLAN_PATH)
    save_scene_plan(output_path, scenes, metadata)

    print(f"saved {len(scenes)} scenes to {output_path}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
