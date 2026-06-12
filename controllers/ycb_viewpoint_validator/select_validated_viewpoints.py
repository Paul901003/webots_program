#!/usr/bin/env python3
"""A-3: 從 validated_viewpoints 中選取分布最廣的視角子集。

單半徑：  validated_viewpoints_latest.json        → selected_viewpoints[_x+022].json
多半徑：  validated_viewpoints_multi_latest.json  → selected_viewpoints_multi[_x+022].json
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
VIEWPOINTS_DIR = REPO_ROOT / "data" / "viewpoints"
CAPTURE_CONFIG_DIR = REPO_ROOT / "controllers" / "ycb_supervisor_capture"
sys.path.insert(0, str(CAPTURE_CONFIG_DIR))

import candidate_viewpoint_config as planner_config  # noqa: E402

NUM_OUTPUT_POSES = planner_config.NUM_OUTPUT_POSES
ROLL_SELECTION_PENALTY = 0.05


def _find_validated_files(multi, x_offset_m):
    """掃描所有符合 x_offset 的 A-2 具名輸出檔，排除 _latest。"""
    base_val = "validated_viewpoints_multi" if multi else "validated_viewpoints"
    if x_offset_m != 0.0:
        x_tag = f"_x{int(x_offset_m * 100):+04d}"
        files = sorted(VIEWPOINTS_DIR.glob(f"{base_val}_*{x_tag}*.json"))
    else:
        files = sorted(VIEWPOINTS_DIR.glob(f"{base_val}_*.json"))
        # 排除其他 x_offset 的檔案（帶有 _x+ 或 _x- 的）
        files = [f for f in files if "_x+" not in f.name and "_x-" not in f.name]
    # 排除 _latest.json
    return [f for f in files if "_latest" not in f.name]


def _make_paths(multi, x_offset_m):
    x_tag = f"_x{int(x_offset_m * 100):+04d}" if x_offset_m != 0.0 else ""
    base_sel = "selected_viewpoints_multi" if multi else "selected_viewpoints"
    selected_named  = VIEWPOINTS_DIR / f"{base_sel}{x_tag}.json"
    selected_latest = VIEWPOINTS_DIR / f"{base_sel}_latest.json"
    return selected_named, selected_latest


def norm(v):
    l = math.sqrt(sum(x * x for x in v))
    return [x / l for x in v] if l > 1e-12 else [0.0, 0.0, 0.0]


def subtract(a, b):
    return [a[i] - b[i] for i in range(3)]


def dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def angle_between(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, dot(norm(a), norm(b))))))


def candidate_position(record):
    for key in (("meta", "camera_position_m"), ("ray", "ray_origin_m")):
        pos = record.get(key[0], {}).get(key[1])
        if isinstance(pos, list) and len(pos) == 3:
            return [float(v) for v in pos]
    return None


def candidate_roll_error(record):
    for key in (("ray", "roll_err_deg"), ("meta", "roll_err_deg")):
        v = record.get(key[0], {}).get(key[1])
        if isinstance(v, (int, float)):
            return float(v)
    return 180.0


def select_spread_viewpoints(validated, count, target_m):
    pool = [r for r in validated if candidate_position(r) is not None]
    selected, dirs = [], []
    up = [0.0, 0.0, 1.0]
    while pool and len(selected) < count:
        if not dirs:
            best = max(range(len(pool)),
                       key=lambda i: dot(norm(subtract(candidate_position(pool[i]), target_m)), up))
        else:
            best = max(range(len(pool)),
                       key=lambda i: min(
                           angle_between(subtract(candidate_position(pool[i]), target_m), d)
                           for d in dirs
                       ) - ROLL_SELECTION_PENALTY * candidate_roll_error(pool[i]))
        record = pool.pop(best)
        selected.append(record)
        dirs.append(subtract(candidate_position(record), target_m))
    return selected


def print_camera_poses(selected, target_m):
    print("CAMERA_POSES = {")
    for idx, record in enumerate(selected, 1):
        rounded = [round(float(v), 2) for v in record["joint_deg"]]
        pos = candidate_position(record)
        delta = subtract(pos, target_m)
        dist = math.sqrt(dot(delta, delta))
        el = math.degrees(math.asin(max(-1.0, min(1.0, delta[2] / max(dist, 1e-12)))))
        az = math.degrees(math.atan2(delta[1], delta[0]))
        roll_err = candidate_roll_error(record)
        print(f"    {idx}: {{\"joint_deg\": {rounded}}},  "
              f"# id={record['id']} el={el:.0f} az={az:.0f} roll={roll_err:.1f}deg")
    print("}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--multi", action="store_true", help="多半徑模式")
    parser.add_argument("--x-offset", type=float, default=0.0,
                        help="物體 x 軸偏移（m），與 A-1/A-2 一致（預設 0.0）")
    parser.add_argument("--count", type=int, default=None,
                        help=f"選取視角數（預設 {NUM_OUTPUT_POSES}，來自 config）")
    parser.add_argument("--radius", type=float, default=None,
                        help="強制指定拍攝半徑（m），不指定則自動選最大可用半徑")
    args = parser.parse_args()

    count = args.count or NUM_OUTPUT_POSES
    oc = list(planner_config.OBJECT_CENTER_M)
    target_m = [oc[0] + args.x_offset, oc[1], oc[2]]

    selected_named, selected_latest = _make_paths(args.multi, args.x_offset)

    # 掃描所有符合 x_offset 的 A-2 具名檔並合併
    validated_files = _find_validated_files(args.multi, args.x_offset)
    if not validated_files:
        print(f"ERROR: 找不到任何符合 x_offset={args.x_offset:+.3f} 的 A-2 輸出檔")
        sys.exit(1)

    validated = []
    seen_ids = set()
    for vf in validated_files:
        print(f"[A-3] 讀取: {vf.name}")
        data = json.loads(vf.read_text(encoding="utf-8"))
        for record in data.get("validated", []):
            uid = (record.get("id"), str(record.get("source", "")))
            if uid not in seen_ids:
                seen_ids.add(uid)
                validated.append(record)

    if not validated:
        print("ERROR: 所有輸入檔的 validated 欄位皆為空")
        sys.exit(1)

    print(f"[A-3] 合併後共 {len(validated)} 個通過視角，來自 {len(validated_files)} 個檔案")
    print(f"[A-3] target_m = {target_m}  count = {count}")

    # 方位角群組（度）：group_a=90/135，group_b=270/225（=-90/-135）
    AZ_GROUP_A = {90, 135}
    AZ_GROUP_B = {270, -90, 225, -135}

    def az_normalized(v):
        az = v.get("meta", {}).get("azimuth_deg", 0)
        return round(float(az))

    def has_group_a(pool):
        return any(az_normalized(v) in AZ_GROUP_A for v in pool)

    def has_group_b(pool):
        return any(az_normalized(v) in AZ_GROUP_B for v in pool)

    def satisfies_az_constraint(pool):
        return has_group_a(pool) and has_group_b(pool)

    # 依半徑分組，從大到小找第一個滿足方位角限制且視角數 >= count 的半徑
    from collections import defaultdict
    by_radius = defaultdict(list)
    for v in validated:
        r = v.get("meta", {}).get("radius_m") or planner_config.HEMISPHERE_RADIUS_M
        by_radius[round(float(r), 3)].append(v)

    if args.radius is not None:
        best_radius = round(float(args.radius), 3)
        if best_radius not in by_radius:
            print(f"ERROR: 指定半徑 {best_radius}m 沒有通過的視角")
            sys.exit(1)
        print(f"[A-3] 強制使用半徑 {best_radius}m")
    else:
        best_radius = None
        for r in sorted(by_radius.keys(), reverse=True):
            pool_r = by_radius[r]
            if len(pool_r) >= count and satisfies_az_constraint(pool_r):
                best_radius = r
                break
        if best_radius is None:
            candidates_r = [r for r in by_radius if satisfies_az_constraint(by_radius[r])]
            if candidates_r:
                best_radius = max(candidates_r, key=lambda r: (r, len(by_radius[r])))
            else:
                best_radius = max(by_radius.keys(), key=lambda r: len(by_radius[r]))
                print("[A-3] 警告：無任何半徑能滿足方位角限制，使用視角數最多的半徑")

    pool = by_radius[best_radius]
    print(f"[A-3] 選用半徑 {best_radius}m（{len(pool)} 個視角，工作球半徑最大）")
    print(f"[A-3] 方位角群組 A(90/135): {'有' if has_group_a(pool) else '無'}  群組 B(270/225): {'有' if has_group_b(pool) else '無'}")

    low_el = [v for v in pool if abs(v.get("meta", {}).get("elevation_deg", 90)) < 30]

    if not low_el:
        print("[A-3] 警告：無 20 度仰角視角可選")

    # 先從群組 A 和群組 B 各保留一個（分布最佳的），再保留 20 度仰角一個，其餘空間分布選取
    forced = []
    used = set()

    # 群組 A：az=90/135，選 roll 最小的
    group_a_pool = [v for v in pool if az_normalized(v) in AZ_GROUP_A]
    if group_a_pool:
        pick_a = min(group_a_pool, key=lambda v: candidate_roll_error(v))
        forced.append(pick_a)
        used.add(id(pick_a))
        print(f"[A-3] 群組 A 保留 1 個: el={round(pick_a.get('meta',{}).get('elevation_deg',0))}  az={az_normalized(pick_a)}")

    # 群組 B：az=270/225，選 roll 最小的
    group_b_pool = [v for v in pool if az_normalized(v) in AZ_GROUP_B]
    if group_b_pool:
        pick_b = min(group_b_pool, key=lambda v: candidate_roll_error(v))
        forced.append(pick_b)
        used.add(id(pick_b))
        print(f"[A-3] 群組 B 保留 1 個: el={round(pick_b.get('meta',{}).get('elevation_deg',0))}  az={az_normalized(pick_b)}")

    # 20 度仰角：選 roll 最小的（未已選）
    low_el_avail = [v for v in low_el if id(v) not in used]
    if low_el_avail:
        pick_low = min(low_el_avail, key=lambda v: candidate_roll_error(v))
        forced.append(pick_low)
        used.add(id(pick_low))
        print(f"[A-3] 20 度仰角保留 1 個: el={round(pick_low.get('meta',{}).get('elevation_deg',0))}  az={az_normalized(pick_low)}")
    else:
        print("[A-3] 警告：無額外 20 度仰角視角可選")

    # 剩餘視角用空間分布算法補滿
    remaining_pool = [v for v in pool if id(v) not in used]
    rest = select_spread_viewpoints(remaining_pool, count - len(forced), target_m)
    selected = forced + rest

    result = {
        "sources": [str(f) for f in validated_files],
        "target_m": target_m,
        "x_offset_m": args.x_offset,
        "requested_count": count,
        "validated_count": len(validated),
        "selected_count": len(selected),
        "selected": selected,
    }
    selected_named.parent.mkdir(parents=True, exist_ok=True)
    selected_named.write_text(json.dumps(result, indent=2), encoding="utf-8")
    shutil.copy2(selected_named, selected_latest)

    print(f"選取 {len(selected)}/{len(validated)} 個視角")
    print_camera_poses(selected, target_m)
    print(f"輸出: {selected_named}")
    print(f"最新: {selected_latest}")


if __name__ == "__main__":
    main()
