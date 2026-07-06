#!/usr/bin/env python3
"""
多半徑候選視角生成器（A-1 多半徑版）。

對 HEMISPHERE_RADII_M 中每個半徑各自取樣並過濾，合併輸出至
data/viewpoints/candidate_viewpoints_multi.json。

使用方式：
  /usr/bin/python3 generate_candidate_viewpoints_multi.py
  /usr/bin/python3 generate_candidate_viewpoints_multi.py --radii 0.5 0.65 0.8
  /usr/bin/python3 generate_candidate_viewpoints_multi.py --x-offset 0.1
"""

import argparse
import importlib.util
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_VIEWPOINTS = os.path.join(REPO_ROOT, "data", "viewpoints")

# 引入單半徑版的所有工具函式，不重複維護
_spec = importlib.util.spec_from_file_location(
    "gen_single",
    os.path.join(SCRIPT_DIR, "generate_candidate_viewpoints.py"),
)
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)

import candidate_viewpoint_config as config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--radii", type=float, nargs="+",
                        default=config.HEMISPHERE_RADII_M,
                        help=f"半球半徑清單（預設來自 config: {config.HEMISPHERE_RADII_M}）")
    parser.add_argument("--x-offset", type=float, default=0.0,
                        help="物體中心 x 軸偏移（m），拍攝球體與工作球體同步移動（預設 0.0）")
    parser.add_argument("--output", default=None,
                        help="輸出路徑（預設依 x_offset 自動命名）")
    parser.add_argument("--moveit-ik", action="store_true",
                        help="改用 MoveIt 解關節角：每視角的多組 IK 解逐一送 MoveIt 規劃，"
                             "挑第一個無自撞+從Home可達者（需先 ros2 launch planning_bridge）")
    args = parser.parse_args()

    # 套用 x_offset：覆蓋 _gen 模組的 OBJECT_CENTER_M
    import numpy as np
    x_off = args.x_offset
    new_center = np.array([config.OBJECT_CENTER_M[0] + x_off,
                           config.OBJECT_CENTER_M[1],
                           config.OBJECT_CENTER_M[2]], dtype=float)
    _gen.OBJECT_CENTER_M = new_center

    # 輸出路徑：x_offset=0 維持舊名，非 0 加上 x tag
    if args.output is not None:
        output_path = args.output
    elif x_off == 0.0:
        output_path = os.path.join(DATA_VIEWPOINTS, "candidate_viewpoints_multi.json")
    else:
        x_tag = f"x{int(x_off * 100):+04d}"
        output_path = os.path.join(DATA_VIEWPOINTS, f"candidate_viewpoints_multi_{x_tag}.json")

    print("多半徑候選視角生成器")
    print(f"  Robot base  : {config.ROBOT_BASE_M} m")
    print(f"  Object      : {new_center.tolist()} m  (x_offset={x_off:+.3f})")
    print(f"  Radii       : {args.radii} m")
    print(f"  Elevations  : {config.ELEVATION_ANGLES_DEG} deg")
    print(f"  Azimuths    : {config.AZIMUTH_STEPS} steps")
    print(f"  Output      : {output_path}")

    if args.moveit_ik:
        print("  IK 模式     : MoveIt（多解逐一規劃挑無自撞+可達）")
        _gen.start_moveit_bridge()
    else:
        print("  IK 模式     : Webots 數值 IK + capsule 自碰撞（原行為）")

    import math

    def _base_record(rid, r, p_cam, j_deg):
        el, az = _gen._elevation_azimuth(p_cam)
        joints_rad = [math.radians(v) for v in j_deg]
        return {
            "id": rid,
            "radius_m": float(r),
            "joint_deg": [round(v, 4) for v in j_deg],
            "camera_position_m": [float(v) for v in p_cam],
            "elevation_deg": float(el),
            "azimuth_deg": float(az),
            "target_err_deg": _gen.camera_target_angle_deg(joints_rad, _gen.OBJECT_CENTER_M),
            "ray_miss_m": _gen.camera_ray_miss_distance_m(joints_rad, _gen.OBJECT_CENTER_M),
            "roll_err_deg": _gen.camera_roll_error_deg(joints_rad, _gen.OBJECT_CENTER_M),
        }

    all_records = []
    try:
        for r in args.radii:
            _gen.HEMISPHERE_RADIUS_M = r
            print(f"\n  半徑 {r}m ...")
            if args.moveit_ik:
                # MoveIt 模式:每視角取「殘差最小中第一個規劃成功」的解 + 記錄 home→視角路徑。
                for p_cam, sol in _gen.moveit_solve_viewpoints():
                    rec = _base_record(len(all_records) + 1, r, p_cam, sol["joint_deg"])
                    rec["n_ik_solutions"] = sol["n_ik_solutions"]
                    rec["ik_rank_used"] = sol["ik_rank_used"]
                    rec["n_waypoints"] = sol["n_waypoints"]
                    rec["path_rad"] = sol["path_rad"]
                    all_records.append(rec)
            else:
                valid = _gen.find_valid_viewpoints()
                print(f"    有效: {len(valid)} 個")
                for p_cam, j_deg in valid:
                    all_records.append(_base_record(len(all_records) + 1, r, p_cam, j_deg))
    finally:
        if args.moveit_ik:
            _gen.stop_moveit_bridge()

    if not all_records:
        print("ERROR: 沒有有效的候選點")
        sys.exit(1)

    os.makedirs(DATA_VIEWPOINTS, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)

    print(f"\n總計: {len(all_records)} 個候選點（{len(args.radii)} 個半徑）")
    for r in args.radii:
        count = sum(1 for rec in all_records if rec["radius_m"] == r)
        print(f"  r={r}m: {count} 個")
    print(f"輸出: {output_path}")


if __name__ == "__main__":
    main()
