"""
A-0: 掃描不同 x 偏移的視角可達率，用來選定物體最佳擺放位置。

不需要任何前置步驟，直接從 IK 生成候選視角並以工作空間球體測試 MoveIt 可達性。
選定 x_offset 後再跑主流程：A-1 → A-2 → A-3 → A-4 → A-5。

前置條件：
  已啟動 planning bridge：
  ros2 launch ur5e_2f140_planning planning_bridge_launch.py

使用方式：
  /usr/bin/python3 scan_x_offset.py --multi
  /usr/bin/python3 scan_x_offset.py --multi --x-offsets 0.0 0.1 0.2 0.3
  /usr/bin/python3 scan_x_offset.py --multi --ws-offset 0.15 --x-offsets 0.1 0.2 0.22 0.25

工作空間球體半徑 = cam_r - ws_offset（固定差值）

輸出：
  data/viewpoints/x_offset_scan_{TAG}.json   ← 具名
  data/viewpoints/x_offset_scan_latest.json  ← 最新指標
"""

import argparse
import importlib.util
import json
import math
import os
import shutil
import sys
import time

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT      = os.path.dirname(os.path.dirname(SCRIPT_DIR))
VIEWPOINTS_DIR = os.path.join(REPO_ROOT, "data", "viewpoints")
GEN_DIR        = os.path.join(REPO_ROOT, "controllers", "ycb_supervisor_capture")

sys.path.insert(0, GEN_DIR)

ROBOT_BASE_M = [-0.4, 0.0, 0.0]
HOME_DEG     = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]
HOME_RAD     = [math.radians(d) for d in HOME_DEG]

RESULT_WAIT_SEC   = 15.0
NUM_PLAN_ATTEMPTS = 5
ALLOWED_PLAN_TIME = 10.0


def _load_gen():
    spec = importlib.util.spec_from_file_location(
        "gen_single",
        os.path.join(GEN_DIR, "generate_candidate_viewpoints.py"),
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    return gen


def generate_candidates(gen, radii, x_offset_m):
    import numpy as np
    import candidate_viewpoint_config as cfg
    center = np.array([cfg.OBJECT_CENTER_M[0] + x_offset_m,
                       cfg.OBJECT_CENTER_M[1],
                       cfg.OBJECT_CENTER_M[2]], dtype=float)
    gen.OBJECT_CENTER_M = center

    records = []
    for r in radii:
        gen.HEMISPHERE_RADIUS_M = r
        valid = gen.find_valid_viewpoints()
        for p_cam, j_deg in valid:
            records.append({
                "id": f"x{x_offset_m:+.3f}_r{r:.2f}_{len(records)}",
                "radius_m": float(r),
                "x_offset_m": float(x_offset_m),
                "joint_deg": j_deg,
                "joint_rad": [math.radians(d) for d in j_deg],
            })
    return records


def _object_center_base_link(x_offset_m):
    import candidate_viewpoint_config as cfg
    return [
        cfg.OBJECT_CENTER_M[0] + x_offset_m - ROBOT_BASE_M[0],
        cfg.OBJECT_CENTER_M[1]               - ROBOT_BASE_M[1],
        cfg.OBJECT_CENTER_M[2]               - ROBOT_BASE_M[2],
    ]


class LatestResult:
    def __init__(self):
        self._data = None

    def callback(self, msg):
        try:
            self._data = json.loads(msg.data)
        except Exception:
            pass

    def get_latest(self):
        d, self._data = self._data, None
        return d


def plan_home_to_vp(node, pub, sub_result, vp_id, target_rad, collision_objects):
    import rclpy
    from std_msgs.msg import String

    plan_id = f"scan_{vp_id}"
    req = {
        "id": plan_id,
        "start_joints": list(HOME_RAD),
        "target_joints": list(target_rad),
        "collision_objects": collision_objects,
        "velocity_scaling": 0.5,
        "acceleration_scaling": 0.5,
        "num_planning_attempts": NUM_PLAN_ATTEMPTS,
        "allowed_planning_time": ALLOWED_PLAN_TIME,
    }
    msg = String()
    msg.data = json.dumps(req)
    pub.publish(msg)

    deadline = time.time() + RESULT_WAIT_SEC
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        result = sub_result.get_latest()
        if result and result.get("id") == plan_id:
            return result.get("success", False)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--multi", action="store_true",
                        help="多半徑模式（使用 HEMISPHERE_RADII_M）")
    parser.add_argument("--x-offsets", type=float, nargs="+",
                        default=[0.0, 0.05, 0.1, 0.15, 0.2],
                        help="掃描的 x 軸偏移清單（m），預設: 0.0 0.05 0.1 0.15 0.2")
    parser.add_argument("--ws-offset", type=float, default=0.2,
                        help="sphere_r = cam_r - ws_offset（預設 0.2）")
    args = parser.parse_args()

    import candidate_viewpoint_config as cfg
    radii = cfg.HEMISPHERE_RADII_M if args.multi else [cfg.HEMISPHERE_RADIUS_M]

    print(f"掃描 x 偏移: {args.x_offsets}")
    print(f"拍攝半徑: {radii}")
    print(f"ws_offset = {args.ws_offset} m  （sphere_r = cam_r - {args.ws_offset}）")
    print()

    try:
        import rclpy
        from std_msgs.msg import String
    except ImportError:
        print("ERROR: 需要 ROS2 環境")
        sys.exit(1)

    rclpy.init()
    node    = rclpy.create_node("scan_x_offsets")
    pub     = node.create_publisher(String, "/ur5e/plan_request", 10)
    sub_res = LatestResult()
    node.create_subscription(String, "/ur5e/plan_result", sub_res.callback, 10)

    print("等待 planning bridge...")
    deadline = time.time() + 3.0
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    print()

    gen = _load_gen()
    results = {}  # x_offset → {pass, total, by_radius}

    for x_off in args.x_offsets:
        print(f"── x_offset = {x_off:+.3f}m ──")
        candidates = generate_candidates(gen, radii, x_off)
        print(f"  候選視角: {len(candidates)} 個")

        passed = 0
        by_r   = {}
        for vp in candidates:
            sphere_r = vp["radius_m"] - args.ws_offset
            col = [{
                "id": "ycb_workspace_sphere",
                "shape": "sphere",
                "size": [sphere_r * 2],
                "position": _object_center_base_link(x_off),
            }]
            ok = plan_home_to_vp(node, pub, sub_res, vp["id"], vp["joint_rad"], col)
            if ok:
                passed += 1
                by_r[vp["radius_m"]] = by_r.get(vp["radius_m"], 0) + 1

        total_by_r = {}
        for r in radii:
            total_by_r[r] = sum(1 for v in candidates if v["radius_m"] == r)

        total = len(candidates)
        results[x_off] = {"pass": passed, "total": total, "by_radius": by_r, "total_by_radius": total_by_r}
        print(f"  通過: {passed}/{total}")
        for r in sorted(radii):
            print(f"    cam_r={r:.2f}m  sphere_r={r - args.ws_offset:.2f}m  {by_r.get(r, 0)}/{total_by_r[r]}")
        print()

    node.destroy_node()
    rclpy.shutdown()

    # 輸出表格
    sorted_radii = sorted(radii)
    col = 14
    sep = "=" * (12 + 9 + len(sorted_radii) * col)
    print(sep)
    print(f"ws_offset={args.ws_offset}m  →  workspace sphere_r = cam_r - {args.ws_offset}")
    print()
    # 表頭兩行：第一行 cam_r，第二行對應 ws_r
    head1 = f"  {'x_offset':>8}  {'合計':>6}"
    head2 = f"  {'':>8}  {'':>6}"
    for r in sorted_radii:
        head1 += f"  {'cam_r='+f'{r:.2f}m':>{col-2}}"
        head2 += f"  {'ws_r='+f'{r-args.ws_offset:.2f}m':>{col-2}}"
    print(head1)
    print(head2)
    print("-" * len(head1))
    for x_off in args.x_offsets:
        r_dict = results[x_off]
        row = f"  {x_off:+.3f}m  {r_dict['pass']:3d}/{r_dict['total']:3d}"
        for r in sorted_radii:
            p = r_dict["by_radius"].get(r, 0)
            t = r_dict["total_by_radius"].get(r, 0)
            cell = f"{p}/{t}"
            row += f"  {cell:>{col-2}}"
        print(row)
    print()

    # 儲存結果
    x_tag = "_".join(f"{int(x * 100):+04d}" for x in args.x_offsets)
    tag   = f"ws_minus{int(args.ws_offset * 100):03d}_x{x_tag}"
    out    = os.path.join(VIEWPOINTS_DIR, f"x_offset_scan_{tag}.json")
    latest = os.path.join(VIEWPOINTS_DIR, "x_offset_scan_latest.json")

    payload = {
        "ws_offset_m": args.ws_offset,
        "x_offsets": args.x_offsets,
        "radii": radii,
        "results": {
            str(x): {
                "pass": results[x]["pass"],
                "total": results[x]["total"],
                "by_radius": results[x]["by_radius"],
                "total_by_radius": {str(r): v for r, v in results[x]["total_by_radius"].items()},
            }
            for x in args.x_offsets
        },
    }
    os.makedirs(VIEWPOINTS_DIR, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    shutil.copy2(out, latest)
    print(f"結果已存至: {out}")
    print(f"最新結果:   {latest}")


if __name__ == "__main__":
    main()
