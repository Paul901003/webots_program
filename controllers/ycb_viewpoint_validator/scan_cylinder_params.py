"""
掃描不同圓柱（半徑×高度）對視角可達率的影響。

前置條件：
  1. 已用 ycb_viewpoint_validator_no_cylinder.wbt 跑過 A-2，
     產生 validated_viewpoints.json（或 --multi 版本）
  2. 已啟動 planning bridge：
     ros2 launch ur5e_2f140_planning planning_bridge_launch.py

使用方式：
  # 單半徑（預設）
  /usr/bin/python3 scan_cylinder_params.py

  # 多半徑
  /usr/bin/python3 scan_cylinder_params.py --multi

  # 自訂掃描範圍
  /usr/bin/python3 scan_cylinder_params.py \\
      --radii 0.2 0.25 0.3 0.35 0.4 \\
      --heights 0.25 0.3 0.35 0.4 0.45
"""

import argparse
import json
import math
import os
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(CURRENT_DIR))
VIEWPOINTS_DIR = os.path.join(REPO_ROOT, "data", "viewpoints")

ROBOT_BASE_M    = [-0.4, 0.0, 0.0]
OBJECT_CENTER_M = [0.0, 0.0, 0.0]
HOME_DEG        = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]
HOME_RAD        = [math.radians(d) for d in HOME_DEG]

RESULT_WAIT_SEC = 15.0  # 掃描用短 timeout，快速判斷可行性
JOINT_NAMES     = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


def build_collision_object(radius_m, height_m, x_offset_m=0.0):
    """x_offset_m：圓柱中心在 world x 軸的偏移量（正 = 遠離手臂底座）。"""
    return [{
        "id": "ycb_workspace_cylinder",
        "shape": "cylinder",
        "size": [radius_m * 2, radius_m * 2, height_m],
        "position": [
            OBJECT_CENTER_M[0] - ROBOT_BASE_M[0] + x_offset_m,
            OBJECT_CENTER_M[1] - ROBOT_BASE_M[1],
            OBJECT_CENTER_M[2] - ROBOT_BASE_M[2] + height_m / 2.0,
        ],
    }]


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
        "num_planning_attempts": 5,   # 掃描用：快速判斷可行性
        "allowed_planning_time": 10.0,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--multi", action="store_true",
                        help="讀取 validated_viewpoints_multi.json")
    parser.add_argument("--radii", type=float, nargs="+",
                        default=[0.2, 0.25, 0.3, 0.35, 0.4],
                        help="測試的圓柱半徑清單（m）")
    parser.add_argument("--heights", type=float, nargs="+",
                        default=[0.25, 0.3, 0.35, 0.4, 0.45],
                        help="測試的圓柱高度清單（m）")
    parser.add_argument("--x-offsets", type=float, nargs="+",
                        default=[0.0],
                        help="圓柱中心 x 軸偏移清單（m，正=遠離手臂，預設 0）")
    args = parser.parse_args()

    if args.multi:
        validated_path = os.path.join(VIEWPOINTS_DIR, "validated_viewpoints_multi_latest.json")
    else:
        validated_path = os.path.join(VIEWPOINTS_DIR, "validated_viewpoints_latest.json")

    with open(validated_path, encoding="utf-8") as f:
        data = json.load(f)
    validated = data.get("validated", [])
    if not validated:
        print("ERROR: 無已驗證視角，請先跑 A-2（no-cylinder 模式）")
        sys.exit(1)

    viewpoints = [
        {"id": v["id"], "joint_rad": [math.radians(d) for d in v["joint_deg"]]}
        for v in validated
    ]
    print(f"載入 {len(viewpoints)} 個已驗證視角")
    print(f"掃描半徑: {args.radii}")
    print(f"掃描高度: {args.heights}")
    print(f"掃描 x 偏移: {args.x_offsets}")
    print()

    try:
        import rclpy
        from std_msgs.msg import String
    except ImportError:
        print("ERROR: 需要 ROS2 環境")
        sys.exit(1)

    rclpy.init()
    node = rclpy.create_node("scan_cylinder_params")
    pub = node.create_publisher(String, "/ur5e/plan_request", 10)
    result_store = LatestResult()
    node.create_subscription(String, "/ur5e/plan_result", result_store.callback, 10)

    print("等待 planning bridge...")
    deadline = time.time() + 3.0
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    # 結果矩陣
    results = {}
    total = len(args.radii) * len(args.heights) * len(args.x_offsets) * len(viewpoints)
    done = 0

    for x in args.x_offsets:
        for r in args.radii:
            for h in args.heights:
                col_objects = build_collision_object(r, h, x)
                ok = 0
                for vp in viewpoints:
                    success = plan_home_to_vp(node, pub, result_store,
                                              vp["id"], vp["joint_rad"], col_objects)
                    if success:
                        ok += 1
                    done += 1
                    if done % 10 == 0:
                        print(f"  進度: {done}/{total}  (x={x:+.2f} r={r} h={h} 通過={ok})")
                results[(x, r, h)] = ok
                print(f"  x={x:+.2f}m  r={r}m  h={h}m  →  {ok}/{len(viewpoints)} 通過")

    node.destroy_node()
    rclpy.shutdown()

    # 輸出表格（每個 x 偏移各一張）
    print()
    for x in args.x_offsets:
        print(f"{'=' * 60}")
        print(f"x 偏移 = {x:+.2f}m  （通過視角數）")
        print(f"{'=' * 60}")
        header = "半徑＼高度   " + "  ".join(f"{h:.2f}" for h in args.heights)
        print(header)
        print("-" * len(header))
        for r in args.radii:
            row = f"  {r:.2f}m     " + "  ".join(f"{results[(x,r,h)]:4d}" for h in args.heights)
            print(row)
        print()

    # 儲存結果（參數化命名，不同參數不覆蓋）
    r_tag  = "_".join(f"{int(r*100):03d}" for r in args.radii)
    h_tag  = "_".join(f"{int(h*100):03d}" for h in args.heights)
    x_tag  = "_".join(f"{int(x*100):+04d}" for x in args.x_offsets)
    tag    = f"r{r_tag}_h{h_tag}_x{x_tag}"
    out_path        = os.path.join(VIEWPOINTS_DIR, f"cylinder_scan_{tag}.json")
    out_path_latest = os.path.join(VIEWPOINTS_DIR, "cylinder_scan_latest.json")

    payload = {
        "radii": args.radii,
        "heights": args.heights,
        "x_offsets": args.x_offsets,
        "viewpoint_count": len(viewpoints),
        "results": {f"x{x}_r{r}_h{h}": results[(x, r, h)]
                    for x in args.x_offsets
                    for r in args.radii
                    for h in args.heights},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    import shutil
    shutil.copy2(out_path, out_path_latest)
    print(f"結果已存至: {out_path}")
    print(f"最新結果:   {out_path_latest}")


if __name__ == "__main__":
    main()
