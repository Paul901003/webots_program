"""
A-2b: 以工作空間球體約束驗證視角可達性，輸出通過的視角作為 selected_viewpoints。

工作空間球體半徑 = 拍攝半球半徑 - ws_offset（預設 0.2 m）。
不需要 Webots，只需 planning bridge 運行中。

前置條件：
  1. 已跑 A-2（ycb_viewpoint_validator_multi.wbt），產生 validated_viewpoints_multi_latest.json
  2. 已啟動 planning bridge：
     ros2 launch ur5e_2f140_planning planning_bridge_launch.py

使用方式：
  # 多半徑（預設 ws_offset=0.2m）
  /usr/bin/python3 validate_workspace_sphere.py --multi

  # 調整偏移量
  /usr/bin/python3 validate_workspace_sphere.py --multi --ws-offset 0.15

輸出：
  data/viewpoints/selected_viewpoints_multi_ws_minus{N}.json  ← 具名
  data/viewpoints/selected_viewpoints_multi_latest.json        ← 最新指標
"""

import argparse
import json
import math
import os
import shutil
import sys
import time

CURRENT_DIR    = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT      = os.path.dirname(os.path.dirname(CURRENT_DIR))
VIEWPOINTS_DIR = os.path.join(REPO_ROOT, "data", "viewpoints")

ROBOT_BASE_M    = [-0.4, 0.0, 0.0]
OBJECT_CENTER_M = [0.0,  0.0, 0.0]
HOME_DEG        = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]
HOME_RAD        = [math.radians(d) for d in HOME_DEG]

def _object_center_base_link(x_offset_m=0.0):
    return [
        OBJECT_CENTER_M[0] + x_offset_m - ROBOT_BASE_M[0],
        OBJECT_CENTER_M[1] - ROBOT_BASE_M[1],
        OBJECT_CENTER_M[2] - ROBOT_BASE_M[2],
    ]

RESULT_WAIT_SEC    = 15.0
NUM_PLAN_ATTEMPTS  = 5
ALLOWED_PLAN_TIME  = 10.0

HEMISPHERE_RADIUS_M = 0.65  # 單半徑 fallback


def build_workspace_sphere(cam_radius_m, ws_offset_m, x_offset_m=0.0):
    sphere_r = cam_radius_m - ws_offset_m
    return [{
        "id": "ycb_workspace_sphere",
        "shape": "sphere",
        "size": [sphere_r * 2],
        "position": _object_center_base_link(x_offset_m),
    }]


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

    plan_id = f"ws_check_{vp_id}"
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
                        help="多半徑模式")
    parser.add_argument("--ws-offset", type=float, default=0.2,
                        help="工作空間球體偏移（m），sphere_r = cam_r - offset（預設 0.2）")
    parser.add_argument("--x-offset", type=float, default=0.0,
                        help="物體中心 x 軸偏移（m），拍攝球體與工作球體同步移動（預設 0.0）")
    args = parser.parse_args()

    ws_offset = args.ws_offset
    x_offset  = args.x_offset

    if args.multi:
        validated_path = os.path.join(VIEWPOINTS_DIR, "validated_viewpoints_multi_latest.json")
        base = "selected_viewpoints_multi"
    else:
        validated_path = os.path.join(VIEWPOINTS_DIR, "validated_viewpoints_latest.json")
        base = "selected_viewpoints"

    x_tag       = f"_x{int(x_offset * 100):+04d}" if x_offset != 0.0 else ""
    tag         = f"ws_minus{int(ws_offset * 100):03d}{x_tag}"
    output_path = os.path.join(VIEWPOINTS_DIR, f"{base}_{tag}.json")
    latest_path = os.path.join(VIEWPOINTS_DIR, f"{base}_latest.json")

    with open(validated_path, encoding="utf-8") as f:
        data = json.load(f)
    validated = data.get("validated", [])
    if not validated:
        print("ERROR: 無已驗證視角，請先跑 A-2")
        sys.exit(1)

    print(f"載入 {len(validated)} 個 A-2 通過視角")
    print(f"工作空間球體偏移: {ws_offset} m  (sphere_r = cam_r - {ws_offset})")
    print(f"x 軸偏移:         {x_offset:+.3f} m  (球體中心 = [{OBJECT_CENTER_M[0]+x_offset:.3f}, 0, 0])")
    print()

    # 顯示各視角對應的球體半徑
    radii_set = {}
    for vp in validated:
        r = vp.get("radius_m") or vp.get("meta", {}).get("radius_m") or HEMISPHERE_RADIUS_M
        radii_set[r] = radii_set.get(r, 0) + 1
    for r in sorted(radii_set):
        print(f"  cam_r={r:.2f}m → sphere_r={r - ws_offset:.2f}m  ({radii_set[r]} 個視角)")
    print()

    try:
        import rclpy
        from std_msgs.msg import String
    except ImportError:
        print("ERROR: 需要 ROS2 環境")
        sys.exit(1)

    rclpy.init()
    node    = rclpy.create_node("validate_workspace_sphere")
    pub     = node.create_publisher(String, "/ur5e/plan_request", 10)
    sub_res = LatestResult()
    node.create_subscription(String, "/ur5e/plan_result", sub_res.callback, 10)

    print("等待 planning bridge...")
    deadline = time.time() + 3.0
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    print()

    passed = []
    failed = []

    for idx, vp in enumerate(validated):
        cam_r    = vp.get("radius_m") or vp.get("meta", {}).get("radius_m") or HEMISPHERE_RADIUS_M
        joint_rad = [math.radians(d) for d in vp["joint_deg"]]
        col_objs  = build_workspace_sphere(cam_r, ws_offset, x_offset)

        ok = plan_home_to_vp(node, pub, sub_res, vp["id"], joint_rad, col_objs)
        status = "✓" if ok else "✗"
        print(f"  [{idx+1:3d}/{len(validated)}] {vp['id']:20s}  cam_r={cam_r:.2f}m  {status}")

        if ok:
            passed.append(vp)
        else:
            failed.append(vp)

    node.destroy_node()
    rclpy.shutdown()

    print()
    print(f"通過: {len(passed)}/{len(validated)}")
    if failed:
        print(f"失敗視角: {[v['id'] for v in failed]}")

    # 依半徑統計
    pass_by_r = {}
    for vp in passed:
        r = vp.get("radius_m") or vp.get("meta", {}).get("radius_m") or HEMISPHERE_RADIUS_M
        pass_by_r[r] = pass_by_r.get(r, 0) + 1
    for r in sorted(pass_by_r):
        total = radii_set.get(r, "?")
        print(f"  cam_r={r:.2f}m: {pass_by_r[r]}/{total} 通過")

    result = {
        "source": validated_path,
        "ws_offset_m": ws_offset,
        "validated_count": len(validated),
        "selected_count": len(passed),
        "failed_ids": [v["id"] for v in failed],
        "selected": passed,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    shutil.copy2(output_path, latest_path)
    print(f"\n輸出: {output_path}")
    print(f"最新: {latest_path}")


if __name__ == "__main__":
    main()
