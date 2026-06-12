"""
A-4: 對所有選定視角間轉換規劃繞球路徑（使用 MoveIt），存入 planned_paths.json。

工作空間為球體，半徑 = 拍攝半球半徑 - ws_offset（預設 0.2 m）。
多半徑模式下每段路徑依目標視角的半徑自動調整工作空間球體大小。

使用方式：
  # 先啟動 planning bridge（需 ROS2 + MoveIt 環境）
  ros2 launch ur5e_2f140_planning planning_bridge_launch.py

  # 單半徑（預設）
  /usr/bin/python3 plan_viewpoint_paths.py --vel-scale 0.2 --acc-scale 0.2

  # 多半徑
  /usr/bin/python3 plan_viewpoint_paths.py --multi --vel-scale 0.2 --acc-scale 0.2

  # 調整工作空間偏移
  /usr/bin/python3 plan_viewpoint_paths.py --multi --ws-offset 0.15

單半徑：  selected_viewpoints.json        → planned_paths_ws{offset}.json
多半徑：  selected_viewpoints_multi.json  → planned_paths_multi_ws{offset}.json
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
VIEWPOINTS_DIR = os.path.join(REPO_ROOT, "data", "viewpoints")

_MULTI_MODE = "--multi" in " ".join(sys.argv[1:])


def _make_selected_path(ws_offset_m, x_offset_m):
    """根據 x_offset 定位 A-3 的具名輸出，避免 _latest 被其他 x_offset 覆蓋。"""
    x_tag  = f"_x{int(x_offset_m * 100):+04d}" if x_offset_m != 0.0 else ""
    base   = "selected_viewpoints_multi" if _MULTI_MODE else "selected_viewpoints"
    named  = os.path.join(VIEWPOINTS_DIR, f"{base}{x_tag}.json")
    latest = os.path.join(VIEWPOINTS_DIR, f"{base}_latest.json")
    if os.path.exists(named):
        return named
    return latest  # fallback：具名檔不存在時退回 _latest

ROBOT_BASE_M = [-0.4, 0.0, 0.0]
OBJECT_CENTER_M = [0.0, 0.0, 0.0]
HEMISPHERE_RADIUS_M = 0.65  # 單半徑模式 fallback

# 工作空間球體：半徑 = 拍攝半球半徑 - WORKSPACE_SPHERE_OFFSET_M
WORKSPACE_SPHERE_OFFSET_M = 0.2

def _object_center_base_link(x_offset_m=0.0):
    return [
        OBJECT_CENTER_M[0] + x_offset_m - ROBOT_BASE_M[0],
        OBJECT_CENTER_M[1] - ROBOT_BASE_M[1],
        OBJECT_CENTER_M[2] - ROBOT_BASE_M[2],
    ]

HOME_DEG = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]
HOME_RAD = [math.radians(d) for d in HOME_DEG]

RESULT_WAIT_SEC = 90.0


def _make_paths(ws_offset_m, x_offset_m=0.0):
    x_tag  = f"_x{int(x_offset_m * 100):+04d}" if x_offset_m != 0.0 else ""
    tag    = f"ws_minus{int(ws_offset_m * 100):03d}{x_tag}"
    base   = "planned_paths_multi" if _MULTI_MODE else "planned_paths"
    named  = os.path.join(VIEWPOINTS_DIR, f"{base}_{tag}.json")
    latest = os.path.join(VIEWPOINTS_DIR, f"{base}_latest.json")
    return named, latest



def deg_to_rad(deg_list):
    return [math.radians(d) for d in deg_list]


def normalize_joints(target_rad, start_rad):
    result = []
    for t, s in zip(target_rad, start_rad):
        diff = (t - s + math.pi) % (2 * math.pi) - math.pi
        result.append(s + diff)
    return result


def joint_distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def nearest_neighbor_order(viewpoints, start_rad):
    remaining = list(viewpoints)
    ordered = []
    current = start_rad
    while remaining:
        best_i = min(range(len(remaining)),
                     key=lambda i: joint_distance(current, remaining[i]["joint_rad"]))
        ordered.append(remaining.pop(best_i))
        current = ordered[-1]["joint_rad"]
    return ordered


def load_selected_viewpoints(path=None):
    with open(path or SELECTED_VIEWPOINTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    selected = data.get("selected", [])
    if not selected:
        raise ValueError("selected_viewpoints.json 無 selected 欄位")
    result = []
    for i, vp in enumerate(selected):
        radius_m = (vp.get("radius_m")
                    or vp.get("meta", {}).get("radius_m")
                    or HEMISPHERE_RADIUS_M)
        result.append({
            "id": i + 1,
            "joint_rad": deg_to_rad(vp["joint_deg"]),
            "radius_m": radius_m,
        })
    return result


JOINT_NAMES = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]


def publish_joint_state(node, js_pub, joints_rad):
    import rclpy
    from sensor_msgs.msg import JointState
    msg = JointState()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.name = JOINT_NAMES
    msg.position = [float(v) for v in joints_rad]
    js_pub.publish(msg)
    for _ in range(5):
        rclpy.spin_once(node, timeout_sec=0.05)


def plan_segment(node, pub, js_pub, sub_result, from_id, from_rad, to_id, to_rad,
                 vel_scale, acc_scale, collision_objects):
    import rclpy
    from std_msgs.msg import String

    publish_joint_state(node, js_pub, from_rad)

    plan_id = f"{from_id}_to_{to_id}"
    req = {
        "id": plan_id,
        "start_joints": list(from_rad),
        "target_joints": list(to_rad),
        "collision_objects": collision_objects,
        "velocity_scaling": vel_scale,
        "acceleration_scaling": acc_scale,
    }
    msg = String()
    msg.data = json.dumps(req)
    pub.publish(msg)
    node.get_logger().info(f"  規劃路段: {from_id} → {to_id}")

    deadline = time.time() + RESULT_WAIT_SEC
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        result = sub_result.get_latest()
        if result and result.get("id") == plan_id:
            if result.get("success"):
                wps = result.get("waypoints", [])
                node.get_logger().info(f"    成功：{len(wps)} 個 waypoint")
                return wps, True
            else:
                node.get_logger().warn(f"    失敗: {result.get('error')}")
                return [], False
    node.get_logger().warn(f"    超時: {plan_id}")
    return [], False


def plan_segment_via_home(node, pub, js_pub, sub_result, from_id, from_rad, to_id, to_rad,
                          vel_scale, acc_scale, collision_objects):
    node.get_logger().info(f"  → 繞道 HOME: {from_id} → home → {to_id}")
    home_norm = normalize_joints(HOME_RAD, from_rad)
    wps1, ok1 = plan_segment(node, pub, js_pub, sub_result,
                             from_id, from_rad, "home", home_norm,
                             vel_scale, acc_scale, collision_objects)
    if not ok1:
        raise RuntimeError(f"繞道規劃失敗: {from_id} → home")

    actual_home = wps1[-1]["positions"] if isinstance(wps1[0], dict) else wps1[-1]
    to_norm = normalize_joints(to_rad, actual_home)
    wps2, ok2 = plan_segment(node, pub, js_pub, sub_result,
                             "home", actual_home, to_id, to_norm,
                             vel_scale, acc_scale, collision_objects)
    if not ok2:
        raise RuntimeError(f"繞道規劃失敗: home → {to_id}")

    return wps1 + wps2


class LatestResult:
    def __init__(self):
        self._data = None

    def callback(self, msg):
        try:
            self._data = json.loads(msg.data)
        except Exception:
            pass

    def get_latest(self):
        d = self._data
        self._data = None
        return d


def main():
    parser = argparse.ArgumentParser(description="A-4: 規劃視角間路徑")
    parser.add_argument("--vel-scale", type=float, default=0.2,
                        help="速度 scaling 係數，0.0~1.0（預設 0.2）")
    parser.add_argument("--acc-scale", type=float, default=0.2,
                        help="加速度 scaling 係數，0.0~1.0（預設 0.2）")
    parser.add_argument("--multi", action="store_true",
                        help="多半徑模式：讀取 selected_viewpoints_multi.json")
    parser.add_argument("--ws-offset", type=float, default=WORKSPACE_SPHERE_OFFSET_M,
                        help=f"sphere_r = cam_r - ws_offset（預設 {WORKSPACE_SPHERE_OFFSET_M}）")
    parser.add_argument("--x-offset", type=float, default=0.0,
                        help="物體中心 x 軸偏移（m），拍攝球體與工作球體同步移動（預設 0.0）")
    args = parser.parse_args()

    ws_offset  = args.ws_offset
    x_offset   = args.x_offset
    planned_paths_path, planned_paths_latest = _make_paths(ws_offset, x_offset)
    selected_viewpoints_path = _make_selected_path(ws_offset, x_offset)

    if _MULTI_MODE:
        print("[A-4] 多半徑模式")
    print(f"[A-4] 讀取視角: {os.path.basename(selected_viewpoints_path)}")

    if not (0.0 < args.vel_scale <= 1.0) or not (0.0 < args.acc_scale <= 1.0):
        print("ERROR: --vel-scale 與 --acc-scale 需介於 0.0（不含）到 1.0 之間")
        sys.exit(1)

    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
    except ImportError:
        print("ERROR: 需要 ROS2 環境，請先執行:")
        print("  source /opt/ros/jazzy/setup.bash")
        print("  source ~/webots_program/ros2_ws/install/setup.bash")
        sys.exit(1)

    viewpoints = load_selected_viewpoints(selected_viewpoints_path)
    viewpoints = nearest_neighbor_order(viewpoints, HOME_RAD)
    print(f"載入 {len(viewpoints)} 個視角（nearest-neighbor 排序後）")
    print("  排序: home →", " → ".join(str(vp["id"]) for vp in viewpoints), "→ home")
    print(f"速度 scaling={args.vel_scale}  加速度 scaling={args.acc_scale}")
    print(f"ws_offset={ws_offset} m  （sphere_r = cam_r - {ws_offset}）")
    print(f"x 軸偏移         = {x_offset:+.3f} m")
    for vp in viewpoints:
        sphere_r = vp["radius_m"] - ws_offset
        print(f"  視角 {vp['id']:2d}: cam_r={vp['radius_m']:.2f}m → sphere_r={sphere_r:.2f}m")
    print(f"輸出: {os.path.basename(planned_paths_path)}")

    from sensor_msgs.msg import JointState

    rclpy.init()
    node = rclpy.create_node("plan_viewpoint_paths")
    pub = node.create_publisher(String, "/ur5e/plan_request", 10)
    js_pub = node.create_publisher(JointState, "/joint_states", 10)

    result_store = LatestResult()
    node.create_subscription(String, "/ur5e/plan_result", result_store.callback, 10)

    print("等待 planning bridge...")
    deadline = time.time() + 3.0
    while time.time() < deadline:
        publish_joint_state(node, js_pub, HOME_RAD)
    rclpy.spin_once(node, timeout_sec=1.0)

    paths = []
    # all_joints 帶上 radius_m，home 用 None
    all_joints = [("home", HOME_RAD, None)] + [
        (vp["id"], vp["joint_rad"], vp["radius_m"]) for vp in viewpoints
    ]

    try:
        for i in range(len(all_joints) - 1):
            from_id, from_rad, _ = all_joints[i]
            to_id, to_rad, to_radius = all_joints[i + 1]
            # 工作空間約束依目標視角半徑決定
            col_objects = [{
                    "id": "ycb_workspace_sphere",
                    "shape": "sphere",
                    "size": [(to_radius - ws_offset) * 2],
                    "position": _object_center_base_link(x_offset),
                }]
            to_rad_norm = normalize_joints(to_rad, from_rad)
            waypoints, ok = plan_segment(node, pub, js_pub, result_store,
                                         from_id, from_rad, to_id, to_rad_norm,
                                         args.vel_scale, args.acc_scale, col_objects)
            if not ok:
                waypoints = plan_segment_via_home(node, pub, js_pub, result_store,
                                                  from_id, from_rad, to_id, to_rad_norm,
                                                  args.vel_scale, args.acc_scale, col_objects)
            paths.append({
                "from_id": from_id,
                "to_id": to_id,
                "waypoints": waypoints,
            })
            last_wp = waypoints[-1]
            all_joints[i + 1] = (to_id,
                                  last_wp["positions"] if isinstance(last_wp, dict) else last_wp,
                                  to_radius)

        # 最後一段：最後視角回 home，沿用最後視角的工作空間約束
        last_id, last_rad, last_radius = all_joints[-1]
        col_objects = [{
                "id": "ycb_workspace_sphere",
                "shape": "sphere",
                "size": [(last_radius - ws_offset) * 2],
                "position": _object_center_base_link(x_offset),
            }]
        home_norm = normalize_joints(HOME_RAD, last_rad)
        waypoints, ok = plan_segment(node, pub, js_pub, result_store,
                                     last_id, last_rad, "home", home_norm,
                                     args.vel_scale, args.acc_scale, col_objects)
        if not ok:
            raise RuntimeError(f"最後段規劃失敗: {last_id} → home")
        paths.append({"from_id": last_id, "to_id": "home", "waypoints": waypoints})

    except (RuntimeError, TimeoutError) as e:
        print(f"ERROR: {e}")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "object_center_base_link_m": _object_center_base_link(x_offset),
            "viewpoint_count": len(viewpoints),
            "path_count": len(paths),
            "velocity_scaling": args.vel_scale,
            "acceleration_scaling": args.acc_scale,
            "workspace_sphere_offset_m": ws_offset,
            "workspace_sphere_max_r_m": max(vp["radius_m"] for vp in viewpoints) - ws_offset,
            "x_offset_m": x_offset,
        },
        "home_joints_rad": HOME_RAD,
        "paths": paths,
    }

    os.makedirs(VIEWPOINTS_DIR, exist_ok=True)
    with open(planned_paths_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    shutil.copy2(planned_paths_path, planned_paths_latest)

    print(f"\n完成：{len(paths)} 條路徑")
    print(f"輸出: {planned_paths_path}")
    print(f"最新: {planned_paths_latest}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
