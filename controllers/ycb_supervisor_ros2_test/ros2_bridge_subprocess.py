#!/usr/bin/env python3.12
"""
ros2_bridge_subprocess.py

用 python3.12 執行，由 ycb_supervisor_ros2_test 以 subprocess 啟動。
從 stdin 讀取 JSON 請求，透過 ROS2 向 planning_bridge 請求 MoveIt 規劃，
將結果以 JSON 寫回 stdout。

stdin 格式：
  {"current_joints": [6 floats], "target_joints": [6 floats],
   "collision_objects": [...]}

stdout 格式：
  {"success": bool, "error": str, "waypoints": [[6 floats], ...]}
"""

import json
import sys
import time
import uuid

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
PLAN_TIMEOUT_SEC = 20.0


class BridgeNode(Node):
    def __init__(self):
        super().__init__("webots_ros2_bridge")
        self._joint_pub = self.create_publisher(JointState, "/webots/joint_states", 10)
        self._req_pub = self.create_publisher(String, "/ur5e/plan_request", 10)
        self._result: dict | None = None
        self._pending_id: str | None = None
        self.create_subscription(String, "/ur5e/plan_result", self._on_result, 10)

    def _on_result(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if data.get("id") == self._pending_id:
            self._result = data

    def publish_joint_state(self, joints: list):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = [float(v) for v in joints]
        self._joint_pub.publish(msg)
        rclpy.spin_once(self, timeout_sec=0.05)

    def is_plan_result_subscriber_ready(self) -> bool:
        return self.count_publishers("/ur5e/plan_result") > 0

    def request_plan(self, current_joints: list, target_joints: list, collision_objects: list) -> dict:
        if not self.is_plan_result_subscriber_ready():
            return {"success": False, "error": "planning_bridge not connected", "waypoints": []}

        self.publish_joint_state(current_joints)

        request_id = str(uuid.uuid4())[:8]
        self._pending_id = request_id
        self._result = None

        payload = {
            "id": request_id,
            "start_joints": [float(v) for v in current_joints],
            "target_joints": [float(v) for v in target_joints],
            "collision_objects": collision_objects,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self._req_pub.publish(msg)

        deadline = time.time() + PLAN_TIMEOUT_SEC
        while self._result is None and time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

        if self._result is None:
            return {"success": False, "error": "timeout", "waypoints": []}
        return self._result


def _spin_safe(node, timeout_sec=0.1):
    """spin_once，捕捉 rclpy context 失效時的例外。回傳 False 表示應停止。"""
    try:
        rclpy.spin_once(node, timeout_sec=timeout_sec)
        return True
    except Exception:
        return False


def _wait_for_move_group(node: BridgeNode, timeout_sec: float = 60.0):
    """等待 planning_bridge 和 move_group action server 都就緒。"""
    import time as _time
    deadline = _time.time() + timeout_sec

    print("[Bridge] 等待 planning_bridge 就緒...", file=sys.stderr, flush=True)
    for _ in range(20):
        if not _spin_safe(node, timeout_sec=0.1):
            print("[Bridge] rclpy context 無效，停止等待", file=sys.stderr, flush=True)
            return

    while _time.time() < deadline:
        if not rclpy.ok():
            print("[Bridge] rclpy 已關閉，停止等待", file=sys.stderr, flush=True)
            return
        if node.is_plan_result_subscriber_ready():
            print("[Bridge] planning_bridge 已連線", file=sys.stderr, flush=True)
            break
        if not _spin_safe(node, timeout_sec=0.2):
            return
    else:
        print("[Bridge] 警告：planning_bridge 未在時限內就緒，繼續啟動", file=sys.stderr, flush=True)
        return

    print("[Bridge] 等待 move_group action server 就緒...", file=sys.stderr, flush=True)
    while _time.time() < deadline:
        if not rclpy.ok():
            return
        if node.count_publishers("/move_group/_action/status") > 0:
            print("[Bridge] move_group action server 已就緒", file=sys.stderr, flush=True)
            break
        if not _spin_safe(node, timeout_sec=0.2):
            return
    else:
        print("[Bridge] 警告：move_group 未在時限內就緒，繼續啟動", file=sys.stderr, flush=True)

    for _ in range(10):
        if not _spin_safe(node, timeout_sec=0.05):
            return


def main():
    try:
        rclpy.init()
        node = BridgeNode()
    except Exception as e:
        print(f"INIT_FAILED: {e}", flush=True)
        return

    import threading as _threading

    # Event：_wait_for_move_group 結束後設定，確保不與 request_plan 競爭 spin_once
    _ready_event = _threading.Event()

    def _wait_and_signal(node):
        _wait_for_move_group(node)
        _ready_event.set()

    _threading.Thread(target=_wait_and_signal, args=(node,), daemon=True).start()

    # 最多等 0.5s 讓 subscription 建立，之後立即回報 READY
    for _ in range(10):
        if not _spin_safe(node, timeout_sec=0.05):
            break

    print("READY", flush=True)  # 通知 supervisor 已啟動完成

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            print(json.dumps({"success": False, "error": f"JSON parse: {e}", "waypoints": []}), flush=True)
            continue

        # 等待 _wait_for_move_group 執行緒完成，避免與 spin_once 競爭
        if not _ready_event.is_set():
            print("[Bridge] 等待 move_group 就緒...", file=sys.stderr, flush=True)
            _ready_event.wait(timeout=120.0)

        try:
            result = node.request_plan(
                req["current_joints"],
                req["target_joints"],
                req["collision_objects"],
            )
        except Exception as e:
            result = {"success": False, "error": str(e), "waypoints": []}

        print(json.dumps(result), flush=True)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
