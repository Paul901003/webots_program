import json
import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from geometry_msgs.msg import Pose
from moveit_msgs.msg import (
    CollisionObject,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    PlanningScene,
    RobotState,
)
from moveit_msgs.srv import ApplyPlanningScene, GetMotionPlan
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# All robot joints for a fully specified start state (gripper stays at 0 = open)
ALL_JOINT_NAMES = ARM_JOINT_NAMES + [
    "finger_joint",
    "left_inner_knuckle_joint",
    "left_inner_finger_joint",
    "right_outer_knuckle_joint",
    "right_inner_knuckle_joint",
    "right_inner_finger_joint",
]


class PlanningBridge(Node):
    def __init__(self):
        super().__init__("planning_bridge")

        cb = ReentrantCallbackGroup()

        self._plan_client = self.create_client(
            GetMotionPlan, "/plan_kinematic_path", callback_group=cb
        )
        self._scene_client = self.create_client(
            ApplyPlanningScene, "/apply_planning_scene", callback_group=cb
        )
        self._result_pub = self.create_publisher(String, "/ur5e/plan_result", 10)
        self.create_subscription(
            String, "/ur5e/plan_request", self._on_request, 10, callback_group=cb
        )

        self.get_logger().info("Planning bridge ready.")

    def _wait_future(self, future, timeout_sec=15.0):
        deadline = time.time() + timeout_sec
        while not future.done():
            if time.time() > deadline:
                return None
            time.sleep(0.05)
        return future.result()

    def _on_request(self, msg: String):
        try:
            req_data = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"JSON parse error: {e}")
            return

        plan_id = req_data.get("id", "")
        start_joints = req_data.get("start_joints", [])
        target_joints = req_data.get("target_joints", [])
        collision_objects = req_data.get("collision_objects", [])
        vel_scale = float(req_data.get("velocity_scaling", 0.2))
        acc_scale = float(req_data.get("acceleration_scaling", 0.2))
        num_attempts = int(req_data.get("num_planning_attempts", 30))
        allowed_time = float(req_data.get("allowed_planning_time", 45.0))

        self.get_logger().info(f"Plan request: {plan_id} (vel={vel_scale} acc={acc_scale} attempts={num_attempts} time={allowed_time})")

        self._apply_scene(collision_objects)
        if collision_objects:
            time.sleep(0.5)
        result = self._plan(start_joints, target_joints, vel_scale, acc_scale, num_attempts, allowed_time)
        result["id"] = plan_id

        out = String()
        out.data = json.dumps(result)
        self._result_pub.publish(out)
        self.get_logger().info(
            f"Plan result: {plan_id} success={result['success']} "
            f"waypoints={len(result.get('waypoints', []))}"
        )

    def _apply_scene(self, objects: list):
        if not objects:
            return
        if not self._scene_client.service_is_ready():
            self.get_logger().warn("ApplyPlanningScene not ready, skipping collision objects")
            return

        scene = PlanningScene()
        scene.is_diff = True

        for obj_data in objects:
            co = CollisionObject()
            co.header.frame_id = "world"
            co.id = obj_data["id"]
            co.operation = CollisionObject.ADD

            prim = SolidPrimitive()
            size = obj_data.get("size", [0.1, 0.1, 0.1])
            shape = obj_data.get("shape", "box")

            if shape == "cylinder":
                # plan_viewpoint_paths gives size=[d, d, height]
                # SolidPrimitive.CYLINDER dimensions: [height, radius]
                prim.type = SolidPrimitive.CYLINDER
                prim.dimensions = [float(size[2]), float(size[0]) / 2.0]
            elif shape == "sphere":
                prim.type = SolidPrimitive.SPHERE
                prim.dimensions = [float(size[0]) / 2.0]
            else:
                prim.type = SolidPrimitive.BOX
                prim.dimensions = [float(v) for v in size[:3]]

            pos = obj_data.get("position", [0.0, 0.0, 0.0])
            pose = Pose()
            pose.position.x = float(pos[0])
            pose.position.y = float(pos[1])
            pose.position.z = float(pos[2])
            pose.orientation.w = 1.0

            co.primitives = [prim]
            co.primitive_poses = [pose]
            scene.world.collision_objects.append(co)

        req = ApplyPlanningScene.Request()
        req.scene = scene
        future = self._scene_client.call_async(req)
        self._wait_future(future, timeout_sec=5.0)

    def _plan(self, start_joints: list, target_joints: list,
              vel_scale: float = 0.2, acc_scale: float = 0.2,
              num_attempts: int = 30, allowed_time: float = 45.0) -> dict:
        if not self._plan_client.service_is_ready():
            self.get_logger().warn("Waiting for /plan_kinematic_path...")
            if not self._plan_client.wait_for_service(timeout_sec=10.0):
                return {"success": False, "waypoints": [], "error": "Plan service unavailable"}

        mpr = MotionPlanRequest()
        mpr.group_name = "ur5e_arm"
        mpr.planner_id = ""
        mpr.num_planning_attempts = num_attempts
        mpr.allowed_planning_time = allowed_time
        mpr.max_velocity_scaling_factor = vel_scale
        mpr.max_acceleration_scaling_factor = acc_scale

        if start_joints:
            # Explicit start state provided (e.g. from plan_viewpoint_paths.py)
            ss = RobotState()
            ss.is_diff = False
            ss.joint_state.name = list(ALL_JOINT_NAMES)
            arm_vals = [float(v) for v in start_joints]
            gripper_vals = [0.0] * (len(ALL_JOINT_NAMES) - len(ARM_JOINT_NAMES))
            ss.joint_state.position = arm_vals + gripper_vals
            mpr.start_state = ss
        # else: use current state from /joint_states (published by caller)

        c = Constraints()
        for name, val in zip(ARM_JOINT_NAMES, target_joints):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(val)
            jc.tolerance_above = 0.001
            jc.tolerance_below = 0.001
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        mpr.goal_constraints = [c]

        req = GetMotionPlan.Request()
        req.motion_plan_request = mpr
        future = self._plan_client.call_async(req)
        result = self._wait_future(future, timeout_sec=15.0)

        if result is None:
            return {"success": False, "waypoints": [], "error": "Planning timeout"}

        resp = result.motion_plan_response
        err_val = resp.error_code.val
        if err_val != 1:  # MoveItErrorCodes.SUCCESS = 1
            return {"success": False, "waypoints": [], "error": f"MoveIt error code: {err_val}"}

        waypoints = [
            {
                "positions": list(pt.positions),
                "velocities": list(pt.velocities),
                "accelerations": list(pt.accelerations),
                "time_from_start": pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9,
            }
            for pt in resp.trajectory.joint_trajectory.points
        ]
        return {"success": True, "waypoints": waypoints, "error": ""}


def main():
    rclpy.init()
    node = PlanningBridge()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
