from controller import Supervisor
import importlib.util
import json
import math
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_CONTROLLER_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "ycb_supervisor")
if SOURCE_CONTROLLER_DIR not in sys.path:
    sys.path.insert(0, SOURCE_CONTROLLER_DIR)

from config import (  # noqa: E402
    ARM_MOTOR_VELOCITY_RAD_PER_SEC,
    ARM_SETTLE_TIME_BUFFER_SEC,
    ARM_SETTLE_TIME_SEC,
    POST_ARRIVAL_PAUSE_SEC,
)

ARM_COMMAND_EMITTER = "arm_command_emitter"
ARM_STATUS_RECEIVER = "arm_status_receiver"
UR5E_DEF = "UR5E"
FALLBACK_VIEW_SEQUENCE = (1, 2, 3, 4)
HOME_POSE_RAD = [0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0]


def get_arm_controller_name(supervisor: Supervisor):
    ur5e_node = supervisor.getFromDef(UR5E_DEF)
    if ur5e_node is None:
        print(f"[Supervisor] 找不到 DEF {UR5E_DEF}，使用 fallback view sequence")
        return None
    controller_field = ur5e_node.getField("controller")
    if controller_field is None:
        print("[Supervisor] 找不到 UR5E controller field，使用 fallback view sequence")
        return None
    return controller_field.getSFString()


def load_camera_poses_from_arm_controller(controller_name: str | None):
    if not controller_name:
        return {}, FALLBACK_VIEW_SEQUENCE
    controller_path = os.path.join(
        os.path.dirname(CURRENT_DIR),
        controller_name,
        f"{controller_name}.py",
    )
    try:
        spec = importlib.util.spec_from_file_location(controller_name, controller_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        camera_poses = getattr(module, "CAMERA_POSES", {})
        sequence = tuple(sorted(int(index) for index in camera_poses.keys()))
        return camera_poses, sequence or FALLBACK_VIEW_SEQUENCE
    except Exception as error:
        print(f"[Supervisor] 無法讀取 {controller_path} 的 CAMERA_POSES: {error}")
        return {}, FALLBACK_VIEW_SEQUENCE


def pose_joints_rad(camera_poses: dict, view_index: int):
    pose = camera_poses.get(view_index)
    if not isinstance(pose, dict):
        return None
    joint_deg = pose.get("joint_deg")
    if not isinstance(joint_deg, list) or len(joint_deg) != 6:
        return None
    return [math.radians(float(value)) for value in joint_deg]


def estimate_settle_time(current_joints_rad, target_joints_rad):
    if current_joints_rad is None or target_joints_rad is None:
        return ARM_SETTLE_TIME_SEC
    max_delta = max(abs(target - current) for target, current in zip(target_joints_rad, current_joints_rad))
    motion_time = max_delta / max(ARM_MOTOR_VELOCITY_RAD_PER_SEC, 1e-6)
    return max(ARM_SETTLE_TIME_SEC, motion_time + ARM_SETTLE_TIME_BUFFER_SEC)


def wait_seconds(supervisor: Supervisor, timestep: int, seconds: float):
    steps = max(0, int(seconds * 1000 / max(1, timestep)))
    for _ in range(steps):
        if supervisor.step(timestep) == -1:
            return False
    return True


def clear_receiver(receiver):
    if receiver is None:
        return
    while receiver.getQueueLength() > 0:
        receiver.nextPacket()


def wait_for_arm_arrival(supervisor: Supervisor, timestep: int, receiver, command_id: str, timeout_sec: float):
    if receiver is None:
        return wait_seconds(supervisor, timestep, timeout_sec)
    start_time = supervisor.getTime()
    while supervisor.getTime() - start_time <= timeout_sec:
        if supervisor.step(timestep) == -1:
            return False
        while receiver.getQueueLength() > 0:
            message = receiver.getString()
            receiver.nextPacket()
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                continue
            if (
                data.get("status") == "arrived"
                and str(data.get("command_id")) == str(command_id)
            ):
                print(
                    "[Supervisor] Arm arrived "
                    f"(max_error={float(data.get('max_error_rad', 0.0)):.4f} rad)"
                )
                return True
    print(f"[Supervisor] Arm arrival timeout for command {command_id}; stopping sequence.")
    return False


def send_arm_pose_command(emitter, view_index: int):
    if emitter is None:
        print("[Supervisor] 找不到手臂 emitter，無法送出移動指令")
        return False
    emitter.send(str(view_index).encode("utf-8"))
    return True


def main():
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())
    arm_emitter = supervisor.getDevice(ARM_COMMAND_EMITTER)
    arm_status_receiver = supervisor.getDevice(ARM_STATUS_RECEIVER)
    if arm_status_receiver is not None:
        arm_status_receiver.enable(timestep)
    else:
        print(f"[Supervisor] 找不到 {ARM_STATUS_RECEIVER}，改用時間等待。")
    arm_controller_name = get_arm_controller_name(supervisor)
    camera_poses, view_sequence = load_camera_poses_from_arm_controller(arm_controller_name)

    print(f"[Supervisor] Arm controller: {arm_controller_name}")
    print(f"[Supervisor] View sequence: {view_sequence}")
    current_joints_rad = HOME_POSE_RAD[:]
    for view_index in view_sequence:
        target_joints_rad = pose_joints_rad(camera_poses, view_index)
        settle_time = estimate_settle_time(current_joints_rad, target_joints_rad)
        print(f"[Supervisor] Moving arm to view {view_index}...")
        clear_receiver(arm_status_receiver)
        if not send_arm_pose_command(arm_emitter, view_index):
            return
        print(f"[Supervisor] Waiting for arm arrival (timeout {settle_time:.2f}s)...")
        if not wait_for_arm_arrival(
            supervisor,
            timestep,
            arm_status_receiver,
            str(view_index),
            settle_time,
        ):
            return
        current_joints_rad = target_joints_rad or current_joints_rad
        if POST_ARRIVAL_PAUSE_SEC > 0.0:
            print(f"[Supervisor] Pausing {POST_ARRIVAL_PAUSE_SEC:.2f}s after arrival...")
            if not wait_seconds(supervisor, timestep, POST_ARRIVAL_PAUSE_SEC):
                return

    print("[Supervisor] All viewpoints visited.")


if __name__ == "__main__":
    main()
