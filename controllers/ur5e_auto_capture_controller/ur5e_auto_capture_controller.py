import math
import json
import os
import shutil
import sys

from controller import Robot, Keyboard

CONTROLLERS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UR5E_CONTROLLER_DIR = os.path.join(CONTROLLERS_DIR, "ur5e_controller")
KINEMATICS_DIR = os.path.join(UR5E_CONTROLLER_DIR, "my_ur_kinematics")
for path in (UR5E_CONTROLLER_DIR, KINEMATICS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from Foward_Kinematics import FK
PRINT_KEY_DEBOUNCE_MS = 250
ARM_COMMAND_RECEIVER = "arm_command_receiver"
ARM_STATUS_EMITTER = "arm_status_emitter"
ARRIVAL_TOLERANCE_RAD = 0.005
ARRIVAL_HOLD_SEC = 0.3
VIA_TOLERANCE_RAD = 0.08
FLANGE_TO_CAMERA_TRANSLATION_M = [0.005, -0.03, 0.05]
FLANGE_TO_CAMERA_AXIS_ANGLE = [0.0, 0.0, 1.0, 1.5708]

# 可直接填入 6 軸關節角度，單位為 degree。
CAMERA_POSES = {
    1: {"joint_deg": [-30.49, -123.31, 101.34, -81.02, -82.45, -29.63]},  # source_id=27 el=75 az=180 ray_miss=0.2mm roll=0.0deg
    2: {"joint_deg": [43.18, -33.39, 33.57, -40.34, -129.17, 151.94]},  # source_id=1 el=30 az=90 ray_miss=0.7mm roll=0.0deg
    3: {"joint_deg": [-61.93, -30.75, 27.45, -29.9, -65.95, -165.06]},  # source_id=7 el=30 az=-90 ray_miss=0.7mm roll=0.0deg
    4: {"joint_deg": [-125.06, -103.76, 129.83, -58.55, -68.58, -166.91]},  # source_id=5 el=30 az=-150 ray_miss=0.6mm roll=0.0deg
    5: {"joint_deg": [17.07, -35.36, 0.01, -32.45, -110.13, 132.95]},  # source_id=14 el=60 az=60 ray_miss=0.5mm roll=6.9deg
    6: {"joint_deg": [55.56, -125.96, 131.44, -99.9, -134.83, 83.73]},  # source_id=10 el=45 az=150 ray_miss=0.5mm roll=0.0deg
}


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def nearest_joint_angle(target, current, lo, hi):
    best, best_dist = target, float('inf')
    for k in range(-3, 4):
        candidate = target + k * 2 * math.pi
        if lo <= candidate <= hi:
            dist = abs(candidate - current)
            if dist < best_dist:
                best_dist, best = dist, candidate
    return best


def resolve_gripper_device_names():
    # Official Webots Robotiq2f140Gripper device names.
    gripper_name = "ROBOTIQ 2F-140 Gripper"
    motor_candidates = [[
        f"{gripper_name}::left finger joint",
        f"{gripper_name}::right finger joint",
    ]]
    sensor_candidates = [[
        f"{gripper_name} left finger joint sensor",
        f"{gripper_name} right finger joint sensor",
    ]]
    return motor_candidates, sensor_candidates


def get_separator_line(fill_char="="):
    terminal_width = shutil.get_terminal_size(fallback=(80, 20)).columns
    return fill_char * max(terminal_width - 1, 20)


def rotation_matrix_to_rpy(matrix):
    sy = math.sqrt(matrix[0][0] ** 2 + matrix[1][0] ** 2)
    singular = sy < 1e-9

    if not singular:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        pitch = math.atan2(-matrix[2][0], sy)
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        roll = math.atan2(-matrix[1][2], matrix[1][1])
        pitch = math.atan2(-matrix[2][0], sy)
        yaw = 0.0

    return [roll, pitch, yaw]


def axis_angle_to_rotation_matrix(axis_x, axis_y, axis_z, angle):
    axis_norm = math.sqrt(axis_x ** 2 + axis_y ** 2 + axis_z ** 2)
    if axis_norm < 1e-9:
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]

    x = axis_x / axis_norm
    y = axis_y / axis_norm
    z = axis_z / axis_norm
    c = math.cos(angle)
    s = math.sin(angle)
    one_minus_c = 1.0 - c

    return [
        [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
        [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
        [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
    ]


def rpy_to_rotation_matrix(roll, pitch, yaw):
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)

    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def compose_transform(rotation, translation_m):
    return [
        [rotation[0][0], rotation[0][1], rotation[0][2], translation_m[0]],
        [rotation[1][0], rotation[1][1], rotation[1][2], translation_m[1]],
        [rotation[2][0], rotation[2][1], rotation[2][2], translation_m[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def multiply_transforms(transform_a, transform_b):
    result = [[0.0] * 4 for _ in range(4)]
    for row in range(4):
        for col in range(4):
            result[row][col] = sum(transform_a[row][k] * transform_b[k][col] for k in range(4))
    return result


def invert_transform(transform):
    rotation = [[transform[row][col] for col in range(3)] for row in range(3)]
    translation = [transform[row][3] for row in range(3)]
    rotation_t = [[rotation[col][row] for col in range(3)] for row in range(3)]
    inverted_translation = [
        -sum(rotation_t[row][col] * translation[col] for col in range(3))
        for row in range(3)
    ]
    return compose_transform(rotation_t, inverted_translation)


def pose_from_transform(transform):
    rotation = [[transform[row][col] for col in range(3)] for row in range(3)]
    position_m = [transform[0][3], transform[1][3], transform[2][3]]
    rpy = rotation_matrix_to_rpy(rotation)
    return {
        "matrix": transform,
        "position_m": position_m,
        "rpy": rpy,
    }


def get_flange_pose(joint_positions):
    transform = FK(joint_positions)
    position_m = [
        transform[0][3] / 1000.0,
        transform[1][3] / 1000.0,
        transform[2][3] / 1000.0,
    ]
    rpy = rotation_matrix_to_rpy(transform)
    return {
        "matrix": transform,
        "position_m": position_m,
        "rpy": rpy,
    }


def is_within_joint_limits(joint_positions, joint_limits):
    return all(
        joint_limits[index][0] <= joint_positions[index] <= joint_limits[index][1]
        for index in range(len(joint_positions))
    )


def make_fixed_transform(translation_m, axis_angle):
    rotation = axis_angle_to_rotation_matrix(*axis_angle)
    return compose_transform(rotation, translation_m)


def transform_pose_from_frame(parent_pose, parent_to_child_transform):
    parent_transform_m = [
        [parent_pose["matrix"][0][0], parent_pose["matrix"][0][1], parent_pose["matrix"][0][2], parent_pose["matrix"][0][3] / 1000.0],
        [parent_pose["matrix"][1][0], parent_pose["matrix"][1][1], parent_pose["matrix"][1][2], parent_pose["matrix"][1][3] / 1000.0],
        [parent_pose["matrix"][2][0], parent_pose["matrix"][2][1], parent_pose["matrix"][2][2], parent_pose["matrix"][2][3] / 1000.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    child_transform_m = multiply_transforms(parent_transform_m, parent_to_child_transform)
    return pose_from_transform(child_transform_m)


def camera_pose_to_joint_positions(camera_pose):
    joint_deg = camera_pose.get("joint_deg")
    if joint_deg is None or len(joint_deg) != 6:
        raise ValueError("CAMERA_POSES 必須提供 6 個 joint_deg")
    return [math.radians(value) for value in joint_deg]


def send_arm_status(emitter, status, command_id, max_error_rad):
    if emitter is None or command_id is None:
        return
    payload = {
        "status": status,
        "command_id": str(command_id),
        "max_error_rad": float(max_error_rad),
    }
    emitter.send(json.dumps(payload).encode("utf-8"))


def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    keyboard = robot.getKeyboard()
    keyboard.enable(timestep)

    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    motors = []
    for name in joint_names:
        motor = robot.getDevice(name)
        if motor is None:
            print(f"找不到關節馬達: {name}")
            return
        motor.setVelocity(1.5)
        motors.append(motor)

    sensors = []
    for name in joint_names:
        sensor = robot.getDevice(name + "_sensor")
        if sensor:
            sensor.enable(timestep)
        sensors.append(sensor)

    home_pose = [0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0]
    target_positions = home_pose[:]

    # 啟動時以各關節最高速直衝 HOME，避免初始姿態碰桌面
    for motor, pos in zip(motors, home_pose):
        motor.setVelocity(motor.getMaxVelocity())
        motor.setPosition(pos)
    robot.step(timestep)
    for motor in motors:
        motor.setVelocity(1.5)

    angle_step = 0.05
    joint_limits = [
        (-2 * math.pi, 2 * math.pi),  # shoulder_pan
        (-2 * math.pi, 2 * math.pi),  # shoulder_lift
        (-math.pi,     math.pi),       # elbow_joint (PROTO: ±π)
        (-2 * math.pi, 2 * math.pi),  # wrist_1
        (-2 * math.pi, 2 * math.pi),  # wrist_2
        (-2 * math.pi, 2 * math.pi),  # wrist_3
    ]
    flange_to_camera_transform = make_fixed_transform(
        FLANGE_TO_CAMERA_TRANSLATION_M,
        FLANGE_TO_CAMERA_AXIS_ANGLE,
    )

    gripper_motors = []
    gripper_motor_candidates, gripper_sensor_candidates = resolve_gripper_device_names()
    for candidate_names in gripper_motor_candidates:
        candidate_motors = []
        for name in candidate_names:
            motor = robot.getDevice(name)
            if motor is None:
                candidate_motors = []
                break
            candidate_motors.append(motor)
        if candidate_motors:
            gripper_motors = candidate_motors
            break

    for motor in gripper_motors:
        motor.setVelocity(2.0)

    gripper_sensors = []
    for candidate_names in gripper_sensor_candidates:
        candidate_sensors = []
        for name in candidate_names:
            sensor = robot.getDevice(name)
            if sensor is None:
                candidate_sensors = []
                break
            candidate_sensors.append(sensor)
        if candidate_sensors:
            gripper_sensors = candidate_sensors
            break

    for sensor in gripper_sensors:
        sensor.enable(timestep)

    gripper_target = 0.0  # 由指令控制，預設全開
    gripper_step = 0.02
    gripper_min = 0.0
    gripper_max = 0.7
    last_print_time_ms = -PRINT_KEY_DEBOUNCE_MS
    receiver = robot.getDevice(ARM_COMMAND_RECEIVER)
    if receiver:
        receiver.enable(timestep)
    else:
        print(f"警告：找不到接收器 '{ARM_COMMAND_RECEIVER}'，自動切換視角將無法使用。")
    status_emitter = robot.getDevice(ARM_STATUS_EMITTER)
    if status_emitter is None:
        print(f"警告：找不到狀態 emitter '{ARM_STATUS_EMITTER}'，Supervisor 將無法用到點確認。")

    active_command_id = None
    arrived_reported = False
    arrival_stable_start = None
    path_waypoints = []   # path 指令的剩餘 waypoints 佇列
    is_path_command = False

    print("\n=========================================")
    print("自動四視角 UR5e controller 已啟動")
    print("Supervisor 會透過 customData 指定 pose=1/2/3/4")
    print("四組 preset 現在直接使用關節角度")
    print("仍可用鍵盤微調: Z=Home, X=Hold, P=Print")
    print("=========================================\n")
    if gripper_motors:
        print("夾爪馬達裝置: ROBOTIQ 2F-140 Gripper::left/right finger joint")
    if not gripper_motors:
        print("警告：找不到夾爪馬達，C/V 開合將停用。")
    if gripper_sensors:
        print("夾爪感測器裝置: ROBOTIQ 2F-140 Gripper left/right finger joint sensor")
    if not gripper_sensors:
        print("警告：找不到夾爪感測器，將改用預設開口值。")

    while robot.step(timestep) != -1:

        if receiver:
            while receiver.getQueueLength() > 0:
                message = receiver.getString().strip()
                receiver.nextPacket()
                # 嘗試解析為 JSON waypoint 指令
                try:
                    json_cmd = json.loads(message)
                except (ValueError, json.JSONDecodeError):
                    json_cmd = None

                if json_cmd is not None and json_cmd.get("type") == "path":
                    waypoints = json_cmd.get("waypoints", [])
                    command_id = str(json_cmd.get("id", "path"))
                    if not waypoints or any(len(w) != 6 for w in waypoints):
                        print(f"path 指令格式錯誤 (id={command_id})")
                    else:
                        gripper_target = clamp(float(json_cmd.get("gripper", 0.0)), gripper_min, gripper_max)
                        current_positions = [s.getValue() for s in sensors]
                        first_wp = [float(v) for v in waypoints[0]]
                        target_positions = [
                            nearest_joint_angle(pos, cur, lo, hi)
                            for pos, cur, (lo, hi) in zip(first_wp, current_positions, joint_limits)
                        ]
                        path_waypoints = [[float(v) for v in w] for w in waypoints[1:]]
                        active_command_id = command_id
                        is_path_command = True
                        arrived_reported = False
                        arrival_stable_start = None
                        send_arm_status(status_emitter, "moving", active_command_id, 999.0)
                        print(f"執行路徑 (id={command_id}, {len(waypoints)} 個 waypoint, gripper={gripper_target:.3f})")
                    continue

                if json_cmd is not None and json_cmd.get("type") == "waypoint":
                    raw_joints = json_cmd.get("joints", [])
                    command_id = str(json_cmd.get("id", "wp"))
                    if len(raw_joints) != 6:
                        print(f"waypoint 指令關節數錯誤: {len(raw_joints)}")
                    else:
                        wp_joints = [float(v) for v in raw_joints]
                        if not is_within_joint_limits(wp_joints, joint_limits):
                            print(f"waypoint {command_id} 超出 joint limit，跳過")
                        else:
                            gripper_target = clamp(float(json_cmd.get("gripper", 0.0)), gripper_min, gripper_max)
                            current_positions = [s.getValue() for s in sensors]
                            target_positions = [
                                nearest_joint_angle(pos, cur, lo, hi)
                                for pos, cur, (lo, hi) in zip(wp_joints, current_positions, joint_limits)
                            ]
                            active_command_id = command_id
                            is_path_command = False
                            path_waypoints = []
                            arrived_reported = False
                            arrival_stable_start = None
                            send_arm_status(status_emitter, "moving", active_command_id, 999.0)
                            print(f"執行 waypoint (id={command_id}, gripper={gripper_target:.3f})")
                    continue

                try:
                    pose_index = int(message)
                except ValueError:
                    pose_index = None

                if message == "home":
                    current_positions = [s.getValue() for s in sensors]
                    new_target = [
                        nearest_joint_angle(pos, cur, lo, hi)
                        for pos, cur, (lo, hi) in zip(home_pose, current_positions, joint_limits)
                    ]
                    target_positions = new_target
                    active_command_id = "home"
                    arrived_reported = False
                    arrival_stable_start = None
                    send_arm_status(status_emitter, "moving", active_command_id, 999.0)
                    print("回到 Home pose (supervisor 指令)")
                elif pose_index in CAMERA_POSES:
                    try:
                        pose_joint_positions = camera_pose_to_joint_positions(CAMERA_POSES[pose_index])
                    except ValueError as error:
                        print(f"自動拍攝視角 {pose_index} 切換失敗: {error}")
                        continue

                    if not is_within_joint_limits(pose_joint_positions, joint_limits):
                        print(f"自動拍攝視角 {pose_index} 切換失敗: 關節角度超出 joint limit")
                        continue

                    current_positions = [s.getValue() for s in sensors]
                    pose_joint_positions = [
                        nearest_joint_angle(pos, cur, lo, hi)
                        for pos, cur, (lo, hi) in zip(pose_joint_positions, current_positions, joint_limits)
                    ]
                    target_positions = pose_joint_positions
                    active_command_id = str(pose_index)
                    arrived_reported = False
                    arrival_stable_start = None
                    send_arm_status(status_emitter, "moving", active_command_id, 999.0)
                    print(f"切到自動拍攝視角 {pose_index} (joint preset)")

        key = keyboard.getKey()
        while key != -1:
            if key == ord("C"):
                gripper_target += gripper_step
            elif key == ord("V"):
                gripper_target -= gripper_step
            elif key == ord("Q"):
                target_positions[0] += angle_step
            elif key == ord("A"):
                target_positions[0] -= angle_step
            elif key == ord("W"):
                target_positions[1] += angle_step
            elif key == ord("S"):
                target_positions[1] -= angle_step
            elif key == ord("E"):
                target_positions[2] += angle_step
            elif key == ord("D"):
                target_positions[2] -= angle_step
            elif key == ord("R"):
                target_positions[3] += angle_step
            elif key == ord("F"):
                target_positions[3] -= angle_step
            elif key == ord("T"):
                target_positions[4] += angle_step
            elif key == ord("G"):
                target_positions[4] -= angle_step
            elif key == ord("Y"):
                target_positions[5] += angle_step
            elif key == ord("H"):
                target_positions[5] -= angle_step
            elif key == ord("Z"):
                target_positions = home_pose[:]
                print("回到 Home pose")
            elif key == ord("X"):
                if all(sensor is not None for sensor in sensors):
                    target_positions = [sensor.getValue() for sensor in sensors]
                print("保持目前姿態")
            elif key == ord("P"):
                current_time_ms = robot.getTime() * 1000.0
                if current_time_ms - last_print_time_ms >= PRINT_KEY_DEBOUNCE_MS:
                    last_print_time_ms = current_time_ms
                    if all(sensor is not None for sensor in sensors):
                        current = [round(sensor.getValue(), 4) for sensor in sensors]
                        current_deg = [round(math.degrees(value), 2) for value in current]
                        flange_pose = get_flange_pose([sensor.getValue() for sensor in sensors])
                        print(f"rad: {current}")
                        print(f"deg: {current_deg}")
                    else:
                        flange_pose = get_flange_pose(target_positions)
                    camera_pose = transform_pose_from_frame(flange_pose, flange_to_camera_transform)
                    camera_xyz_mm = [round(value * 1000.0, 2) for value in camera_pose["position_m"]]
                    camera_rpy_deg = [round(math.degrees(value), 2) for value in camera_pose["rpy"]]
                    print(f"camera xyz(mm): {camera_xyz_mm}")
                    print(f"camera rpy(deg): {camera_rpy_deg}")
                    print(f"gripper: {round(gripper_target, 4)}")
                    print(get_separator_line())

            key = keyboard.getKey()

        for i in range(6):
            target_positions[i] = clamp(
                target_positions[i],
                joint_limits[i][0],
                joint_limits[i][1],
            )

        gripper_target = clamp(gripper_target, gripper_min, gripper_max)

        for i in range(6):
            motors[i].setPosition(target_positions[i])

        for motor in gripper_motors:
            motor.setPosition(gripper_target)

        if active_command_id is not None and all(sensor is not None for sensor in sensors):
            current_positions = [sensor.getValue() for sensor in sensors]
            max_error = max(
                abs(current_positions[i] - target_positions[i])
                for i in range(6)
            )
            if is_path_command and path_waypoints:
                # 中繼點：用寬鬆 tolerance 推進到下一個 waypoint
                if max_error <= VIA_TOLERANCE_RAD:
                    next_wp = path_waypoints.pop(0)
                    target_positions = [float(v) for v in next_wp]
                    arrival_stable_start = None
            else:
                # 最終目標：嚴格 tolerance + 穩定確認後回報 arrived
                if max_error <= ARRIVAL_TOLERANCE_RAD:
                    if arrival_stable_start is None:
                        arrival_stable_start = robot.getTime()
                    elif (
                        not arrived_reported
                        and robot.getTime() - arrival_stable_start >= ARRIVAL_HOLD_SEC
                    ):
                        send_arm_status(status_emitter, "arrived", active_command_id, max_error)
                        arrived_reported = True
                        is_path_command = False
                else:
                    arrival_stable_start = None


if __name__ == "__main__":
    main()
