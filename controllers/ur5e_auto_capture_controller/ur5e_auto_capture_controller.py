import math
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
FLANGE_TO_CAMERA_TRANSLATION_M = [0.005, -0.03, 0.05]
FLANGE_TO_CAMERA_AXIS_ANGLE = [0.0, 0.0, 1.0, 1.5708]

# 可直接填入 6 軸關節角度，單位為 degree。
CAMERA_POSES = {
    1: {"joint_deg": [-0.0, -86.09, 63.92, -67.83, -90.0, 0.0]},
    2: {"joint_deg": [0.0, -142.07, 99.48, -81.57, -90, 0.0]},
    3: {"joint_deg": [31.48, -63.87, 63.66, -64.5, -123.96, -51.03]},
    4: {"joint_deg": [-57.15, -71.88, 44.32, -28.13, -60.21, 31.99]},
}


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


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

    angle_step = 0.05
    joint_limits = [(-2 * math.pi, 2 * math.pi)] * 6
    flange_to_camera_transform = make_fixed_transform(
        FLANGE_TO_CAMERA_TRANSLATION_M,
        FLANGE_TO_CAMERA_AXIS_ANGLE,
    )

    gripper_base_name = "ROBOTIQ 2F-140 Gripper"
    gripper_motor_names = [
        f"{gripper_base_name}::left finger joint",
        f"{gripper_base_name}::right finger joint",
    ]

    gripper_motors = []
    for name in gripper_motor_names:
        motor = robot.getDevice(name)
        if motor:
            motor.setVelocity(2.0)
            gripper_motors.append(motor)

    gripper_position = 0.0
    gripper_step = 0.02
    gripper_min = 0.0
    gripper_max = 0.7
    last_print_time_ms = -PRINT_KEY_DEBOUNCE_MS
    receiver = robot.getDevice(ARM_COMMAND_RECEIVER)
    if receiver:
        receiver.enable(timestep)
    else:
        print(f"警告：找不到接收器 '{ARM_COMMAND_RECEIVER}'，自動切換視角將無法使用。")

    print("\n=========================================")
    print("自動四視角 UR5e controller 已啟動")
    print("Supervisor 會透過 customData 指定 pose=1/2/3/4")
    print("四組 preset 現在直接使用關節角度")
    print("仍可用鍵盤微調: Z=Home, X=Hold, P=Print")
    print("=========================================\n")

    while robot.step(timestep) != -1:
        if receiver:
            while receiver.getQueueLength() > 0:
                message = receiver.getString().strip()
                receiver.nextPacket()
                try:
                    pose_index = int(message)
                except ValueError:
                    pose_index = None

                if pose_index in CAMERA_POSES:
                    try:
                        pose_joint_positions = camera_pose_to_joint_positions(CAMERA_POSES[pose_index])
                    except ValueError as error:
                        print(f"自動拍攝視角 {pose_index} 切換失敗: {error}")
                        continue

                    if not is_within_joint_limits(pose_joint_positions, joint_limits):
                        print(f"自動拍攝視角 {pose_index} 切換失敗: 關節角度超出 joint limit")
                        continue

                    target_positions = pose_joint_positions
                    print(f"切到自動拍攝視角 {pose_index} (joint preset)")

        key = keyboard.getKey()
        while key != -1:
            if key == ord("C"):
                gripper_position += gripper_step
            elif key == ord("V"):
                gripper_position -= gripper_step
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
                    print(f"gripper: {round(gripper_position, 4)}")
                    print(get_separator_line())

            key = keyboard.getKey()

        for i in range(6):
            target_positions[i] = clamp(
                target_positions[i],
                joint_limits[i][0],
                joint_limits[i][1],
            )

        gripper_position = clamp(gripper_position, gripper_min, gripper_max)

        for i in range(6):
            motors[i].setPosition(target_positions[i])

        for motor in gripper_motors:
            motor.setPosition(gripper_position)


if __name__ == "__main__":
    main()
