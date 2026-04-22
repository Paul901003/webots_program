from controller import Robot, Keyboard
import math
import os
import shutil
import sys

KINEMATICS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "my_ur_kinematics",
)
if KINEMATICS_DIR not in sys.path:
    sys.path.insert(0, KINEMATICS_DIR)

from Foward_Kinematics import FK
from Inverse_Kinematics import IK
from select_ik_solution import select_ik_solution

PRINT_KEY_DEBOUNCE_MS = 250
TRANSLATION_STEP_M = 0.003
ROTATION_STEP_RAD = math.radians(1.0)
CAMERA_TRANSLATION_STEP_M = 0.003
CAMERA_ROTATION_STEP_RAD = math.radians(1.0)
FLANGE_TO_CAMERA_TRANSLATION_M = [0.005, -0.03, 0.05]
FLANGE_TO_CAMERA_AXIS_ANGLE = [0.0, 0.0, 1.0, 1.5708]
# TCP derived from the official Webots Robotiq2f140Gripper.proto:
# midpoint between the left/right inner finger pads, transformed back to the flange frame.
FLANGE_TO_TCP_TRANSLATION_M = [0.0, 0.0, 0.176962]
# Matches the gripper mounting chain in the world: Ry(pi/2) followed by Rx(-pi/2).
FLANGE_TO_TCP_AXIS_ANGLE = [-0.5773502692, 0.5773502692, 0.5773502692, 2.0943951024]


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def get_separator_line(fill_char='='):
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


def rpy_to_rotation_matrix(roll, pitch, yaw):
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)

    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


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


def build_transform_from_pose(position_m, rpy):
    rotation = rpy_to_rotation_matrix(rpy[0], rpy[1], rpy[2])
    return [
        [rotation[0][0], rotation[0][1], rotation[0][2], position_m[0] * 1000.0],
        [rotation[1][0], rotation[1][1], rotation[1][2], position_m[1] * 1000.0],
        [rotation[2][0], rotation[2][1], rotation[2][2], position_m[2] * 1000.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def convert_transform_m_to_mm(transform_m):
    return [
        [transform_m[0][0], transform_m[0][1], transform_m[0][2], transform_m[0][3] * 1000.0],
        [transform_m[1][0], transform_m[1][1], transform_m[1][2], transform_m[1][3] * 1000.0],
        [transform_m[2][0], transform_m[2][1], transform_m[2][2], transform_m[2][3] * 1000.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def convert_transform_mm_to_m(transform_mm):
    return [
        [transform_mm[0][0], transform_mm[0][1], transform_mm[0][2], transform_mm[0][3] / 1000.0],
        [transform_mm[1][0], transform_mm[1][1], transform_mm[1][2], transform_mm[1][3] / 1000.0],
        [transform_mm[2][0], transform_mm[2][1], transform_mm[2][2], transform_mm[2][3] / 1000.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def is_within_joint_limits(joint_positions, joint_limits):
    return all(
        joint_limits[index][0] <= joint_positions[index] <= joint_limits[index][1]
        for index in range(len(joint_positions))
    )


def solve_best_ik_for_flange_pose(position_m, rpy, reference_joints, joint_limits):
    try:
        solutions = IK(build_transform_from_pose(position_m, rpy))
    except ValueError as error:
        return None, f"IK 求解失敗: {error}"

    valid_solutions = []
    for solution in solutions:
        if all(math.isfinite(value) for value in solution) and is_within_joint_limits(solution, joint_limits):
            valid_solutions.append(solution)

    if not valid_solutions:
        return None, "IK 找不到符合 joint limit 的有效解"

    return select_ik_solution(reference_joints, valid_solutions), None


def format_pose_for_print(pose):
    position_mm = [round(value * 1000.0, 2) for value in pose["position_m"]]
    rpy_deg = [round(math.degrees(value), 2) for value in pose["rpy"]]
    return position_mm, rpy_deg


def make_fixed_transform(translation_m, axis_angle):
    rotation = axis_angle_to_rotation_matrix(*axis_angle)
    return compose_transform(rotation, translation_m)


def transform_pose_from_frame(parent_pose, parent_to_child_transform):
    parent_transform_m = convert_transform_mm_to_m(parent_pose["matrix"])
    child_transform_m = multiply_transforms(parent_transform_m, parent_to_child_transform)
    return pose_from_transform(child_transform_m)


def make_transform_from_pose_m(position_m, rpy):
    return compose_transform(rpy_to_rotation_matrix(rpy[0], rpy[1], rpy[2]), position_m)


def apply_local_jog_to_pose(
    pose,
    translation_delta_local=None,
    rotation_axis_local=None,
    rotation_angle=0.0,
):
    transform_m = make_transform_from_pose_m(pose["position_m"], pose["rpy"])
    delta_transform = compose_transform(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation_delta_local or [0.0, 0.0, 0.0],
    )
    transform_m = multiply_transforms(transform_m, delta_transform)

    if rotation_axis_local is not None and abs(rotation_angle) > 1e-12:
        rotation_delta = compose_transform(
            axis_angle_to_rotation_matrix(
                rotation_axis_local[0],
                rotation_axis_local[1],
                rotation_axis_local[2],
                rotation_angle,
            ),
            [0.0, 0.0, 0.0],
        )
        transform_m = multiply_transforms(transform_m, rotation_delta)

    return pose_from_transform(transform_m)


def solve_best_ik_for_child_pose(position_m, rpy, reference_joints, flange_to_child_transform, joint_limits):
    child_target_transform_m = compose_transform(
        rpy_to_rotation_matrix(rpy[0], rpy[1], rpy[2]),
        position_m,
    )
    flange_target_transform_m = multiply_transforms(
        child_target_transform_m,
        invert_transform(flange_to_child_transform),
    )

    try:
        solutions = IK(convert_transform_m_to_mm(flange_target_transform_m))
    except ValueError as error:
        return None, f"IK 求解失敗: {error}"

    valid_solutions = []
    for solution in solutions:
        if all(math.isfinite(value) for value in solution) and is_within_joint_limits(solution, joint_limits):
            valid_solutions.append(solution)

    if not valid_solutions:
        return None, "IK 找不到符合 joint limit 的有效解"

    return select_ik_solution(reference_joints, valid_solutions), None

def main():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    keyboard = robot.getKeyboard()
    keyboard.enable(timestep)

    # =========================================================
    # 1. UR5e 手臂設定
    # =========================================================
    joint_names = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint'
    ]

    motors = []
    for name in joint_names:
        motor = robot.getDevice(name)
        if motor is None:
            print(f"找不到關節馬達: {name}")
            return
        motor.setVelocity(1.5)
        motors.append(motor)

    # 若有 position sensor 就一起開
    sensors = []
    for name in joint_names:
        sensor = robot.getDevice(name + '_sensor')
        if sensor:
            sensor.enable(timestep)
        sensors.append(sensor)

    # Home pose
    home_pose = [0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0]
    target_positions = home_pose[:]

    # 三組預設姿態，以 degree 定義後再轉成 rad 給控制器使用。
    pose_1_deg = [0.0, -130.11, 69.95, -52.76, -90.0, 0.0]
    pose_2_deg = [31.48, -63.87, 63.66, -64.5, -123.96, -51.03]
    pose_3_deg = [-57.15, -71.88, 44.32, -28.13, -60.21, 31.99]
    pose_1 = [math.radians(value) for value in pose_1_deg]
    pose_2 = [math.radians(value) for value in pose_2_deg]
    pose_3 = [math.radians(value) for value in pose_3_deg]
    flange_to_camera_transform = make_fixed_transform(
        FLANGE_TO_CAMERA_TRANSLATION_M,
        FLANGE_TO_CAMERA_AXIS_ANGLE,
    )
    flange_to_tcp_transform = make_fixed_transform(
        FLANGE_TO_TCP_TRANSLATION_M,
        FLANGE_TO_TCP_AXIS_ANGLE,
    )
    flange_target = get_flange_pose(target_positions)
    tcp_target = transform_pose_from_frame(flange_target, flange_to_tcp_transform)
    camera_target = transform_pose_from_frame(flange_target, flange_to_camera_transform)
    ik_control_mode = "tcp"

    angle_step = 0.05

    # UR5e 常見安全範圍，先給寬一點
    joint_limits = [
        (-2 * math.pi, 2 * math.pi),   # shoulder_pan
        (-2 * math.pi, 2 * math.pi),   # shoulder_lift
        (-2 * math.pi, 2 * math.pi),   # elbow
        (-2 * math.pi, 2 * math.pi),   # wrist_1
        (-2 * math.pi, 2 * math.pi),   # wrist_2
        (-2 * math.pi, 2 * math.pi),   # wrist_3
    ]

    # =========================================================
    # 2. ROBOTIQ 2F-140 夾爪設定
    # =========================================================
    gripper_base_name = "ROBOTIQ 2F-140 Gripper"
    gripper_motor_names = [
        f"{gripper_base_name}::left finger joint",
        f"{gripper_base_name}::right finger joint"
    ]

    gripper_motors = []
    for name in gripper_motor_names:
        motor = robot.getDevice(name)
        if motor:
            motor.setVelocity(2.0)
            gripper_motors.append(motor)
        else:
            print(f"警告：找不到夾爪馬達 '{name}'")

    gripper_position = 0.0
    gripper_step = 0.02
    gripper_min = 0.0
    gripper_max = 0.7
    last_print_time_ms = -PRINT_KEY_DEBOUNCE_MS

    print("\n=========================================")
    print("控制器已啟動")
    print("先點擊 Webots 3D 視窗再按鍵")
    print("夾爪: C 關 / V 開")
    print("手臂: Q/A, W/S, E/D, R/F, T/G, Y/H")
    print("Cartesian IK: U/J=X+/X-, I/K=Y+/Y-, O/L=Z+/Z-")
    print("姿態 IK: 4/5=Roll+/-, 6/7=Pitch+/-, 8/9=Yaw+/-")
    print("B: 切換 IK 控制模式 (Flange / TCP / Camera)")
    print("新功能: Z=Home, N=Hold, P=Print(FK), 1/2/3=Preset")
    print("相機: X=視窗開關, P=拍照 (由 realsense_controller 處理)")
    print(f"目前 IK 模式: {ik_control_mode.upper()}")
    print("=========================================\n")

    while robot.step(timestep) != -1:
        key = keyboard.getKey()

        while key != -1:
            joint_target_changed = False

            # =========================
            # 夾爪控制
            # =========================
            if key == ord('C'):
                gripper_position += gripper_step
            elif key == ord('V'):
                gripper_position -= gripper_step

            # =========================
            # 手臂控制
            # =========================
            elif key == ord('Q'):
                target_positions[0] += angle_step
                joint_target_changed = True
            elif key == ord('A'):
                target_positions[0] -= angle_step
                joint_target_changed = True
            elif key == ord('W'):
                target_positions[1] += angle_step
                joint_target_changed = True
            elif key == ord('S'):
                target_positions[1] -= angle_step
                joint_target_changed = True
            elif key == ord('E'):
                target_positions[2] += angle_step
                joint_target_changed = True
            elif key == ord('D'):
                target_positions[2] -= angle_step
                joint_target_changed = True
            elif key == ord('R'):
                target_positions[3] += angle_step
                joint_target_changed = True
            elif key == ord('F'):
                target_positions[3] -= angle_step
                joint_target_changed = True
            elif key == ord('T'):
                target_positions[4] += angle_step
                joint_target_changed = True
            elif key == ord('G'):
                target_positions[4] -= angle_step
                joint_target_changed = True
            elif key == ord('Y'):
                target_positions[5] += angle_step
                joint_target_changed = True
            elif key == ord('H'):
                target_positions[5] -= angle_step
                joint_target_changed = True

            # =========================
            # Cartesian IK 控制
            # =========================
            elif key in (ord('U'), ord('J'), ord('I'), ord('K'), ord('O'), ord('L'),
                         ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9')):
                if ik_control_mode == "camera":
                    current_target_pose = camera_target
                    translation_step = CAMERA_TRANSLATION_STEP_M
                    rotation_step = CAMERA_ROTATION_STEP_RAD
                elif ik_control_mode == "tcp":
                    current_target_pose = tcp_target
                    translation_step = TRANSLATION_STEP_M
                    rotation_step = ROTATION_STEP_RAD
                else:
                    current_target_pose = flange_target
                    translation_step = TRANSLATION_STEP_M
                    rotation_step = ROTATION_STEP_RAD

                translation_delta_local = [0.0, 0.0, 0.0]
                rotation_axis_local = None
                rotation_angle = 0.0

                if key == ord('U'):
                    translation_delta_local[0] += translation_step
                elif key == ord('J'):
                    translation_delta_local[0] -= translation_step
                elif key == ord('I'):
                    translation_delta_local[1] += translation_step
                elif key == ord('K'):
                    translation_delta_local[1] -= translation_step
                elif key == ord('O'):
                    translation_delta_local[2] += translation_step
                elif key == ord('L'):
                    translation_delta_local[2] -= translation_step
                elif key == ord('4'):
                    rotation_axis_local = [1.0, 0.0, 0.0]
                    rotation_angle = rotation_step
                elif key == ord('5'):
                    rotation_axis_local = [1.0, 0.0, 0.0]
                    rotation_angle = -rotation_step
                elif key == ord('6'):
                    rotation_axis_local = [0.0, 1.0, 0.0]
                    rotation_angle = rotation_step
                elif key == ord('7'):
                    rotation_axis_local = [0.0, 1.0, 0.0]
                    rotation_angle = -rotation_step
                elif key == ord('8'):
                    rotation_axis_local = [0.0, 0.0, 1.0]
                    rotation_angle = rotation_step
                elif key == ord('9'):
                    rotation_axis_local = [0.0, 0.0, 1.0]
                    rotation_angle = -rotation_step

                desired_target_pose = apply_local_jog_to_pose(
                    current_target_pose,
                    translation_delta_local=translation_delta_local,
                    rotation_axis_local=rotation_axis_local,
                    rotation_angle=rotation_angle,
                )
                desired_position = desired_target_pose["position_m"]
                desired_rpy = desired_target_pose["rpy"]

                if ik_control_mode == "camera":
                    ik_solution, error_message = solve_best_ik_for_child_pose(
                        desired_position,
                        desired_rpy,
                        target_positions,
                        flange_to_camera_transform,
                        joint_limits,
                    )
                elif ik_control_mode == "tcp":
                    ik_solution, error_message = solve_best_ik_for_child_pose(
                        desired_position,
                        desired_rpy,
                        target_positions,
                        flange_to_tcp_transform,
                        joint_limits,
                    )
                else:
                    ik_solution, error_message = solve_best_ik_for_flange_pose(
                        desired_position,
                        desired_rpy,
                        target_positions,
                        joint_limits,
                    )
                if ik_solution is None:
                    print(error_message)
                else:
                    target_positions = ik_solution[:]
                    flange_target = get_flange_pose(target_positions)
                    tcp_target = transform_pose_from_frame(flange_target, flange_to_tcp_transform)
                    camera_target = transform_pose_from_frame(flange_target, flange_to_camera_transform)

            # =========================
            # 新功能
            # =========================
            elif key == ord('B'):
                if ik_control_mode == "flange":
                    ik_control_mode = "tcp"
                elif ik_control_mode == "tcp":
                    ik_control_mode = "camera"
                else:
                    ik_control_mode = "flange"
                print(f"IK 控制模式切換為: {ik_control_mode.upper()}")

            elif key == ord('Z'):
                target_positions = home_pose[:]
                flange_target = get_flange_pose(target_positions)
                tcp_target = transform_pose_from_frame(flange_target, flange_to_tcp_transform)
                camera_target = transform_pose_from_frame(flange_target, flange_to_camera_transform)
                print("回到 Home pose")

            elif key == ord('N'):
                # 停在目前位置
                if all(s is not None for s in sensors):
                    target_positions = [s.getValue() for s in sensors]
                flange_target = get_flange_pose(target_positions)
                tcp_target = transform_pose_from_frame(flange_target, flange_to_tcp_transform)
                camera_target = transform_pose_from_frame(flange_target, flange_to_camera_transform)
                print("保持目前姿態")

            elif key == ord('X'):
                # X is handled by the camera controller; keep the arm controller inert here.
                pass

            elif key == ord('P'):
                current_time_ms = robot.getTime() * 1000.0
                if current_time_ms - last_print_time_ms >= PRINT_KEY_DEBOUNCE_MS:
                    last_print_time_ms = current_time_ms
                    if all(s is not None for s in sensors):
                        current = [round(s.getValue(), 4) for s in sensors]
                        current_deg = [round(math.degrees(v), 2) for v in current]
                        print(f"rad: {current}")
                        print(f"deg: {current_deg}")
                        flange_pose = get_flange_pose([s.getValue() for s in sensors])
                    else:
                        print(f"target rad: {[round(v, 4) for v in target_positions]}")
                        flange_pose = get_flange_pose(target_positions)

                    tcp_pose = transform_pose_from_frame(flange_pose, flange_to_tcp_transform)
                    camera_pose = transform_pose_from_frame(flange_pose, flange_to_camera_transform)
                    flange_position_mm, flange_rpy_deg = format_pose_for_print(flange_pose)
                    tcp_position_mm, tcp_rpy_deg = format_pose_for_print(tcp_pose)
                    camera_position_mm, camera_rpy_deg = format_pose_for_print(camera_pose)
                    print(f"flange xyz(mm): {flange_position_mm}")
                    print(f"flange rpy(deg): {flange_rpy_deg}")
                    print(f"tcp xyz(mm): {tcp_position_mm}")
                    print(f"tcp rpy(deg): {tcp_rpy_deg}")
                    print(f"camera xyz(mm): {camera_position_mm}")
                    print(f"camera rpy(deg): {camera_rpy_deg}")
                    print(f"ik mode: {ik_control_mode.upper()}")
                    print(f"gripper: {round(gripper_position, 4)}")
                    print(get_separator_line())

            elif key == ord('1'):
                target_positions = pose_1[:]
                flange_target = get_flange_pose(target_positions)
                tcp_target = transform_pose_from_frame(flange_target, flange_to_tcp_transform)
                camera_target = transform_pose_from_frame(flange_target, flange_to_camera_transform)
                print("切到 preset 1")

            elif key == ord('2'):
                target_positions = pose_2[:]
                flange_target = get_flange_pose(target_positions)
                tcp_target = transform_pose_from_frame(flange_target, flange_to_tcp_transform)
                camera_target = transform_pose_from_frame(flange_target, flange_to_camera_transform)
                print("切到 preset 2")

            elif key == ord('3'):
                target_positions = pose_3[:]
                flange_target = get_flange_pose(target_positions)
                tcp_target = transform_pose_from_frame(flange_target, flange_to_tcp_transform)
                camera_target = transform_pose_from_frame(flange_target, flange_to_camera_transform)
                print("切到 preset 3")

            if joint_target_changed:
                flange_target = get_flange_pose(target_positions)
                tcp_target = transform_pose_from_frame(flange_target, flange_to_tcp_transform)
                camera_target = transform_pose_from_frame(flange_target, flange_to_camera_transform)

            key = keyboard.getKey()

        # joint limit clamp
        for i in range(6):
            target_positions[i] = clamp(
                target_positions[i],
                joint_limits[i][0],
                joint_limits[i][1]
            )

        flange_target = get_flange_pose(target_positions)
        tcp_target = transform_pose_from_frame(flange_target, flange_to_tcp_transform)
        camera_target = transform_pose_from_frame(flange_target, flange_to_camera_transform)

        gripper_position = clamp(gripper_position, gripper_min, gripper_max)

        # 更新手臂
        for i in range(6):
            motors[i].setPosition(target_positions[i])

        # 更新夾爪
        # 多數情況兩側可直接同值；若你的 gripper 模型需要一正一負，再改成 gm.setPosition(sign * gripper_position)
        for gm in gripper_motors:
            gm.setPosition(gripper_position)

if __name__ == '__main__':
    main()
