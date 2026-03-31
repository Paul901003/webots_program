from controller import Robot, Keyboard
import math

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

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

    # 三組預設姿態
    pose_1 = [0.0, -1.2, 1.8, -1.6, -1.57, 0.0]
    pose_2 = [0.5, -1.0, 1.3, -1.8, -1.57, 0.3]
    pose_3 = [-0.5, -1.3, 1.6, -1.4, -1.57, -0.3]

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

    print("\n=========================================")
    print("控制器已啟動")
    print("先點擊 Webots 3D 視窗再按鍵")
    print("夾爪: C 關 / V 開")
    print("手臂: Q/A, W/S, E/D, R/F, T/G, Y/H")
    print("新功能: Z=Home, X=Stop, P=Print, 1/2/3=Preset")
    print("=========================================\n")

    while robot.step(timestep) != -1:
        key = keyboard.getKey()

        while key != -1:
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
            elif key == ord('A'):
                target_positions[0] -= angle_step
            elif key == ord('W'):
                target_positions[1] += angle_step
            elif key == ord('S'):
                target_positions[1] -= angle_step
            elif key == ord('E'):
                target_positions[2] += angle_step
            elif key == ord('D'):
                target_positions[2] -= angle_step
            elif key == ord('R'):
                target_positions[3] += angle_step
            elif key == ord('F'):
                target_positions[3] -= angle_step
            elif key == ord('T'):
                target_positions[4] += angle_step
            elif key == ord('G'):
                target_positions[4] -= angle_step
            elif key == ord('Y'):
                target_positions[5] += angle_step
            elif key == ord('H'):
                target_positions[5] -= angle_step

            # =========================
            # 新功能
            # =========================
            elif key == ord('Z'):
                target_positions = home_pose[:]
                print("回到 Home pose")

            elif key == ord('X'):
                # 停在目前位置
                if all(s is not None for s in sensors):
                    target_positions = [s.getValue() for s in sensors]
                print("保持目前姿態")

            elif key == ord('P'):
                if all(s is not None for s in sensors):
                    current = [round(s.getValue(), 4) for s in sensors]
                    current_deg = [round(math.degrees(v), 2) for v in current]
                    print(f"rad: {current}")
                    print(f"deg: {current_deg}")
                else:
                    print(f"target rad: {[round(v, 4) for v in target_positions]}")
                print(f"gripper: {round(gripper_position, 4)}")

            elif key == ord('1'):
                target_positions = pose_1[:]
                print("切到 preset 1")

            elif key == ord('2'):
                target_positions = pose_2[:]
                print("切到 preset 2")

            elif key == ord('3'):
                target_positions = pose_3[:]
                print("切到 preset 3")

            key = keyboard.getKey()

        # joint limit clamp
        for i in range(6):
            target_positions[i] = clamp(
                target_positions[i],
                joint_limits[i][0],
                joint_limits[i][1]
            )

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