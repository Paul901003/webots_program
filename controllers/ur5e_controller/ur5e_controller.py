from controller import Robot, Keyboard
import math

def main():
    # 建立 Robot 實例
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # 啟用鍵盤
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
        if motor:
            motor.setVelocity(1.0)
            motors.append(motor)

    # 初始手臂姿態
    target_positions = [0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0]
    angle_step = 0.05

    # =========================================================
    # 2. ROBOTIQ 2F-140 夾爪設定
    # =========================================================
    # 注意：如果你的夾爪在 Scene Tree 中的 name 不是預設值，請修改下方字串
    gripper_base_name = "ROBOTIQ 2F-140 Gripper"
    gripper_motor_names = [
        f"{gripper_base_name}::left finger joint",
        f"{gripper_base_name}::right finger joint"
    ]
    
    gripper_motors = []
    for name in gripper_motor_names:
        motor = robot.getDevice(name)
        if motor:
            # 夾爪的馬達速度可以設快一點
            motor.setVelocity(2.0)
            gripper_motors.append(motor)
        else:
            print(f"警告：找不到夾爪馬達 '{name}'！請確認夾爪的 name 屬性。")

    # 夾爪初始位置 (0.0 為全開，0.7 為全關)
    gripper_position = 0.0
    gripper_step = 0.02  # 每次按鍵的開合幅度

    print("\n=========================================")
    print("手臂與夾爪控制器已啟動！")
    print("【重要】請先點擊 Webots 的 3D 模擬視窗再操作鍵盤。")
    print("【夾爪控制】 [C] 閉合夾爪 / [V] 張開夾爪")
    print("【手臂控制】 Q/A, W/S, E/D, R/F, T/G, Y/H 控制六個關節")
    print("=========================================\n")

    # 主控制迴圈
    while robot.step(timestep) != -1:
        key = keyboard.getKey()
        
        if key != -1:
            # --- 夾爪控制 ---
            if key == ord('C'):
                gripper_position += gripper_step
                # 限制最大閉合數值為 0.7
                if gripper_position > 0.7:
                    gripper_position = 0.7
            elif key == ord('V'):
                gripper_position -= gripper_step
                # 限制最大張開數值為 0.0
                if gripper_position < 0.0:
                    gripper_position = 0.0

            # --- 手臂控制 ---
            elif key == ord('Q'): target_positions[0] += angle_step
            elif key == ord('A'): target_positions[0] -= angle_step
            elif key == ord('W'): target_positions[1] += angle_step
            elif key == ord('S'): target_positions[1] -= angle_step
            elif key == ord('E'): target_positions[2] += angle_step
            elif key == ord('D'): target_positions[2] -= angle_step
            elif key == ord('R'): target_positions[3] += angle_step
            elif key == ord('F'): target_positions[3] -= angle_step
            elif key == ord('T'): target_positions[4] += angle_step
            elif key == ord('G'): target_positions[4] -= angle_step
            elif key == ord('Y'): target_positions[5] += angle_step
            elif key == ord('H'): target_positions[5] -= angle_step

        # 更新手臂關節位置
        for i in range(len(motors)):
            motors[i].setPosition(target_positions[i])
            
        # 更新夾爪馬達位置
        for gm in gripper_motors:
            gm.setPosition(gripper_position)

if __name__ == '__main__':
    main()