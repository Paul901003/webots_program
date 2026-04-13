from controller import Robot, Keyboard
import math
import shutil

PRINT_KEY_DEBOUNCE_MS = 250
ARM_COMMAND_RECEIVER = "arm_command_receiver"

POSES = {
    1: [0.0, -2.2708, 1.2208, -0.9208, -1.5708, 0.0],
    2: [0.8, -1.4208, 0.9208, -0.7708, -2.0708, -0.0],
    3: [-1.0, -1.0708, 0.5708, -0.4208, -1.0708, -0.0],
}


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def get_separator_line(fill_char="="):
    terminal_width = shutil.get_terminal_size(fallback=(80, 20)).columns
    return fill_char * max(terminal_width - 1, 20)


def parse_custom_data(raw_text: str):
    data = {}
    for item in raw_text.split(";"):
        key, sep, value = item.partition("=")
        if sep:
            data[key.strip().lower()] = value.strip()
    return data


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
    print("自動三視角 UR5e controller 已啟動")
    print("Supervisor 會透過 customData 指定 pose=1/2/3")
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

                if pose_index in POSES:
                    target_positions = POSES[pose_index][:]
                    print(f"切到自動拍攝視角 {pose_index}")

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
                        print(f"rad: {current}")
                        print(f"deg: {current_deg}")
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
