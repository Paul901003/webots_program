from controller import Robot, Keyboard

ARM_JOINTS = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

GRIPPER_JOINTS = [
    'finger_joint',
    'left_inner_finger_joint',
    'left_inner_knuckle_joint',
    'right_outer_knuckle_joint',
    'right_inner_finger_joint',
    'right_inner_knuckle_joint',
]

HOME_Q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
READY_Q = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]

GRIPPER_OPEN_Q = {
    'finger_joint': 0.6,
    'left_inner_finger_joint': 0.0,
    'left_inner_knuckle_joint': 0.0,
    'right_outer_knuckle_joint': 0.0,
    'right_inner_finger_joint': 0.0,
    'right_inner_knuckle_joint': 0.0,
}

class Controller:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())

        self.keyboard = Keyboard()
        self.keyboard.enable(self.timestep)

        self.motors = {}
        self.sensors = {}

        for name in ARM_JOINTS + GRIPPER_JOINTS:
            self.motors[name] = self.robot.getDevice(name)
            sensor = self.robot.getDevice(f'{name}_sensor')
            sensor.enable(self.timestep)
            self.sensors[name] = sensor

        print("Press 1 -> HOME, 2 -> READY, O -> open gripper")

    def step(self):
        return self.robot.step(self.timestep)

    def read_joint(self, name):
        return self.sensors[name].getValue()

    def command_joint(self, name, value):
        self.motors[name].setPosition(value)

    def move_arm_linear(self, q_goal, steps=120):
        q_start = [self.read_joint(name) for name in ARM_JOINTS]
        for k in range(steps + 1):
            a = k / steps
            for i, name in enumerate(ARM_JOINTS):
                q = (1 - a) * q_start[i] + a * q_goal[i]
                self.command_joint(name, q)
            if self.step() == -1:
                return False
        return True

    def open_gripper(self, hold_steps=50):
        for name, value in GRIPPER_OPEN_Q.items():
            self.command_joint(name, value)
        for _ in range(hold_steps):
            if self.step() == -1:
                return False
        return True

ctrl = Controller()

while ctrl.step() != -1:
    key = ctrl.keyboard.getKey()

    if key == ord('1'):
        print("Go HOME")
        ctrl.move_arm_linear(HOME_Q)

    elif key == ord('2'):
        print("Go READY")
        ctrl.move_arm_linear(READY_Q)

    elif key == ord('O') or key == ord('o'):
        print("Open gripper")
        ctrl.open_gripper()
