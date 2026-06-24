"""arm_home — 把 UR5e 設到 HOME 姿態並保持(靜態可視化用,無需 supervisor)。

HOME = [0, -90, 90, -90, -90, 0] deg(同拍攝端 HOME_POSE_DEG / config REFERENCE_DEG)。
"""
import math

from controller import Robot

HOME_DEG = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]
NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
         "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

robot = Robot()
ts = int(robot.getBasicTimeStep())
for name, deg in zip(NAMES, HOME_DEG):
    m = robot.getDevice(name)
    if m is not None:
        m.setPosition(math.radians(deg))

while robot.step(ts) != -1:
    pass
