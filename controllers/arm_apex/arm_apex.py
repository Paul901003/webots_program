"""arm_apex — 把 UR5e 設到「圓球頂點拍攝姿態」(view_04, 仰角 90°,相機在中心正上方)並保持。

joint_deg 取自拍攝視角 view_04(static 可視化用,無需 supervisor)。
"""
import math

from controller import Robot

APEX_DEG = [-10.8183, -49.3642, 11.7086, -52.3448, -90.0002, -7.7822]  # view_04
NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
         "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

robot = Robot()
ts = int(robot.getBasicTimeStep())
for name, deg in zip(NAMES, APEX_DEG):
    m = robot.getDevice(name)
    if m is not None:
        m.setPosition(math.radians(deg))
while robot.step(ts) != -1:
    pass
