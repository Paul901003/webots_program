"""teleport_align_probe — 讀真實 Webots 夾爪/相機的三軸方向,和 generate_labels 的算法比對。

真實側:teleport world 設 joint_deg,讀夾爪(finger_joint 的 endpoint solid)與相機 getOrientation → 三軸。
GL 側 :同 joint_deg 算 gripper=wrist_3@Ry(90)Rx(-90) 的三軸。
純 python。
"""
import math
from controller import Supervisor

JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
TEST_JD = [-15.1, -88.0, 77.1, -93.6, -86.2, -14.7]


def matmul(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]

def aa(ax, an):
    n = math.sqrt(sum(v*v for v in ax)); x, y, z = [v/n for v in ax]
    c, s, C = math.cos(an), math.sin(an), 1-math.cos(an)
    return [[c+x*x*C, x*y*C-z*s, x*z*C+y*s], [y*x*C+z*s, c+y*y*C, y*z*C-x*s], [z*x*C-y*s, z*y*C+x*s, c+z*z*C]]

def tf(xyz=(0, 0, 0), ax=None, an=0.0):
    R = aa(ax, an) if ax is not None else [[1,0,0],[0,1,0],[0,0,1]]
    return [R[0]+[xyz[0]], R[1]+[xyz[1]], R[2]+[xyz[2]], [0,0,0,1]]

def col(T, j):  # 取第 j 直行(軸)
    return [round(T[0][j], 3), round(T[1][j], 3), round(T[2][j], 3)]

def ori_axes(node):  # getOrientation 回傳 9 元素 row-major 3x3
    o = node.getOrientation()
    return ([round(o[0],3),round(o[3],3),round(o[6],3)],   # X
            [round(o[1],3),round(o[4],3),round(o[7],3)],   # Y
            [round(o[2],3),round(o[5],3),round(o[8],3)])   # Z


def main():
    sv = Supervisor(); ts = int(sv.getBasicTimeStep())
    arm = sv.getFromDef("ARM"); cam = sv.getFromDef("PROBE_CAM")
    jn = [arm.getFromProtoDef(n) for n in JOINTS]
    fj = arm.getFromProtoDef("finger_joint")
    print(f"[ax] arm={arm is not None} cam={cam is not None} joints_ok={all(x is not None for x in jn)} finger_joint={fj is not None}", flush=True)
    for j, d in zip(jn, TEST_JD):
        j.setJointPosition(math.radians(d))
    sv.simulationResetPhysics()
    for _ in range(3):
        sv.step(ts)

    # 真實 wrist_3 endpoint solid
    w3 = jn[5].getField("endpoint").getSFNode()
    print("[ax] === 真實 Webots ===", flush=True)
    if w3:
        X, Y, Z = ori_axes(w3); print(f"[ax] wrist_3  X={X} Y={Y} Z={Z}", flush=True)
    if fj:
        gnode = fj.getField("endpoint").getSFNode()
        if gnode:
            X, Y, Z = ori_axes(gnode); print(f"[ax] gripper  X={X} Y={Y} Z={Z}  pos={[round(v,4) for v in gnode.getPosition()]}", flush=True)
    Xc, Yc, Zc = ori_axes(cam); print(f"[ax] camera   X={Xc} Y={Yc} Z={Zc}", flush=True)

    # GL 算的 gripper
    q = [math.radians(d) for d in TEST_JD]
    def fk_w3(j):
        Tb=tf([-0.4,0,0]); Ts=matmul(matmul(Tb,tf([0,0,0.163])),tf(ax=[0,0,1],an=j[0]))
        Tu=matmul(matmul(matmul(Ts,tf([0,0.138,0])),tf(ax=[0,1,0],an=j[1])),tf(ax=[0,1,0],an=math.pi/2))
        Tf=matmul(matmul(Tu,tf([0,-0.131,0.425])),tf(ax=[0,1,0],an=j[2]))
        T1=matmul(matmul(matmul(Tf,tf([0,0,0.392])),tf(ax=[0,1,0],an=j[3])),tf(ax=[0,1,0],an=math.pi/2))
        T2=matmul(matmul(T1,tf([0,0.127,0])),tf(ax=[0,0,1],an=j[4])); return matmul(matmul(T2,tf([0,0,0.1])),tf(ax=[0,1,0],an=j[5]))
    Tw3=fk_w3(q); Tgrip=matmul(matmul(Tw3,tf(ax=[0,1,0],an=math.pi/2)),tf(ax=[1,0,0],an=-math.pi/2))
    print("[ax] === generate_labels 算的 ===", flush=True)
    print(f"[ax] GL wrist_3 X={col(Tw3,0)} Y={col(Tw3,1)} Z={col(Tw3,2)}", flush=True)
    print(f"[ax] GL gripper X={col(Tgrip,0)} Y={col(Tgrip,1)} Z={col(Tgrip,2)}", flush=True)
    sv.simulationQuit(0)


if __name__ == "__main__":
    main()
