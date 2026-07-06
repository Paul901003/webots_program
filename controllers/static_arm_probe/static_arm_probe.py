"""static_arm_probe — 把真 UR5e 連桿 mesh 當「無物理靜態 Solid」放到一個 FK 姿態。
驗證:(1)外觀是否為真 UR5e (2)無 physics → 不爆、不漂。純 python(無 numpy)。"""
import math
from controller import Supervisor

ROBOT_BASE = [-0.4, 0.0, 0.0]
MESH_DIR = "/home/cho/webots_program/urdfs/webots_proto_meshes/UR5e/meshes"
TEST_DEG = [-15.1, -88.0, 77.1, -93.6, -86.2, -14.7]
LINK_MESHES = {
    "base":     ["base_link_0", "base_link_1"],
    "shoulder": [f"shoulder_link_{i}" for i in range(4)],
    "upper":    [f"upper_arm_link_{i}" for i in range(10)],
    "forearm":  [f"forearm_link_{i}" for i in range(8)],
    "wrist_1":  [f"wrist_1_link_{i}" for i in range(4)],
    "wrist_2":  [f"wrist_2_link_{i}" for i in range(4)],
    "wrist_3":  ["wrist_3_link_0"],
}

def matmul(A, B):
    return [[sum(A[i][k]*B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
def aa(axis, angle):
    n = math.sqrt(sum(v*v for v in axis)); x,y,z = [v/n for v in axis]
    c,s,C = math.cos(angle), math.sin(angle), 1-math.cos(angle)
    return [[c+x*x*C, x*y*C-z*s, x*z*C+y*s],[y*x*C+z*s, c+y*y*C, y*z*C-x*s],[z*x*C-y*s, z*y*C+x*s, c+z*z*C]]
def tf(xyz=(0,0,0), axis=None, angle=0.0):
    R = aa(axis, angle) if axis is not None else [[1,0,0],[0,1,0],[0,0,1]]
    return [R[0]+[xyz[0]], R[1]+[xyz[1]], R[2]+[xyz[2]], [0,0,0,1]]
def fk(jd):
    j = [math.radians(d) for d in jd]
    Tb = tf(ROBOT_BASE)
    Ts = matmul(matmul(Tb, tf([0,0,0.163])), tf(axis=[0,0,1], angle=j[0]))
    Tu = matmul(matmul(matmul(Ts, tf([0,0.138,0])), tf(axis=[0,1,0], angle=j[1])), tf(axis=[0,1,0], angle=math.pi/2))
    Tf = matmul(matmul(Tu, tf([0,-0.131,0.425])), tf(axis=[0,1,0], angle=j[2]))
    T1 = matmul(matmul(matmul(Tf, tf([0,0,0.392])), tf(axis=[0,1,0], angle=j[3])), tf(axis=[0,1,0], angle=math.pi/2))
    T2 = matmul(matmul(T1, tf([0,0.127,0])), tf(axis=[0,0,1], angle=j[4]))
    T3 = matmul(matmul(T2, tf([0,0,0.1])), tf(axis=[0,1,0], angle=j[5]))
    return {"base":Tb,"shoulder":Ts,"upper":Tu,"forearm":Tf,"wrist_1":T1,"wrist_2":T2,"wrist_3":T3}
def mat2aa(T):
    R = [row[:3] for row in T[:3]]
    ang = math.acos(max(-1.0, min(1.0, (R[0][0]+R[1][1]+R[2][2]-1)/2)))
    if abs(ang) < 1e-9: return [0,0,1], 0.0
    d = 2*math.sin(ang)
    ax = [(R[2][1]-R[1][2])/d, (R[0][2]-R[2][0])/d, (R[1][0]-R[0][1])/d]
    return ax, ang
def link_vrml(name, T, meshes):
    p = [T[0][3], T[1][3], T[2][3]]; ax, ang = mat2aa(T)
    shapes = "".join(f'Shape {{ geometry Mesh {{ url "{MESH_DIR}/{m}.obj" }} }} ' for m in meshes)
    return (f'DEF ARM_{name} Solid {{ translation {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} '
            f'rotation {ax[0]:.6f} {ax[1]:.6f} {ax[2]:.6f} {ang:.6f} children [ {shapes} ] }}')

sv = Supervisor(); ts = int(sv.getBasicTimeStep())
root = sv.getRoot().getField("children")
L = fk(TEST_DEG)
for name, meshes in LINK_MESHES.items():
    root.importMFNodeFromString(-1, link_vrml(name, L[name], meshes))
print("[static] 已放 7 個無物理連桿 Solid,看 GUI 是否為真 UR5e 外觀", flush=True)
sh = sv.getFromDef("ARM_shoulder")
p0 = list(sh.getPosition())
for _ in range(30):
    sv.step(ts)
p1 = list(sh.getPosition())
drift = math.sqrt(sum((p1[k]-p0[k])**2 for k in range(3)))*1000
print(f"[static] shoulder 連桿 30 步漂移 = {drift:.4f} mm  → {'✓ 靜態穩定(無物理不爆)' if drift < 0.001 else '有漂移?'}", flush=True)
print("[static] 完成。手臂應靜止停在該姿態,請目視確認外觀。", flush=True)
while sv.step(ts) != -1:
    pass
