"""ycb_multicam_capture — multicam(靜態手臂mesh版,相機/夾爪鎖在 wrist_3 上)。
手臂:真 UR5e 連桿 mesh 當無物理靜態 Solid(update-pose:節點常駐、每視角只更新姿態);連桿套 normal/occlusion 貼圖。
相機+夾爪:當 ARM_wrist_3 子節點(掛 toolSlot,隨手臂移動,不獨立瞬移)。物體:spawn(plan z)+settle 一次,全程不 resetPhysics → 不振動。"""
import json
import math
import os
import sys

from controller import Supervisor

CUR = os.path.dirname(os.path.abspath(__file__))
CTRL_DIR = os.path.dirname(CUR)
for d in (os.path.join(CTRL_DIR, "ycb_supervisor_four_view_multi"), os.path.join(CTRL_DIR, "ycb_supervisor")):
    if d not in sys.path:
        sys.path.insert(0, d)
import ycb_supervisor_four_view_multi as FV
for d in (os.path.join(CTRL_DIR, "ycb_supervisor_multicam"), CUR):
    if d not in sys.path:
        sys.path.insert(0, d)
from gen_multicam_world import el_az_name

DATA_DIR = FV.DATA_DIR
REPO = os.path.normpath(os.path.join(DATA_DIR, ".."))
VIEWPOINTS_DIR = os.path.join(DATA_DIR, "viewpoints")
CAPTURE_ROOT = os.environ.get("MULTICAM_ROOT", os.path.join(DATA_DIR, "captures_multicam"))
X_OFFSET = float(os.environ.get("MULTICAM_X_OFFSET", "0.35"))
TARGET = [X_OFFSET, 0.0, 0.0]
SCENE_SETTLE_SEC = float(os.environ.get("MULTICAM_SCENE_SETTLE", "1.5"))
MOVE_SETTLE_SEC = float(os.environ.get("MULTICAM_SETTLE", "0.15"))
CAPTURE_WAIT_SEC = float(os.environ.get("MULTICAM_WAIT", "1.0"))

ARM_MESH = os.path.join(REPO, "meshes", "UR5e", "meshes")
TEX_DIR = os.path.join(REPO, "meshes", "UR5e", "textures")
GRIP_MESH = os.path.join(REPO, "urdfs", "webots_proto_meshes", "robotiq_2f140", "meshes")
ROBOT_BASE = [-0.4, 0.0, 0.0]
CAM_DEF = "TELE_CAM"
LINK_MESHES = {
    "base": ["base_link_0", "base_link_1"], "shoulder": [f"shoulder_link_{i}" for i in range(4)],
    "upper": [f"upper_arm_link_{i}" for i in range(10)], "forearm": [f"forearm_link_{i}" for i in range(8)],
    "wrist_1": [f"wrist_1_link_{i}" for i in range(4)], "wrist_2": [f"wrist_2_link_{i}" for i in range(4)],
    "wrist_3": ["wrist_3_link_0"],
}
LINK_TEX = {"base": None, "shoulder": 0, "upper": 1, "forearm": 2, "wrist_1": 3, "wrist_2": 4, "wrist_3": 5}
LINK_ORDER = ["base", "shoulder", "upper", "forearm", "wrist_1", "wrist_2", "wrist_3"]

def matmul(A, B): return [[sum(A[i][k]*B[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
def aa(axis, angle):
    n = math.sqrt(sum(v*v for v in axis)); x, y, z = [v/n for v in axis]
    c, s, C = math.cos(angle), math.sin(angle), 1-math.cos(angle)
    return [[c+x*x*C, x*y*C-z*s, x*z*C+y*s], [y*x*C+z*s, c+y*y*C, y*z*C-x*s], [z*x*C-y*s, z*y*C+x*s, c+z*z*C]]
def tf(xyz=(0, 0, 0), axis=None, angle=0.0):
    R = aa(axis, angle) if axis is not None else [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    return [R[0]+[xyz[0]], R[1]+[xyz[1]], R[2]+[xyz[2]], [0, 0, 0, 1]]
def mat2aa(T):
    R = [row[:3] for row in T[:3]]
    ang = math.acos(max(-1.0, min(1.0, (R[0][0]+R[1][1]+R[2][2]-1)/2)))
    if abs(ang) < 1e-9: return [0, 0, 1], 0.0
    d = 2*math.sin(ang)
    return [(R[2][1]-R[1][2])/d, (R[0][2]-R[2][0])/d, (R[1][0]-R[0][1])/d], ang

def fk(jd):
    j = [math.radians(d) for d in jd]
    Tb = tf(ROBOT_BASE)
    Ts = matmul(matmul(Tb, tf([0, 0, 0.163])), tf(axis=[0, 0, 1], angle=j[0]))
    Tu = matmul(matmul(matmul(Ts, tf([0, 0.138, 0])), tf(axis=[0, 1, 0], angle=j[1])), tf(axis=[0, 1, 0], angle=math.pi/2))
    Tf = matmul(matmul(Tu, tf([0, -0.131, 0.425])), tf(axis=[0, 1, 0], angle=j[2]))
    T1 = matmul(matmul(matmul(Tf, tf([0, 0, 0.392])), tf(axis=[0, 1, 0], angle=j[3])), tf(axis=[0, 1, 0], angle=math.pi/2))
    T2 = matmul(matmul(T1, tf([0, 0.127, 0])), tf(axis=[0, 0, 1], angle=j[4]))
    T3 = matmul(matmul(T2, tf([0, 0, 0.1])), tf(axis=[0, 1, 0], angle=j[5]))
    return {"base": Tb, "shoulder": Ts, "upper": Tu, "forearm": Tf, "wrist_1": T1, "wrist_2": T2, "wrist_3": T3}

GRIP_BLACK = "0 0 0"
# 每個 mesh 檔的外觀(baseColor, tex_idx),完全照 protos/UR5e.proto 抽出:
# 藍外殼(0.4902 0.678431 0.8+貼圖)/灰金屬(0.6)/黑關節(0 0 0)/螺絲金屬(0.5)。
_BLUE = "0.4902 0.678431 0.8"
MESH_APP = {
    "base_link_0": ("0.5 0.5 0.5", None), "base_link_1": ("0 0 0", None),
    "shoulder_link_0": ("0.6 0.6 0.6", None), "shoulder_link_1": ("0 0 0", None),
    "shoulder_link_2": (_BLUE, 0), "shoulder_link_3": ("0.5 0.5 0.5", None),
    "upper_arm_link_0": (_BLUE, 1), "upper_arm_link_1": ("0.5 0.5 0.5", None),
    "upper_arm_link_2": ("0.6 0.6 0.6", None), "upper_arm_link_3": ("0 0 0", None),
    "upper_arm_link_4": (_BLUE, 2), "upper_arm_link_5": ("0.5 0.5 0.5", None),
    "upper_arm_link_6": ("0.5 0.5 0.5", None), "upper_arm_link_7": ("0 0 0", None),
    "upper_arm_link_8": ("0.6 0.6 0.6", None), "upper_arm_link_9": ("0 0 0", None),
    "forearm_link_0": ("0.6 0.6 0.6", None), "forearm_link_1": ("0.6 0.6 0.6", None),
    "forearm_link_2": ("0 0 0", None), "forearm_link_3": ("0.5 0.5 0.5", None),
    "forearm_link_4": ("0 0 0", None), "forearm_link_5": (_BLUE, 3),
    "forearm_link_6": ("0.5 0.5 0.5", None), "forearm_link_7": ("0 0 0", None),
    "wrist_1_link_0": ("0.6 0.6 0.6", None), "wrist_1_link_1": (_BLUE, 4),
    "wrist_1_link_2": ("0.5 0.5 0.5", None), "wrist_1_link_3": ("0 0 0", None),
    "wrist_2_link_0": ("0.6 0.6 0.6", None), "wrist_2_link_1": (_BLUE, 5),
    "wrist_2_link_2": ("0.5 0.5 0.5", None), "wrist_2_link_3": ("0 0 0", None),
    "wrist_3_link_0": ("0.6 0.6 0.6", None),
}

def _shape(mesh_url, mesh_name=None, color=None):
    tex_idx = None
    if color is None:
        color, tex_idx = MESH_APP.get(mesh_name, ("0.6 0.6 0.6", None))
    if tex_idx is not None:
        rough = "0.4"
        tex = (f'normalMap ImageTexture {{ url [ "{TEX_DIR}/normal_{tex_idx}.jpg" ] }} '
               f'occlusionMap ImageTexture {{ url [ "{TEX_DIR}/occlusion_{tex_idx}.jpg" ] }} ')
    else:
        rough, tex = ("1" if color == "0 0 0" else "0.7"), ""
    return (f'Shape {{ appearance PBRAppearance {{ baseColor {color} roughness {rough} metalness 0 {tex}}} '
            f'geometry Mesh {{ url [ "{mesh_url}" ] }} }} ')

def _gripper_local():
    """夾爪各 mesh 相對 toolSlot frame 的固定變換(finger_angle=0);回傳 [(mesh, T)]。"""
    Tg = matmul(tf(axis=[0, 1, 0], angle=math.pi/2), tf(axis=[1, 0, 0], angle=-math.pi/2))
    Lk = matmul(matmul(Tg, tf([0, -0.030601, 0.054905])), tf(axis=[1, 0, 0], angle=2.295796))
    Lof = matmul(Lk, tf([0, 0.01822, 0.026002]))
    Lif = matmul(matmul(Lof, tf([0, 0.081755, -0.02822])), tf(axis=[1, 0, 0], angle=-0.725))
    Lik = matmul(matmul(Tg, tf([0, -0.0127, 0.06142])), tf(axis=[1, 0, 0], angle=2.295796))
    Rk = matmul(matmul(Tg, tf([0, 0.030601, 0.054905])), tf(axis=[0, 0.911903298450496, 0.41040513431864556], angle=math.pi))
    Rof = matmul(Rk, tf([0, 0.01822, 0.026002]))
    Rif = matmul(matmul(Rof, tf([0, 0.081755, -0.02822])), tf(axis=[1, 0, 0], angle=-0.725))
    Rik = matmul(matmul(Tg, tf([0, 0.0127, 0.06142])), tf(axis=[0, -0.911903298450496, -0.41040513431864556], angle=math.pi))
    # 內指墊片(Box,非 mesh):inner finger frame + [0,0.045755,-0.02722] Rx(-0.01)
    Lpad = matmul(Lif, tf([0, 0.045755, -0.02722], axis=[1, 0, 0], angle=-0.01))
    Rpad = matmul(Rif, tf([0, 0.045755, -0.02722], axis=[1, 0, 0], angle=-0.01))
    return [("robotiq_base_link", Tg), ("robotiq_2f140_outer_knuckle", Lk), ("robotiq_2f140_outer_finger", Lof),
            ("robotiq_2f140_inner_finger", Lif), ("robotiq_2f140_inner_knuckle", Lik),
            ("robotiq_2f140_outer_knuckle", Rk), ("robotiq_2f140_outer_finger", Rof),
            ("robotiq_2f140_inner_finger", Rif), ("robotiq_2f140_inner_knuckle", Rik),
            ("PAD", Lpad), ("PAD", Rpad)]

def _pose_wrap(T, inner):
    p = [T[0][3], T[1][3], T[2][3]]; ax, an = mat2aa(T)
    return f'Pose {{ translation {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} rotation {ax[0]:.6f} {ax[1]:.6f} {ax[2]:.6f} {an:.6f} children [ {inner} ] }} '

def _toolslot_subtree():
    """toolSlot 下:相機(DEF TELE_CAM) + 夾爪 mesh。相對 wrist_3 的 toolSlot Pose(translation 0 0.1 0)內。"""
    cam = ('DEF TELE_CAM IntelRealsenseD455 { translation 0 -0.03 0.1 rotation 0 0 1 1.5708 '
           'name "movingcam" controller "realsense_auto_capture_controller" resolution "HD" minRange 0.3 maxRange 3.0 fps 30 } ')
    def _gs(m):
        # 照 Robotiq proto:外指節=BrushedAluminium(銀金屬),其餘+墊片=BLACK_METAL(0 0 0 roughness 0.4)
        if m == "robotiq_2f140_outer_knuckle":
            app = 'appearance PBRAppearance { baseColor 0.75 0.75 0.75 roughness 0.3 metalness 1 } '
        else:
            app = 'appearance PBRAppearance { baseColor 0 0 0 roughness 0.4 metalness 0 } '
        if m == "PAD":
            return f'Shape {{ {app}geometry Box {{ size 0.027 0.065 0.0075 }} }} '
        return f'Shape {{ {app}geometry Mesh {{ url [ "{GRIP_MESH}/{m}.stl" ] }} }} '
    grip = "".join(_pose_wrap(T, _gs(m)) for m, T in _gripper_local())
    return f'Pose {{ translation 0 0.1 0 children [ {cam} {grip} ] }} '

def _link_solid(name, T):
    p = [T[0][3], T[1][3], T[2][3]]; ax, an = mat2aa(T)
    shapes = "".join(_shape(f"{ARM_MESH}/{m}.obj", m) for m in LINK_MESHES[name])
    extra = _toolslot_subtree() if name == "wrist_3" else ""
    return (f'DEF ARM_{name} Solid {{ translation {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} '
            f'rotation {ax[0]:.6f} {ax[1]:.6f} {ax[2]:.6f} {an:.6f} children [ {shapes}{extra} ] }}')

def build_arm(sv, jd):
    for name in LINK_ORDER:      # 清舊
        n = sv.getFromDef(f"ARM_{name}")
        if n is not None:
            n.remove()
    root = sv.getRoot().getField("children")
    L = fk(jd)
    for name in LINK_ORDER:
        root.importMFNodeFromString(-1, _link_solid(name, L[name]))

def update_arm(sv, jd):
    L = fk(jd)
    for name in LINK_ORDER:
        n = sv.getFromDef(f"ARM_{name}")
        p = [L[name][0][3], L[name][1][3], L[name][2][3]]; ax, an = mat2aa(L[name])
        n.getField("translation").setSFVec3f(p)
        n.getField("rotation").setSFRotation([ax[0], ax[1], ax[2], an])


def load_all_scenes():
    scenes = {}
    for pf in [os.path.join(DATA_DIR, "scene_plans", f) for f in
               ("multi_scene_plan.json", "occ_scene_plan.json", "stack_scene_plan.json")]:
        if os.path.exists(pf):
            for sc in json.load(open(pf, encoding="utf-8")).get("scenes", []):
                if sc.get("scene_name"):
                    scenes[sc["scene_name"]] = sc
    return scenes

def group_of(n): return n.split("_")[0]

def build_viewpoints():
    src = os.environ.get("MULTICAM_VIEWPOINTS", os.path.join(VIEWPOINTS_DIR, "validated_viewpoints_multi_latest.json"))
    if not os.path.isabs(src) and not os.path.exists(src):
        src = os.path.join(VIEWPOINTS_DIR, src)
    vps = FV.load_teleport_viewpoints(src)
    print(f"[Multicam] 視角來源 {os.path.basename(src)}: {len(vps)} 個  target={TARGET}", flush=True)
    return vps


def capture_at(sv, ts, cam, vp, scene_id, scene_dir):
    update_arm(sv, vp["joint_deg"])   # 手臂到 FK,相機/夾爪(子節點)跟著
    if not FV.wait_seconds(sv, ts, MOVE_SETTLE_SEC):
        return None
    pos, _ = FV.read_camera_pose(cam)
    view = el_az_name(pos, TARGET)
    token = f"{view}_{int(sv.getTime()*1000)}"
    joint_str = ",".join(f"{d:.6f}" for d in vp["joint_deg"])
    cam.getField("customData").setSFString(
        f"capture_token={token};view={view};label={scene_id};scene_dir={scene_dir};joint_deg={joint_str}")
    if not FV.wait_seconds(sv, ts, CAPTURE_WAIT_SEC):
        return None
    pos, rpy = FV.read_camera_pose(cam)
    return view, pos, rpy


def run_scene(sv, ts, vps, scene):
    scene_name = scene.get("scene_name")
    scene_dir = os.path.join(CAPTURE_ROOT, f"multi_{group_of(scene_name)}", scene_name)
    os.makedirs(scene_dir, exist_ok=True)
    print(f"[Multicam] 場景目錄: {scene_dir}", flush=True)
    FV.clear_ycb_objects(sv)
    spawn_positions = FV.spawn_objects(sv, scene["objects"])
    if not FV.wait_seconds(sv, ts, SCENE_SETTLE_SEC):
        return False
    build_arm(sv, vps[0]["joint_deg"])          # 匯入手臂(含相機/夾爪),之後只 update
    cam = sv.getFromDef(CAM_DEF)
    if cam is None:
        print("[Multicam] ★相機匯入失敗", flush=True); return False
    FV.wait_seconds(sv, ts, 0.5)                 # 相機 controller 暖機
    names = [o["name"] for o in scene["objects"]]
    actual = []
    for k, vp in enumerate(vps, 1):
        res = capture_at(sv, ts, cam, vp, scene_name, scene_dir)
        if res is None:
            return False
        view, pos, rpy = res
        actual.append({
            "id": vp["id"], "view": view, "joint_deg": vp["joint_deg"],
            "camera": {"position_m": pos, "rotation_rpy_rad": rpy,
                       "rotation_rpy_deg": [math.degrees(r) for r in rpy]},
            "objects": FV.read_object_poses(sv, names),
            "files": {"rgb": f"{view}.png", "depth_npy": f"{view}_depth.npy", "depth_vis": f"{view}_depth.png"},
        })
        if k % 10 == 0 or k == len(vps):
            print(f"[Multicam]   已拍 {k}/{len(vps)} 視角", flush=True)
    manifest = {
        "scene_id": scene_name, "scene_dir": scene_dir, "camera_spec": FV.CAMERA_SPEC,
        "planned": {"objects": [{"name": n, "spawn_position_m": spawn_positions.get(n, [0, 0, 0]),
                                 "spawn_rotation_axis_angle": [0, 1, 0, 0]} for n in names],
                    "viewpoints": [{"id": v["id"], "joint_deg": v["joint_deg"]} for v in vps]},
        "actual": {"viewpoints": actual},
    }
    json.dump(manifest, open(os.path.join(scene_dir, "scene_manifest.json"), "w", encoding="utf-8"), indent=2)
    print(f"[Multicam] Manifest 已寫入 ({len(actual)} 視角)", flush=True)
    return True


def main():
    sv = Supervisor(); ts = int(sv.getBasicTimeStep())
    vps = build_viewpoints()
    scenes = load_all_scenes()
    scene_arg = os.environ.get("MULTICAM_SCENE"); group_arg = os.environ.get("MULTICAM_GROUP")
    if scene_arg:
        sel = [scenes[scene_arg]] if scene_arg in scenes else []
        if not sel:
            sys.exit(f"[Multicam] 場景不存在: {scene_arg}")
    elif group_arg:
        sel = sorted([s for n, s in scenes.items() if group_of(n) == group_arg], key=lambda s: s["scene_name"])
    else:
        sys.exit("[Multicam] 請設 MULTICAM_SCENE 或 MULTICAM_GROUP")
    print(f"[Multicam] 待拍 {len(sel)} 場景 × {len(vps)} 視角", flush=True)
    for i, scene in enumerate(sel, 1):
        print(f"\n[Multicam] ── 場景 {i}/{len(sel)}: {scene.get('scene_name')} ──", flush=True)
        if not run_scene(sv, ts, vps, scene):
            print("[Multicam] 場景失敗，中止。", flush=True); return
    print("\n[Multicam] 所有場景完成。", flush=True)
    sv.simulationQuit(0)


if __name__ == "__main__":
    main()
