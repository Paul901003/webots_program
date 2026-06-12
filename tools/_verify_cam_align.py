#!/home/cho/.pyenv/versions/3.10.10/bin/python3
"""驗證:用 manifest 實際相機位姿渲染,疊到 Webots 拍攝圖上看是否對齊。"""
import json, math, os, sys
import numpy as np
os.environ["PYOPENGL_PLATFORM"] = "egl"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import generate_labels as G
from PIL import Image

SCENE = "n3_scene0001"
GROUP = 3
VP_ID = 6   # 用哪個視角
CAP = f"data/captures/multi_n{GROUP}/{SCENE}"
man = json.load(open(f"{CAP}/scene_manifest.json"))

# 找該 vp
vp = next(v for v in man["actual"]["viewpoints"] if v["id"] == VP_ID)
pos = np.array(vp["camera"]["position_m"])
roll, pitch, yaw = vp["camera"]["rotation_rpy_rad"]

# 反推 Webots 相機→世界旋轉 (ZYX: R = Rz(yaw)Ry(pitch)Rx(roll))
def Rx(a): return np.array([[1,0,0],[0,math.cos(a),-math.sin(a)],[0,math.sin(a),math.cos(a)]])
def Ry(a): return np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
def Rz(a): return np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])
R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

# Webots 相機看 local +X。pyrender 相機看 local -Z。forward 已確認。
forward = R[:, 0]
backward = -forward

# YCB mesh 置中偏移(Webots _make_vrml 用 Transform[-center])
GEO = json.load(open("controllers/ycb_supervisor/ycb_geometries.json"))
def center(n):
    c = GEO.get(n, {}).get("center", {"x":0,"y":0,"z":0})
    return np.array([c["x"], c["y"], c["z"]])

# 載入物體 mesh + 實際位姿(含 -center 修正,對齊 Webots 擺法)
mesh_cache = {o["name"]: G.load_ycb_mesh(G.DEFAULT_ASSETS, o["name"]) for o in vp["objects"]}
ycb_entries = []
for o in vp["objects"]:
    rot = o.get("rotation_axis_angle", [0,1,0,0])
    Raa = np.eye(3)
    if len(rot) == 4 and rot[3] != 0.0:
        Raa = G._axis_angle_to_mat(np.array(rot[:3]), rot[3])
    tf = np.eye(4)
    tf[:3,:3] = Raa
    tf[:3, 3] = np.array(o["position_m"]) - Raa @ center(o["name"])   # T(pos)@R@T(-center)
    ycb_entries.append((mesh_cache[o["name"]], tf, 2, o["name"]))

joint_rad = [math.radians(d) for d in vp["joint_deg"]]
arm = G.load_robot_scene_nodes(G.PROTO_MESH_DIR, joint_rad)
grip = G.load_gripper_nodes(G.GRIPPER_MESH_DIR, joint_rad, 0.0)
K = G.camera_intrinsics()
cap = Image.open(f"{CAP}/view_{VP_ID:02d}.png").convert("RGB").resize((G.CAM_WIDTH, G.CAM_HEIGHT))

candidates = {"+R1": R[:,1], "-R1": -R[:,1], "+R2": R[:,2], "-R2": -R[:,2]}
for name, up_guess in candidates.items():
    right = np.cross(up_guess, backward); right /= np.linalg.norm(right)
    up = np.cross(backward, right); up /= np.linalg.norm(up)
    cam_pose = np.eye(4)
    cam_pose[:3,0]=right; cam_pose[:3,1]=up; cam_pose[:3,2]=backward; cam_pose[:3,3]=pos
    color = G.render_color(arm+grip, ycb_entries[0][0], ycb_entries[0][1], cam_pose, K, extra_ycb=ycb_entries[1:])
    blend = Image.blend(cap, Image.fromarray(color).convert("RGB"), 0.5)
    out = f"/tmp/cam_align_vp{VP_ID}_up{name}.png"
    blend.save(out)
    print("已存:", out)
