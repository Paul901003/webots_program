"""
A-5: 讀取 planned_paths.json，在 Webots 中依序執行路徑並在每個視角拍攝影像。

使用方式：
  在 Webots 中開啟 worlds/ycb_path_executor.wbt
  影像輸出至 data/captures/a5/<timestamp>/024_bowl/
"""
from controller import Supervisor
from datetime import datetime
from pathlib import Path
import json
import math
import os
import sys

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]

_MULTI_MODE = "--multi" in " ".join(sys.argv[1:])
PLANNED_PATHS_PATH = REPO_ROOT / "data" / "viewpoints" / (
    "planned_paths_multi_latest.json" if _MULTI_MODE else "planned_paths_latest.json"
)
CAPTURES_DIR = REPO_ROOT / "data" / "captures" / ("a5_multi" if _MULTI_MODE else "a5")

# YCB spawn helpers from shared module
SUPERVISOR_DIR = REPO_ROOT / "controllers" / "ycb_supervisor"
if str(SUPERVISOR_DIR) not in sys.path:
    sys.path.insert(0, str(SUPERVISOR_DIR))

from config import ASSET_BASE, MASS_TABLE, SPAWN_CLEARANCE

with (SUPERVISOR_DIR / "ycb_geometries.json").open() as _f:
    YCB_GEO_DATA = json.load(_f)

# Webots devices
CAMERA_DEF = "UR5E_CAMERA"
ARM_COMMAND_EMITTER = "arm_command_emitter"
ARM_STATUS_RECEIVER = "arm_status_receiver"

YCB_OBJECT = "024_bowl"
SCENE_SETTLE_SEC = 1.5
POST_ARRIVAL_PAUSE_SEC = 0.5
CAPTURE_WAIT_SEC = 2.0
ARRIVAL_TIMEOUT_SEC = 120.0

CAMERA_SPEC = {
    "model": "IntelRealsenseD455",
    "resolution": "HD",
    "width": 1280,
    "height": 720,
    "fov_h_rad": 1.4746,
    "min_range_m": 0.3,
    "max_range_m": 3.0,
}


# ── YCB spawn ─────────────────────────────────────────────────────────────────

def _geo(name):
    return YCB_GEO_DATA.get(name, {"center": {"x": 0, "y": 0, "z": 0},
                                    "size":   {"x": .1, "y": .1, "z": .1}})


def spawn_ycb_object(supervisor, name, x=0.0, y=0.0):
    geo = _geo(name)
    cx, cy, cz = geo["center"]["x"], geo["center"]["y"], geo["center"]["z"]
    sz = geo["size"]
    half_z = sz["z"] / 2.0
    z = half_z + SPAWN_CLEARANCE
    mass = MASS_TABLE.get(name, 0.1)
    base = f"{ASSET_BASE}/{name}/google_16k"
    vrml = (
        f'Solid {{\n'
        f'  translation {x:.6f} {y:.6f} {z:.6f}\n'
        f'  children [\n'
        f'    Transform {{\n'
        f'      translation {-cx:.6f} {-cy:.6f} {-cz:.6f}\n'
        f'      children [\n'
        f'        Shape {{\n'
        f'          appearance PBRAppearance {{\n'
        f'            baseColorMap ImageTexture {{ url [ "{base}/texture_map.png" ] }}\n'
        f'            roughness 1  metalness 0\n'
        f'          }}\n'
        f'          geometry Mesh {{ url [ "{base}/textured.obj" ] }}\n'
        f'        }}\n'
        f'      ]\n'
        f'    }}\n'
        f'  ]\n'
        f'  name "{name}"\n'
        f'  boundingObject Box {{ size {sz["x"]:.6f} {sz["y"]:.6f} {sz["z"]:.6f} }}\n'
        f'  physics Physics {{ density -1  mass {mass} }}\n'
        f'}}'
    )
    supervisor.getRoot().getField("children").importMFNodeFromString(-1, vrml)


# ── Planned paths ─────────────────────────────────────────────────────────────

def load_planned_paths():
    """回傳 (metadata, paths)。
    paths 是 list of dict: {from_id, to_id, positions (list of 6-float lists)}。
    相容舊格式 waypoints_rad 與新格式 waypoints（含 positions/velocities/time）。
    """
    with PLANNED_PATHS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    paths = []
    for entry in data.get("paths", []):
        if "waypoints_rad" in entry:
            positions = [list(wp) for wp in entry["waypoints_rad"]]
        else:
            positions = [wp["positions"] for wp in entry["waypoints"]]
        paths.append({
            "from_id": str(entry["from_id"]),
            "to_id":   str(entry["to_id"]),
            "positions": positions,
        })
    return data.get("metadata", {}), paths


# ── Arm control ───────────────────────────────────────────────────────────────

def send_path(emitter, positions, command_id):
    payload = {"type": "path", "waypoints": positions, "gripper": 0.0, "id": str(command_id)}
    emitter.send(json.dumps(payload).encode("utf-8"))


def wait_seconds(supervisor, timestep, seconds):
    steps = max(0, int(seconds * 1000 / max(1, timestep)))
    for _ in range(steps):
        if supervisor.step(timestep) == -1:
            return False
    return True


def wait_for_arrival(supervisor, timestep, receiver, command_id, timeout_sec):
    if receiver is None:
        return wait_seconds(supervisor, timestep, timeout_sec)
    t0 = supervisor.getTime()
    while supervisor.getTime() - t0 <= timeout_sec:
        if supervisor.step(timestep) == -1:
            return False
        while receiver.getQueueLength() > 0:
            msg = receiver.getString()
            receiver.nextPacket()
            try:
                d = json.loads(msg)
            except json.JSONDecodeError:
                continue
            if d.get("status") == "arrived" and str(d.get("command_id")) == str(command_id):
                print(f"  到達 (max_err={float(d.get('max_error_rad', 0)):.4f} rad)")
                return True
    print(f"  等待超時: command={command_id}")
    return False


# ── Camera ───────────────────────────────────────────────────────────────────

def read_camera_pose(camera_node):
    pos = list(camera_node.getPosition())
    ori = list(camera_node.getOrientation())
    m = [[ori[r * 3 + c] for c in range(3)] for r in range(3)]
    sy = math.sqrt(m[0][0] ** 2 + m[1][0] ** 2)
    if sy > 1e-9:
        roll  = math.atan2(m[2][1], m[2][2])
        pitch = math.atan2(-m[2][0], sy)
        yaw   = math.atan2(m[1][0], m[0][0])
    else:
        roll  = math.atan2(-m[1][2], m[1][1])
        pitch = math.atan2(-m[2][0], sy)
        yaw   = 0.0
    return pos, [roll, pitch, yaw]


def trigger_capture(camera_node, scene_dir, view_name, joints_rad, sim_time):
    token = f"a5_{view_name}_{int(sim_time * 1000)}"
    joint_str = ",".join(f"{math.degrees(v):.6f}" for v in joints_rad)
    camera_node.getField("customData").setSFString(
        f"capture_token={token};"
        f"view={view_name};"
        f"scene_dir={scene_dir};"
        f"joint_deg={joint_str}"
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())

    emitter = supervisor.getDevice(ARM_COMMAND_EMITTER)
    receiver = supervisor.getDevice(ARM_STATUS_RECEIVER)
    if receiver:
        receiver.enable(timestep)
    else:
        print(f"[A5] 找不到 {ARM_STATUS_RECEIVER}，改用時間等待")

    camera_node = supervisor.getFromDef(CAMERA_DEF)
    if camera_node is None:
        print(f"[A5] 找不到相機節點 DEF={CAMERA_DEF}")
        return

    metadata, paths = load_planned_paths()
    n_viewpoints = sum(1 for p in paths if p["to_id"] != "home")
    x_offset_m = float(metadata.get("x_offset_m", 0.0))
    print(f"[A5] 載入 {len(paths)} 條路徑段，{n_viewpoints} 個視角")
    print(f"[A5] 規劃時間: {metadata.get('generated_at', 'unknown')}")
    print(f"[A5] x_offset = {x_offset_m:+.3f} m")

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    scene_dir = str(CAPTURES_DIR / date_str / YCB_OBJECT)
    os.makedirs(scene_dir, exist_ok=True)
    print(f"[A5] 輸出目錄: {scene_dir}")

    supervisor.step(timestep)

    ws_max_r = float(metadata.get("workspace_sphere_max_r_m") or 0.3)
    for def_name, radius in [("HEMISPHERE_CAPTURE", None), ("WORKSPACE_SPHERE", ws_max_r)]:
        node = supervisor.getFromDef(def_name)
        if node is None:
            continue
        if x_offset_m != 0.0:
            node.getField("translation").setSFVec3f([x_offset_m, 0.0, 0.0])
        if radius is not None:
            try:
                shape = node.getField("children").getMFNode(0)
                geo   = shape.getField("geometry").getSFNode()
                geo.getField("radius").setSFFloat(radius)
            except Exception:
                pass
    print(f"[A5] 工作空間球體半徑 = {ws_max_r:.3f}m，球體節點 x={x_offset_m:+.3f}m")

    print(f"[A5] 生成 {YCB_OBJECT} at x={x_offset_m:+.3f}m ...")
    spawn_ycb_object(supervisor, YCB_OBJECT, x=x_offset_m, y=0.0)
    if not wait_seconds(supervisor, timestep, SCENE_SETTLE_SEC):
        return

    captured_views = []

    for i, path in enumerate(paths, 1):
        from_id = path["from_id"]
        to_id = path["to_id"]
        positions = path["positions"]
        is_viewpoint = to_id != "home"

        print(f"\n[A5] {i}/{len(paths)}  {from_id} → {to_id}  ({len(positions)} wps)")

        if receiver:
            while receiver.getQueueLength() > 0:
                receiver.nextPacket()

        send_path(emitter, positions, to_id)
        if not wait_for_arrival(supervisor, timestep, receiver, to_id, ARRIVAL_TIMEOUT_SEC):
            print("[A5] 路徑執行失敗，中止")
            return

        if POST_ARRIVAL_PAUSE_SEC > 0:
            if not wait_seconds(supervisor, timestep, POST_ARRIVAL_PAUSE_SEC):
                return

        if is_viewpoint:
            cam_pos, cam_rpy = read_camera_pose(camera_node)
            view_name = f"view_{to_id}"
            trigger_capture(camera_node, scene_dir, view_name, positions[-1], supervisor.getTime())
            print(f"[A5] 拍攝 {view_name}")
            if not wait_seconds(supervisor, timestep, CAPTURE_WAIT_SEC):
                return

            captured_views.append({
                "viewpoint_id": to_id,
                "view_name": view_name,
                "waypoint_count": len(positions),
                "camera": {
                    "position_m": cam_pos,
                    "rotation_rpy_rad": cam_rpy,
                    "rotation_rpy_deg": [math.degrees(r) for r in cam_rpy],
                },
                "files": {
                    "rgb":       f"{view_name}.png",
                    "depth_npy": f"{view_name}_depth.npy",
                    "depth_vis": f"{view_name}_depth.png",
                    "pose":      f"{view_name}_pose.json",
                },
            })

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "scene_dir": scene_dir,
        "ycb_object": YCB_OBJECT,
        "path_plan": {
            "generated_at": metadata.get("generated_at"),
            "viewpoint_count": metadata.get("viewpoint_count"),
            "path_count": metadata.get("path_count"),
        },
        "camera_spec": CAMERA_SPEC,
        "captured_views": captured_views,
    }
    manifest_path = os.path.join(scene_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[A5] 完成：{len(captured_views)} 個視角已拍攝")
    print(f"[A5] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
