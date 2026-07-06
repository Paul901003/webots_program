"""ycb_supervisor_movingcam — 單一相機「瞬移」依序拍攝(取代 40 台同時,降載)。

與多相機等價:同一份 validated 視角、同樣 ray_origin−R·[5mm] 的感測器校正、同樣
el/az 檔名。差別是只有 1 台 D455(DEF MOVING_CAM),supervisor 每個視角設定它的
translation/rotation 把它瞬移過去,realsense 控制器暖機後拍存,再移到下一個視角。
輸出與多相機相同(data/captures_multicam/...) → 同一個 extract_subset.py 抽子集。

  MULTICAM_SCENE=n3_scene0001 webots worlds/ycb_movingcam_capture.wbt   # 單場景
  MULTICAM_GROUP=n3           webots worlds/ycb_movingcam_capture.wbt   # 整組
  MOVINGCAM_WAIT=1.0  每視角等待秒數(暖機+存);MOVE_SETTLE=0.1 瞬移後沉澱秒數
"""
import importlib.util as _ilu
import json
import math
import os
import sys

from controller import Supervisor

CUR = os.path.dirname(os.path.abspath(__file__))
CTRL_DIR = os.path.dirname(CUR)
FV_DIR = os.path.join(CTRL_DIR, "ycb_supervisor_four_view_multi")
SUP_DIR = os.path.join(CTRL_DIR, "ycb_supervisor")
MC_DIR = os.path.join(CTRL_DIR, "ycb_supervisor_multicam")
# 先加 FV/SUP 並 import FV,讓 FV 的 `from config import GRID_COLS` 解析到 ycb_supervisor/config.py
# (若先加 MC_DIR,config 會抓到 multicam 的 config → 缺 GRID_COLS)。
for d in (FV_DIR, SUP_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)
import ycb_supervisor_four_view_multi as FV          # spawn/clear/manifest/pose helper
# 再加 MC_DIR/CUR 取 gen 的 helper(gen 的 import config 此時已快取為 ycb_supervisor,但 helper 不用 CFG)
for d in (MC_DIR, CUR):
    if d not in sys.path:
        sys.path.insert(0, d)
from gen_multicam_world import (                     # 視角→節點位姿/檔名(與多相機同算法)
    R_to_axis_angle, el_az_name, load_viewpoints, camera_R_world)

# multicam 的 config(VIEWPOINTS_FILE / 等待秒數 / 輸出根);明確路徑載入避免 config 名稱衝突
_spec = _ilu.spec_from_file_location("multicam_config", os.path.join(MC_DIR, "config.py"))
CFG = _ilu.module_from_spec(_spec); _spec.loader.exec_module(CFG)

DATA_DIR = FV.DATA_DIR
VIEWPOINTS_DIR = os.path.join(DATA_DIR, "viewpoints")
CAPTURE_ROOT = os.environ.get("MULTICAM_ROOT", os.path.join(DATA_DIR, CFG.CAPTURE_ROOT_NAME))
SCENE_SETTLE_SEC = CFG.SCENE_SETTLE_SEC
CAPTURE_WAIT_SEC = float(os.environ.get("MOVINGCAM_WAIT", "1.0"))
MOVE_SETTLE_SEC = float(os.environ.get("MOVE_SETTLE", "0.1"))
CAM_DEF = "MOVING_CAM"
SENSOR_OFFSET_X = 0.005   # D455 感測器沿相機 +X 偏移(與 gen 一致)

SCENE_PLAN_FILES = [
    os.path.join(DATA_DIR, "scene_plans", "multi_scene_plan.json"),
    os.path.join(DATA_DIR, "scene_plans", "occ_scene_plan.json"),
    os.path.join(DATA_DIR, "scene_plans", "stack_scene_plan.json"),
]


def load_all_scenes():
    scenes = {}
    for pf in SCENE_PLAN_FILES:
        if not os.path.exists(pf):
            continue
        for sc in json.load(open(pf, encoding="utf-8")).get("scenes", []):
            if sc.get("scene_name"):
                scenes[sc["scene_name"]] = sc
    return scenes


def group_of(scene_name):
    return scene_name.split("_")[0]


def build_viewpoints():
    """讀 validated 視角 → 每個算 node 位姿(含感測器校正)+ el/az 檔名。"""
    import numpy as np
    src = os.environ.get("MOVINGCAM_VIEWPOINTS", CFG.VIEWPOINTS_FILE)
    if not os.path.isabs(src) and not os.path.exists(src):
        src = os.path.join(VIEWPOINTS_DIR, src)   # 只給檔名 → 接 data/viewpoints/
    vps, meta = load_viewpoints(src)
    target = meta.get("target_m") or [0.35, 0.0, 0.0]
    out = []
    for vp in vps:
        R = camera_R_world(vp.get("joint_deg"), vp["axis"])   # FK 完整旋轉(含 roll)
        node = np.asarray(vp["origin"], float) - R[:, 0] * SENSOR_OFFSET_X
        out.append({
            "view": el_az_name(vp["origin"], target),
            "id": vp["id"], "joint_deg": vp["joint_deg"],
            "node_pos": [float(x) for x in node],
            "rot_aa": [float(x) for x in R_to_axis_angle(R)],
        })
    print(f"[MovingCam] 視角來源 {os.path.basename(src)}: {len(out)} 個視角  target={target}")
    return out


def capture_at(supervisor, timestep, cam_node, tfield, rfield, vp, scene_id, scene_dir):
    tfield.setSFVec3f(vp["node_pos"])
    rfield.setSFRotation(vp["rot_aa"])
    if not FV.wait_seconds(supervisor, timestep, MOVE_SETTLE_SEC):   # GPS/IMU 沉澱
        return None
    token = f"{vp['view']}_{int(supervisor.getTime() * 1000)}"
    joint_str = ",".join(f"{d:.6f}" for d in (vp["joint_deg"] or []))
    cam_node.getField("customData").setSFString(
        f"capture_token={token};view={vp['view']};"
        f"label={scene_id};scene_dir={scene_dir};joint_deg={joint_str}")
    if not FV.wait_seconds(supervisor, timestep, CAPTURE_WAIT_SEC):   # 暖機+存
        return None
    pos, rpy = FV.read_camera_pose(cam_node)
    return pos, rpy


def run_scene(supervisor, timestep, vps, cam_node, tfield, rfield, scene):
    scene_name = scene.get("scene_name")
    scene_dir = os.path.join(CAPTURE_ROOT, f"multi_{group_of(scene_name)}", scene_name)
    os.makedirs(scene_dir, exist_ok=True)
    print(f"[MovingCam] 場景目錄: {scene_dir}")

    FV.clear_ycb_objects(supervisor)
    spawn_positions = FV.spawn_objects(supervisor, scene["objects"])
    if not FV.wait_seconds(supervisor, timestep, SCENE_SETTLE_SEC):
        return False

    names = [o["name"] for o in scene["objects"]]
    actual = []
    for k, vp in enumerate(vps, 1):
        res = capture_at(supervisor, timestep, cam_node, tfield, rfield, vp, scene_name, scene_dir)
        if res is None:
            return False
        pos, rpy = res
        objs = FV.read_object_poses(supervisor, names)
        actual.append({
            "id": vp["id"], "view": vp["view"], "joint_deg": vp["joint_deg"],
            "camera": {"position_m": pos, "rotation_rpy_rad": rpy,
                       "rotation_rpy_deg": [math.degrees(r) for r in rpy]},
            "objects": objs,
            "files": {"rgb": f"{vp['view']}.png", "depth_npy": f"{vp['view']}_depth.npy",
                      "depth_vis": f"{vp['view']}_depth.png"},
        })
        if k % 10 == 0 or k == len(vps):
            print(f"[MovingCam]   已拍 {k}/{len(vps)} 視角")
    supervisor.simulationResetPhysics()

    manifest = {
        "scene_id": scene_name, "scene_dir": scene_dir, "camera_spec": FV.CAMERA_SPEC,
        "planned": {
            "objects": [{"name": n, "spawn_position_m": spawn_positions.get(n, [0, 0, 0]),
                         "spawn_rotation_axis_angle": [0, 1, 0, 0]} for n in names],
            "viewpoints": [{"id": v["id"], "view": v["view"], "joint_deg": v["joint_deg"]}
                           for v in vps],
        },
        "actual": {"viewpoints": actual},
    }
    json.dump(manifest, open(os.path.join(scene_dir, "scene_manifest.json"), "w",
                             encoding="utf-8"), indent=2)
    print(f"[MovingCam] Manifest 已寫入 ({len(actual)} 視角)")
    return True


def main():
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())

    cam_node = supervisor.getFromDef(CAM_DEF)
    if cam_node is None:
        sys.exit(f"[MovingCam] 找不到相機節點 DEF {CAM_DEF}")
    tfield = cam_node.getField("translation")
    rfield = cam_node.getField("rotation")
    vps = build_viewpoints()

    scenes = load_all_scenes()
    scene_arg = os.environ.get("MULTICAM_SCENE")
    group_arg = os.environ.get("MULTICAM_GROUP")
    if scene_arg:
        sel = [scenes[scene_arg]] if scene_arg in scenes else []
        if not sel:
            sys.exit(f"[MovingCam] 場景不存在於任何 scene_plan: {scene_arg}")
    elif group_arg:
        sel = sorted([s for n, s in scenes.items() if group_of(n) == group_arg],
                     key=lambda s: s["scene_name"])
    else:
        sys.exit("[MovingCam] 請設 MULTICAM_SCENE=<場景> 或 MULTICAM_GROUP=<組>")
    print(f"[MovingCam] 待拍 {len(sel)} 個場景 × {len(vps)} 視角")

    for i, scene in enumerate(sel, 1):
        print(f"\n[MovingCam] ── 場景 {i}/{len(sel)}: {scene.get('scene_name')} ──")
        if not run_scene(supervisor, timestep, vps, cam_node, tfield, rfield, scene):
            print("[MovingCam] 場景失敗，中止。")
            return
    print("\n[MovingCam] 所有場景完成。")
    supervisor.simulationQuit(0)


if __name__ == "__main__":
    main()
