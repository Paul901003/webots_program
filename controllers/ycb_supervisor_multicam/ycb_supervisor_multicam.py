"""ycb_supervisor_multicam — 多相機同步拍攝 supervisor(無手臂)。

N 台靜態 IntelRealsenseD455 已由 gen_multicam_world.py 擺在驗證視角的正確位姿,
本 supervisor 每場景:清場 → spawn 物體 → 靜置 → 同一步觸發全部相機(各自 realsense
控制器存 view_XX.png/_depth/_pose.json)→ 等待 → 寫 scene_manifest。省去手臂逐視角移動。
位姿由各相機 GPS+IMU 讀出,與既有 view_XX_pose.json 同格式同慣例。

用法(env 優先,同 CAPTURE_ARGS 慣例):
  MULTICAM_SCENE=n3_scene0001 webots worlds/ycb_multicam_capture.wbt   # 單場景
  MULTICAM_GROUP=n3         webots worlds/ycb_multicam_capture.wbt     # 整組
輸出根:預設 data/captures_multicam/multi_<grp>/<scene>(不覆蓋既有 captures);可用 MULTICAM_ROOT 覆寫。
"""
from controller import Supervisor
import json
import math
import os
import sys

CUR = os.path.dirname(os.path.abspath(__file__))
CTRL_DIR = os.path.dirname(CUR)
FV_DIR = os.path.join(CTRL_DIR, "ycb_supervisor_four_view_multi")
SUP_DIR = os.path.join(CTRL_DIR, "ycb_supervisor")
for d in (FV_DIR, SUP_DIR, CUR):
    if d not in sys.path:
        sys.path.insert(0, d)

import ycb_supervisor_four_view_multi as FV   # 複用 spawn/clear/manifest/位姿 helper
# 明確以檔案路徑載入 multicam 自己的 config(避免與 ycb_supervisor/config.py 名稱衝突——
# FV 已 from config import 把全域 config 快取,直接 import config 會拿到錯的那個)。
import importlib.util as _ilu
_cfg_spec = _ilu.spec_from_file_location("multicam_config", os.path.join(CUR, "config.py"))
CFG = _ilu.module_from_spec(_cfg_spec); _cfg_spec.loader.exec_module(CFG)

DATA_DIR = FV.DATA_DIR
SIDECAR = os.path.join(CUR, "multicam_viewpoints.json")
SCENE_SETTLE_SEC = CFG.SCENE_SETTLE_SEC
CAPTURE_WAIT_SEC = CFG.CAPTURE_WAIT_SEC         # 每批暖機+存檔,給足裕度
BATCH_SIZE = int(os.environ.get("MULTICAM_BATCH", CFG.CAPTURE_BATCH_SIZE))  # 0=全部同時
CAPTURE_ROOT = os.environ.get(
    "MULTICAM_ROOT", os.path.join(DATA_DIR, CFG.CAPTURE_ROOT_NAME))

SCENE_PLAN_FILES = [
    os.path.join(DATA_DIR, "scene_plans", "single_scene_plan.json"),
    os.path.join(DATA_DIR, "scene_plans", "multi_scene_plan.json"),
    os.path.join(DATA_DIR, "scene_plans", "occ_scene_plan.json"),
    os.path.join(DATA_DIR, "scene_plans", "stack_scene_plan.json"),
]


def load_all_scenes():
    """合併所有 scene_plan → {scene_name: scene}。"""
    scenes = {}
    for pf in SCENE_PLAN_FILES:
        if not os.path.exists(pf):
            continue
        plan = json.load(open(pf, encoding="utf-8"))
        for sc in plan.get("scenes", []):
            nm = sc.get("scene_name")
            if nm:
                scenes[nm] = sc
    return scenes


def group_of(scene_name):
    """n3_scene0001 → n3;occ4_sceneXXXX → occ4。"""
    return scene_name.split("_")[0]


def _trigger_batch(supervisor, batch, scene_id, scene_dir):
    t_ms = int(supervisor.getTime() * 1000)
    for cam in batch:
        joint_str = ",".join(f"{d:.6f}" for d in (cam["joint_deg"] or []))
        cam["node"].getField("customData").setSFString(
            f"capture_token={cam['view']}_{t_ms};"
            f"view={cam['view']};"
            f"label={scene_id};"
            f"scene_dir={scene_dir};"
            f"joint_deg={joint_str}"
        )


def trigger_and_capture(supervisor, timestep, cams, scene, scene_dir):
    """觸發相機並等待存檔(可分批,不必同時)。回傳 actual viewpoints 清單。"""
    names = [o["name"] for o in scene["objects"]]
    scene_id = scene.get("scene_name")
    bs = BATCH_SIZE if BATCH_SIZE > 0 else len(cams)
    n_batch = (len(cams) + bs - 1) // bs
    for bi in range(n_batch):
        batch = cams[bi * bs:(bi + 1) * bs]
        _trigger_batch(supervisor, batch, scene_id, scene_dir)
        tag = f"批次 {bi+1}/{n_batch} ({len(batch)} 台)" if n_batch > 1 else f"全部 {len(batch)} 台"
        print(f"[Multicam] 觸發{tag}...")
        if not FV.wait_seconds(supervisor, timestep, CAPTURE_WAIT_SEC):
            return None
    actual_objects = FV.read_object_poses(supervisor, names)
    actual = []
    for cam in cams:
        pos, rpy = FV.read_camera_pose(cam["node"])
        actual.append({
            "id": cam["id"], "view": cam["view"], "joint_deg": cam["joint_deg"],
            "camera": {"position_m": pos, "rotation_rpy_rad": rpy,
                       "rotation_rpy_deg": [math.degrees(r) for r in rpy]},
            "objects": actual_objects,
            "files": {"rgb": f"{cam['view']}.png",
                      "depth_npy": f"{cam['view']}_depth.npy",
                      "depth_vis": f"{cam['view']}_depth.png"},
        })
    return actual


def run_scene(supervisor, timestep, cams, scene):
    scene_name = scene.get("scene_name")
    grp = group_of(scene_name)
    scene_dir = os.path.join(CAPTURE_ROOT, f"multi_{grp}", scene_name)
    os.makedirs(scene_dir, exist_ok=True)
    print(f"[Multicam] 場景目錄: {scene_dir}")

    FV.clear_ycb_objects(supervisor)
    spawn_positions = FV.spawn_objects(supervisor, scene["objects"])
    if not FV.wait_seconds(supervisor, timestep, SCENE_SETTLE_SEC):
        return False

    actual = trigger_and_capture(supervisor, timestep, cams, scene, scene_dir)
    if actual is None:
        return False
    supervisor.simulationResetPhysics()

    names = [o["name"] for o in scene["objects"]]
    manifest = {
        "scene_id": scene_name, "scene_dir": scene_dir,
        "camera_spec": FV.CAMERA_SPEC,
        "planned": {
            "objects": [{"name": n,
                         "spawn_position_m": spawn_positions.get(n, [0, 0, 0]),
                         "spawn_rotation_axis_angle": [0, 1, 0, 0]} for n in names],
            "viewpoints": [{"id": c["id"], "view": c["view"], "joint_deg": c["joint_deg"]}
                           for c in cams],
        },
        "actual": {"viewpoints": actual},
    }
    json.dump(manifest, open(os.path.join(scene_dir, "scene_manifest.json"),
                             "w", encoding="utf-8"), indent=2)
    print(f"[Multicam] Manifest 已寫入 ({len(actual)} 視角)")
    return True


def main():
    supervisor = Supervisor()
    timestep = int(supervisor.getBasicTimeStep())

    if not os.path.exists(SIDECAR):
        sys.exit(f"[Multicam] 找不到 {SIDECAR}，請先跑 gen_multicam_world.py")
    sc = json.load(open(SIDECAR, encoding="utf-8"))
    cams = []
    for c in sc["cameras"]:
        node = supervisor.getFromDef(c["def"])
        if node is None:
            print(f"[Multicam] 警告:找不到相機 {c['def']}")
            continue
        cams.append({**c, "node": node})
    print(f"[Multicam] 已連結 {len(cams)} 台相機 (target={sc.get('target_m')})")

    scenes = load_all_scenes()
    scene_arg = os.environ.get("MULTICAM_SCENE")
    group_arg = os.environ.get("MULTICAM_GROUP")
    if scene_arg:
        sel = [scenes[scene_arg]] if scene_arg in scenes else []
        if not sel:
            sys.exit(f"[Multicam] 場景不存在於任何 scene_plan: {scene_arg}")
    elif group_arg:
        sel = [s for n, s in scenes.items() if group_of(n) == group_arg]
        sel.sort(key=lambda s: s["scene_name"])
    else:
        sys.exit("[Multicam] 請設 MULTICAM_SCENE=<場景> 或 MULTICAM_GROUP=<組>")
    print(f"[Multicam] 待拍 {len(sel)} 個場景")

    for i, scene in enumerate(sel, 1):
        print(f"\n[Multicam] ── 場景 {i}/{len(sel)}: {scene.get('scene_name')} ──")
        if not run_scene(supervisor, timestep, cams, scene):
            print("[Multicam] 場景失敗，中止。")
            return
    print("\n[Multicam] 所有場景完成。")
    supervisor.simulationQuit(0)


if __name__ == "__main__":
    main()
