"""ycb_supervisor_relation_capture — 拍攝「關係豐富」場景(堆疊 stack{N} / 遮擋 occ{N})。

不動現役 ycb_supervisor_four_view_multi,改用 import + monkeypatch 重用其全部機制,只覆寫:
  ① spawn 尊重 scene_plan 的 z(堆疊上物才放在底物上;原版強制 z=half_height)。
  ② 輸出資料夾 = data/captures/multi_{group}(group=scene 名前綴 stack{N}/occ{N})。
  ③ settle 後穩定性檢查:spawn↔settle 位移 > 1cm → 印警告(該場景仍拍,但記錄不穩;堆疊倒了 GT 關係自然反映)。

參數(RELATION_ARGS 環境變數優先,或 controllerArgs):
  "stack" | "occ"   選場景計畫(預設 stack)
  "--N"             只跑某物數(如 --3)
  "--<num>"         只跑某場景(需配 --N)
例:RELATION_ARGS="occ --4 --7" webots worlds/ycb_relation_capture.wbt
"""
import json
import math
import os
import sys
from pathlib import Path

FOUR_VIEW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "ycb_supervisor_four_view_multi")
if FOUR_VIEW_DIR not in sys.path:
    sys.path.insert(0, FOUR_VIEW_DIR)
import ycb_supervisor_four_view_multi as M   # noqa: E402  重用全部機制

REPO = Path(M.REPO_ROOT)
PLANS = REPO / "data" / "scene_plans"
STABILITY_DRIFT_M = 0.01   # spawn↔settle 位移 > 此值視為不穩


def spawn_respect_z(supervisor, objects):
    """覆寫:有 position_m[2]>0 時直接用 plan 的 z(支援堆疊),否則沿用半高。"""
    root = supervisor.getRoot().getField("children")
    spawn_positions = {}
    for obj in objects:
        name = obj if isinstance(obj, str) else obj["name"]
        if name not in M.MASS_TABLE:
            continue
        pos = None if isinstance(obj, str) else obj.get("position_m")
        if pos is not None:
            x, y = float(pos[0]), float(pos[1])
            z = float(pos[2]) if len(pos) >= 3 and float(pos[2]) > 1e-6 \
                else max(M.SPAWN_HEIGHT, M._half_height(name) + M.SPAWN_CLEARANCE)
        else:
            x = M.REFERENCE_X + M.X_OFFSET; y = M.REFERENCE_Y + M.Z_OFFSET
            z = max(M.SPAWN_HEIGHT, M._half_height(name) + M.SPAWN_CLEARANCE)
        root.importMFNodeFromString(-1, M._make_vrml(name, x, y, z))
        spawn_positions[name] = [x, y, z]
    return spawn_positions


M.spawn_objects = spawn_respect_z   # monkeypatch:run_scene 內部呼叫會解析到此


def check_stability(scene_dir, spawn_positions):
    """讀 manifest 比 spawn↔settle 位移,印不穩物體(堆疊倒塌偵測)。"""
    mp = Path(scene_dir) / "scene_manifest.json"
    if not mp.is_file():
        return
    man = json.loads(mp.read_text())
    vps = man.get("actual", {}).get("viewpoints", [])
    if not vps:
        return
    settled = {o["name"]: o["position_m"] for o in vps[0].get("objects", [])}
    bad = []
    for n, sp in spawn_positions.items():
        st = settled.get(n)
        if st and math.dist(sp, st) > STABILITY_DRIFT_M:
            bad.append((n, round(math.dist(sp, st), 3)))
    if bad:
        print(f"[Relation] ⚠ 不穩(spawn↔settle 位移>{STABILITY_DRIFT_M}m): {bad}")
    else:
        print("[Relation] ✓ 穩定")


def parse_args():
    """RELATION_ARGS: "stack|occ" [--N] [--start] [--end]
    例:occ --4(occ4 全部)、stack --5 --6(單場 0006)、stack --5 --6 --20(0006~0020 範圍)。"""
    a = os.environ.get("RELATION_ARGS")
    a = a.split() if a else sys.argv[1:]
    kind = "stack"
    nf = start = end = None
    for x in a:
        if x in ("stack", "occ"):
            kind = x
        elif x.startswith("--") and x[2:].isdigit():
            v = int(x[2:])
            if nf is None:
                nf = v
            elif start is None:
                start = v
            elif end is None:
                end = v
    if start is not None and end is None:
        end = start          # 單場
    return kind, nf, start, end


def main():
    sv = M.Supervisor()
    ts = int(sv.getBasicTimeStep())
    kind, nf, start, end = parse_args()
    plan = json.loads((PLANS / f"{kind}_scene_plan.json").read_text())
    scenes = plan["scenes"]
    if nf is not None:
        scenes = [s for s in scenes if s["scene_name"].startswith(f"{kind}{nf}_")]
        if start is not None:
            scenes = [s for s in scenes
                      if start <= int(s["scene_name"].split("scene")[1]) <= end]
    print(f"[Relation] kind={kind} N={nf} 範圍={start}..{end} 場景數={len(scenes)}")

    cam = sv.getFromDef(M.CAMERA_DEF)
    em = sv.getDevice(M.ARM_COMMAND_EMITTER)
    rc = sv.getDevice(M.ARM_STATUS_RECEIVER)
    if rc:
        rc.enable(ts)
    path_dict, visit_order = M.load_planned_paths()

    for i, scene in enumerate(scenes, 1):
        group = scene["scene_name"].split("_")[0]            # stack3 / occ4 ...
        cap_dir = os.path.join(M.DATA_DIR, "captures", f"multi_{group}")
        print(f"\n[Relation] ── {i}/{len(scenes)} {scene['scene_name']} → multi_{group} ──")
        ok = M.run_scene(sv, ts, em, rc, cam, scene,
                         path_dict=path_dict, visit_order=visit_order, captures_dir=cap_dir)
        if not ok:
            print("[Relation] 場景失敗,中止。"); break
        # 穩定性檢查需 spawn_positions;從 manifest 的 planned.spawn_position_m 取
        sp = {o["name"]: o["spawn_position_m"]
              for o in json.loads((Path(cap_dir) / scene["scene_name"] / "scene_manifest.json")
                                  .read_text())["planned"]["objects"]}
        check_stability(os.path.join(cap_dir, scene["scene_name"]), sp)
    print("\n[Relation] 完成。")
    sv.simulationQuit(0)


if __name__ == "__main__":
    main()
