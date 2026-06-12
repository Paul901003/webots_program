#!/usr/bin/env python3
"""generate_multi_object_scenes.py

產生 data/scene_plans/multi_scene_plan.json（n3/n4/n5 多物體場景）。

一步完成：
  1. 依物體池為每個 group size（3/4/5）抽選物體組合，每物體剛好出現 N 次（= 該組物體數）。
  2. **強制不重疊 + 整個物體(含高度)都在工作空間球內**：rejection sampling。
     - 工作空間是半徑 = cam_r - WS_OFFSET（不寫死；cam_r 取自 selected_viewpoints）、球心在桌面 z=0 的球。
     - 不出球（含高度）：物體坐在桌上(z:0→h)，頂端外緣須在球內 → 中心水平可動半徑
       = sqrt(WS^2 - h^2) - footprint/2。高物體被推向中心、頂端不戳出球面
       （低仰角視角時手臂沿球面外側移動才不會掃到高物體）。
     - 不重疊：兩物體中心距離 >= (footprintA + footprintB)/2 + MARGIN，footprint = max(size.x, size.y)。
  3. **自動排除過大物體**：若某物體組合在工作空間內排不下，移除失敗場景中 footprint 最大的物體、
     重新生成，直到所有場景皆可擺入；回報被移除的物體。

不存在「先產生重疊、再修正」的中間步驟；不重疊與不出界皆於生成時即強制保證。

輸出覆蓋原檔，首次覆寫時備份成 multi_scene_plan.json.bak。
"""

import json
import math
import os
import random
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(CURRENT_DIR))
SOURCE_CTRL = os.path.join(os.path.dirname(CURRENT_DIR), "ycb_supervisor")
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from config import ALL_OBJECTS, MASS_TABLE, SPACING_MARGIN, TARGET_OBJECTS  # noqa: E402

SELECTED_VIEWPOINTS_PATH = os.path.join(REPO_ROOT, "data", "viewpoints", "selected_viewpoints_multi_latest.json")
SCENE_PLAN_PATH          = os.path.join(REPO_ROOT, "data", "scene_plans", "multi_scene_plan.json")
GEO_PATH                 = os.path.join(SOURCE_CTRL, "ycb_geometries.json")

# 場景組合
GROUP_SIZES  = (3, 4, 5)          # 對應 n3 / n4 / n5
SEED         = 20260609

# 工作空間（可達球體）：半徑 = cam_r - WS_OFFSET，擺放區綁定此半徑（不寫死）。
WS_OFFSET    = 0.30               # 必須與 A-4 plan_viewpoint_paths.py 的 --ws-offset 一致
DEFAULT_CENTER_X = 0.35           # selected_viewpoints 無 x_offset_m 時的後備值
CENTER_Y     = 0.0
MARGIN       = SPACING_MARGIN     # 物體間最小邊距（公尺）
MAX_ATTEMPTS = 8000              # 單物體取樣上限


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_viewpoints():
    data = load_json(SELECTED_VIEWPOINTS_PATH)
    selected = data.get("selected", [])
    if not selected:
        raise ValueError(f"{SELECTED_VIEWPOINTS_PATH} 中沒有 selected 欄位")
    center_x = float(data.get("x_offset_m", DEFAULT_CENTER_X))
    radii = [float(s["meta"]["radius_m"]) for s in selected
             if s.get("meta", {}).get("radius_m") is not None]
    if not radii:
        raise ValueError(f"{SELECTED_VIEWPOINTS_PATH} 的 selected 內找不到 meta.radius_m，無法取得 cam_r")
    cam_r = max(radii)
    viewpoints = [{"id": i + 1, "joint_deg": rec["joint_deg"]} for i, rec in enumerate(selected)]
    return viewpoints, center_x, cam_r


def get_object_pool(geo):
    pool = TARGET_OBJECTS[:] if TARGET_OBJECTS else ALL_OBJECTS[:]
    return [n for n in pool if n in MASS_TABLE and n in geo]


# ── 組合（appearance-balanced）────────────────────────────────────────────────

def build_combos(pool, n, appear, rng):
    """每個物體剛好出現 appear 次（不超量）。

    most-frequent-first：每場景挑「剩餘需求最多」的 n 個相異物體，保證可行性。
    若 物體數×appear 不能被 n 整除，最後不足以湊成一組的殘餘 token 捨棄，
    對應物體會少 1 次（appear-1），其餘全部剛好 appear 次。
    """
    if len(pool) < n:
        raise ValueError(f"物體數 {len(pool)} 不足以組成 {n} 個一組的場景")
    need = {o: appear for o in pool}     # 剩餘待擺放次數
    used = set()
    combos = []
    while True:
        avail = [o for o in pool if need[o] > 0]
        if len(avail) < n:
            break                        # 無法再湊成完整一組 → 殘餘捨棄
        avail.sort(key=lambda o: (-need[o], rng.random()))
        combo = tuple(sorted(avail[:n]))
        if combo in used:                # 嘗試在剩餘最多的一批中換成未用過的組合
            window = avail[:min(len(avail), n + 4)]
            for _ in range(20):
                cand = tuple(sorted(rng.sample(window, n)))
                if cand not in used:
                    combo = cand
                    break
        used.add(combo)
        combos.append(list(combo))
        for o in combo:
            need[o] -= 1
    appearances = {o: appear - need[o] for o in pool}
    return combos, appearances


# ── 不重疊擺位 ────────────────────────────────────────────────────────────────

def footprint(geo, name):
    sz = geo.get(name, {"size": {"x": 0.1, "y": 0.1}})["size"]
    return max(sz["x"], sz["y"])


def sample_in_disk(rng, center_x, radius):
    r = radius * math.sqrt(rng.random())
    a = rng.uniform(0.0, 2.0 * math.pi)
    return center_x + r * math.cos(a), CENTER_Y + r * math.sin(a)


def obj_height(geo, name):
    return geo.get(name, {"size": {"z": 0.1}})["size"]["z"]


def place_scene(rng, geo, names, center_x, workspace_r):
    """在工作空間球內擺放，保證 footprint 不重疊且**整個物體(含高度)都在球內**。

    工作空間是半徑 workspace_r、球心在桌面 (z=0) 的球。物體坐在桌上 (z:0→h)。
    要整個物體在球內，最嚴苛是「頂端外緣」：sqrt((r+fr)^2 + h^2) <= workspace_r，
    故物體中心水平可動半徑 = sqrt(workspace_r^2 - h^2) - fr。
    這同時保證低仰角視角時手臂(沿球面外側到視角)不會掃到戳出球面的高物體。

    成功回傳 [(x, y), ...]；某物體整體放不進球、或取樣上限內排不下 → 回傳 None。
    """
    radii   = [footprint(geo, n) / 2.0 for n in names]
    # 每物體中心可動半徑（3D 球面約束）
    maxr = []
    for fr, n in zip(radii, names):
        inner = workspace_r ** 2 - obj_height(geo, n) ** 2
        maxr.append(math.sqrt(inner) - fr if inner > 0 else -1.0)
    if any(m < 0 for m in maxr):                      # 有物體整體塞不進球
        return None

    # 大物體優先擺放（裝箱啟發法，提高填入成功率），最後再依原順序回傳
    order  = sorted(range(len(names)), key=lambda k: -radii[k])
    result = [None] * len(names)
    placed = []                                       # [(x, y, r)]
    for idx in order:
        r_obj = radii[idx]
        for _ in range(MAX_ATTEMPTS):
            x, y = sample_in_disk(rng, center_x, maxr[idx])
            if all(math.dist((x, y), (px, py)) >= r_obj + pr + MARGIN
                   for px, py, pr in placed):
                placed.append((x, y, r_obj))
                result[idx] = (round(x, 4), round(y, 4))
                break
        else:
            return None
    return result


# ── 主流程 ────────────────────────────────────────────────────────────────────

def build_all(pool, geo, viewpoints, center_x, workspace_r):
    """以目前物體池生成所有場景。

    回傳 (scenes, failed_combos, stats)；failed_combos 為排不下的 (n, names) 清單。
    """
    rng = random.Random(SEED)          # 每次完整嘗試都從同一 SEED 開始 → 可重現
    scenes, failed, stats = [], [], {}
    for n in GROUP_SIZES:
        combos, appearances = build_combos(pool, n, n, rng)
        for i, names in enumerate(combos, 1):
            coords = place_scene(rng, geo, names, center_x, workspace_r)
            if coords is None:
                failed.append((n, names))
                continue
            objects = [{"name": nm, "position_m": [x, y, 0.0]} for nm, (x, y) in zip(names, coords)]
            scenes.append({
                "scene_name": f"n{n}_scene{i:04d}",
                "objects":    objects,
                "viewpoints": viewpoints,
            })
        stats[n] = (len(combos), min(appearances.values()), max(appearances.values()))
    return scenes, failed, stats


def main():
    viewpoints, center_x, cam_r = load_viewpoints()
    geo  = load_json(GEO_PATH)
    pool = get_object_pool(geo)

    workspace_r = cam_r - WS_OFFSET
    print(f"視角: {len(viewpoints)}  中心 x: {center_x}")
    print(f"工作空間半徑 = cam_r {cam_r} - WS_OFFSET {WS_OFFSET} = {workspace_r:.3f} m （擺放上限，物體不出界）")

    # 可行性驅動排除：移除導致場景排不下的過大物體，直到所有場景皆可擺入工作空間
    removed = []
    while True:
        scenes, failed, stats = build_all(pool, geo, viewpoints, center_x, workspace_r)
        if not failed:
            break
        culprits = {nm for _, names in failed for nm in names}
        victim = max(culprits, key=lambda nm: footprint(geo, nm))
        removed.append((victim, round(footprint(geo, victim), 3)))
        pool.remove(victim)
        print(f"  排除過大物體: {victim} (footprint {footprint(geo, victim):.3f} m) — {len(failed)} 個場景塞不下")

    print(f"最終物體池: {len(pool)}（原始 64，移除 {len(removed)} 個）")
    for n in GROUP_SIZES:
        cnt, lo, hi = stats[n]
        print(f"  n{n}: {cnt} 場景  (每物體出現 {lo}~{hi} 次)")
    if removed:
        print("被移除（footprint 過大、無法在工作空間內與他物共存）:")
        for nm, fp in removed:
            print(f"  {nm}: {fp} m")

    bak = SCENE_PLAN_PATH + ".bak"
    if os.path.exists(SCENE_PLAN_PATH) and not os.path.exists(bak):
        os.replace(SCENE_PLAN_PATH, bak)
        print(f"原檔已備份: {bak}")

    os.makedirs(os.path.dirname(SCENE_PLAN_PATH), exist_ok=True)
    with open(SCENE_PLAN_PATH, "w", encoding="utf-8") as f:
        json.dump({"scenes": scenes}, f, indent=2, ensure_ascii=False)

    print(f"生成 {len(scenes)} 個場景  輸出: {SCENE_PLAN_PATH}")
    print(f"所有物體皆落在工作空間半徑 {workspace_r:.3f} m 內、互不重疊。")


if __name__ == "__main__":
    main()
