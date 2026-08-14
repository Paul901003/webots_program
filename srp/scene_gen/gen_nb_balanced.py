#!/usr/bin/env python3
"""gen_nb_balanced.py — 平衡散放場景 nb(nb3/nb4/nb5/nb6)。獨立新檔,不動舊生成器。

★ 生成方法逐字沿用舊 generate_multi_object_scenes.py(place_scene 3D 半球 + 零容忍移除迴圈),
  只改「數量/需求」:
    · 物體池排除 GLOBAL_EXCLUDE(skillet_lid/windex/070a/070b)
    · 每物出現次數 appear = 8N(舊為 N)→ 場景數放大
    · 物數 N = 3/4/5/6(舊 3/4/5)
    · 輸出 nb_scene_plan.json、場景名 nb{N}_scene(舊 multi/n{N};不覆蓋)
  擺放/排除機制完全不變:footprint=max(邊)/2、z=0、3D 半球 R=√、
  零容忍移除(任一場景排不下→移除 footprint 最大者全域→重生,直到 0 失敗)。
用法: ./srp/scene_gen/gen_nb_balanced.py
"""
import json
import math
import os
import sys
import random

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SOURCE_CTRL = os.path.join(REPO_ROOT, "controllers", "ycb_supervisor")
sys.path.insert(0, SOURCE_CTRL)
from config import ALL_OBJECTS, GLOBAL_EXCLUDE, MASS_TABLE, SPACING_MARGIN, TARGET_OBJECTS  # noqa: E402

SELECTED_VIEWPOINTS_PATH = os.path.join(REPO_ROOT, "data", "viewpoints", "selected_viewpoints_multi_latest.json")
SCENE_PLAN_PATH = os.path.join(REPO_ROOT, "data", "scene_plans", "nb_scene_plan.json")
GEO_PATH = os.path.join(SOURCE_CTRL, "ycb_geometries.json")

GROUP_SIZES = (3, 4, 5, 6)             # 改:舊 (3,4,5)
APPEAR = lambda n: 8 * n               # 改:舊為 n
SEED = 20260609
WS_OFFSET = 0.30
DEFAULT_CENTER_X = 0.35
CENTER_Y = 0.0
MARGIN = SPACING_MARGIN
MAX_ATTEMPTS = 8000


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
        raise ValueError(f"{SELECTED_VIEWPOINTS_PATH} 找不到 meta.radius_m")
    cam_r = max(radii)
    viewpoints = [{"id": i + 1, "joint_deg": rec["joint_deg"]} for i, rec in enumerate(selected)]
    return viewpoints, center_x, cam_r


def get_object_pool(geo):
    pool = TARGET_OBJECTS[:] if TARGET_OBJECTS else ALL_OBJECTS[:]
    # 改:加 GLOBAL_EXCLUDE
    return [n for n in pool if n in MASS_TABLE and n in geo and n not in GLOBAL_EXCLUDE]


# ── 以下 build_combos / footprint / sample_in_disk / obj_height / place_scene 與舊版逐字相同 ──
def build_combos(pool, n, appear, rng):
    if len(pool) < n:
        raise ValueError(f"物體數 {len(pool)} 不足以組成 {n} 個一組的場景")
    need = {o: appear for o in pool}
    used = set(); combos = []
    while True:
        avail = [o for o in pool if need[o] > 0]
        if len(avail) < n:
            break
        avail.sort(key=lambda o: (-need[o], rng.random()))
        combo = tuple(sorted(avail[:n]))
        if combo in used:
            window = avail[:min(len(avail), n + 4)]
            for _ in range(20):
                cand = tuple(sorted(rng.sample(window, n)))
                if cand not in used:
                    combo = cand; break
        used.add(combo); combos.append(list(combo))
        for o in combo:
            need[o] -= 1
    appearances = {o: appear - need[o] for o in pool}
    return combos, appearances


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
    radii = [footprint(geo, n) / 2.0 for n in names]
    maxr = []
    for fr, n in zip(radii, names):
        inner = workspace_r ** 2 - obj_height(geo, n) ** 2
        maxr.append(math.sqrt(inner) - fr if inner > 0 else -1.0)
    if any(m < 0 for m in maxr):
        return None
    order = sorted(range(len(names)), key=lambda k: -radii[k])
    result = [None] * len(names)
    placed = []
    for idx in order:
        r_obj = radii[idx]
        for _ in range(MAX_ATTEMPTS):
            x, y = sample_in_disk(rng, center_x, maxr[idx])
            if all(math.dist((x, y), (px, py)) >= r_obj + pr + MARGIN for px, py, pr in placed):
                placed.append((x, y, r_obj))
                result[idx] = (round(x, 4), round(y, 4))
                break
        else:
            return None
    return result


def build_all(pool, geo, viewpoints, center_x, workspace_r):
    """與舊版相同,只改 appear=8N、場景名 nb{n}。回傳 (scenes, failed, stats)。"""
    rng = random.Random(SEED)
    scenes, failed, stats = [], [], {}
    for n in GROUP_SIZES:
        combos, appearances = build_combos(pool, n, APPEAR(n), rng)   # 改:appear=8N
        for i, names in enumerate(combos, 1):
            coords = place_scene(rng, geo, names, center_x, workspace_r)
            if coords is None:
                failed.append((n, names))
                continue
            objects = [{"name": nm, "position_m": [x, y, 0.0]} for nm, (x, y) in zip(names, coords)]
            scenes.append({"scene_name": f"nb{n}_scene{i:04d}",           # 改:nb 前綴
                           "objects": objects, "viewpoints": viewpoints})
        stats[n] = (len(combos), min(appearances.values()), max(appearances.values()))
    return scenes, failed, stats


def main():
    viewpoints, center_x, cam_r = load_viewpoints()
    geo = load_json(GEO_PATH)
    pool = get_object_pool(geo)
    workspace_r = cam_r - WS_OFFSET
    print(f"視角: {len(viewpoints)}  中心 x: {center_x}")
    print(f"工作空間半徑 = {workspace_r:.3f} m")

    # 零容忍移除迴圈(與舊版逐字相同)
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

    print(f"最終物體池: {len(pool)}（原始 {len(ALL_OBJECTS)}，GLOBAL_EXCLUDE {len(GLOBAL_EXCLUDE)}，共置移除 {len(removed)}）")
    for n in GROUP_SIZES:
        cnt, lo, hi = stats[n]
        print(f"  nb{n}: {cnt} 場景  (每物體出現 {lo}~{hi} 次)")
    if removed:
        print("共置移除（footprint 過大、無法與他物共存）:")
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


if __name__ == "__main__":
    main()
