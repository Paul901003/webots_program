#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""gen_relation_scenes.py — 生成「關係豐富」場景計畫:堆疊(產 on)+ 密集遮擋(產 blocks_access)。

與原隨機桌面場景分開儲存:
  scene 前綴 stack{N} / occ{N} → 下游自動進 data/captures/multi_stack{N}/、sam_only/stack{N}_scene*、labels/...
輸出場景計畫(與 multi 分開):
  data/scene_plans/stack_scene_plan.json   (堆疊;objects 帶顯式 z,平頂底物 + 置中上物)
  data/scene_plans/occ_scene_plan.json     (遮擋;物體緊密群聚 → 多視角互遮)

幾何穩定原則(可重現):上物 footprint 完全落在底物頂面內、置中、z=底頂+上半高 → 本就靜止;
實際穩定性由拍攝端 supervisor settle 後驗證(位移>1cm 或翻倒則該場景重生)。
座標系沿用現有拍攝(look-at 中心 [0.35,0,0])。seeded RNG → layout 可重現。
用法: ./srp/scene_gen/gen_relation_scenes.py [--per 20] [--n 3 4 5] [--seed 42]
"""

import argparse
import json
import math
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GEO = json.loads((REPO / "controllers" / "ycb_supervisor" / "ycb_geometries.json").read_text())
PLANS = REPO / "data" / "scene_plans"

CENTER = (0.35, 0.0)          # look-at 中心(沿用現有拍攝)
WS_R = 0.20                    # 物體分佈半徑(框得住、可達)
# 可堆疊白名單(真平頂/平底;SHAPE_TABLE 是碰撞框形狀,pear/dice 都標 Box 不可信)
STACKABLE = {
    "003_cracker_box", "004_sugar_box", "008_pudding_box", "009_gelatin_box",
    "010_potted_meat_can", "036_wood_block", "061_foam_brick", "026_sponge",
    "070-a_colored_wood_blocks", "070-b_colored_wood_blocks",
    "002_master_chef_can", "005_tomato_soup_can", "007_tuna_fish_can",
}


def _shape_table():
    """從 config.py 解析 SHAPE_TABLE(避免 import controller 依賴)。"""
    import re
    txt = (REPO / "controllers" / "ycb_supervisor" / "config.py").read_text()
    m = re.search(r"SHAPE_TABLE\s*=\s*\{(.*?)\n\}", txt, re.S)
    tab = {}
    if m:
        for k, v in re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', m.group(1)):
            tab[k] = v
    return tab


SHAPE = _shape_table()
OBJS = [n for n in GEO if "size" in GEO[n]]


def size(n):
    s = GEO[n]["size"]; return s["x"], s["y"], s["z"]


def foot_r(n):
    sx, sy, _ = size(n); return 0.5 * math.hypot(sx, sy)


def half_h(n):
    return size(n)[2] / 2.0


def flat(n):
    return n in STACKABLE   # 真平頂/平底,可堆疊


def rest_z(n):
    """桌面靜止時物體中心 z ≈ 半高。"""
    return half_h(n)


def in_ws(x, y):
    return (x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2 <= WS_R ** 2


def place_table(rng, names, existing):
    """在工作空間內為 names 找不重疊(含 margin)的桌面位置;existing=[(x,y,r)]。"""
    placed = list(existing); out = {}
    for n in names:
        r = foot_r(n)
        for _ in range(200):
            a = rng.uniform(0, 2 * math.pi); d = WS_R * math.sqrt(rng.random())
            x, y = CENTER[0] + d * math.cos(a), CENTER[1] + d * math.sin(a)
            if not in_ws(x, y):
                continue
            if all(math.hypot(x - px, y - py) > r + pr + 0.02 for px, py, pr in placed):
                placed.append((x, y, r)); out[n] = (x, y); break
        else:
            return None
    return out, placed


def gen_stack(rng, N):
    """1 個堆疊(底+上)+ (N-2) 桌面物。回傳 objects 或 None。"""
    bases = [n for n in OBJS if flat(n) and foot_r(n) > 0.05]
    rng.shuffle(bases)
    for base in bases:
        bx_sx, bx_sy, _ = size(base)
        tops = [n for n in OBJS if flat(n) and size(n)[0] < bx_sx * 0.85
                and size(n)[1] < bx_sy * 0.85]
        if not tops:
            continue
        top = rng.choice(tops)
        # 底物放工作空間中心附近
        bx, by = CENTER[0] + rng.uniform(-0.05, 0.05), CENTER[1] + rng.uniform(-0.05, 0.05)
        objs = [{"name": base, "position_m": [round(bx, 4), round(by, 4), round(rest_z(base), 4)]},
                {"name": top, "position_m": [round(bx, 4), round(by, 4),
                                             round(size(base)[2] + half_h(top), 4)]}]
        existing = [(bx, by, foot_r(base))]
        rest = [n for n in OBJS if n not in (base, top)]
        rng.shuffle(rest)
        res = place_table(rng, rest[:N - 2], existing)
        if res is None:
            continue
        pos, _ = res
        for n, (x, y) in pos.items():
            objs.append({"name": n, "position_m": [round(x, 4), round(y, 4), round(rest_z(n), 4)]})
        if len(objs) == N:
            return objs
    return None


def gen_occ(rng, N):
    """N 物緊密群聚(小間距 → 多視角互遮),皆桌面。"""
    names = rng.sample(OBJS, N)
    # 緊密:用較小 margin 的桌面擺放(near-touching)
    placed = []; out = []
    cx = CENTER[0] + rng.uniform(-0.03, 0.03); cy = CENTER[1] + rng.uniform(-0.03, 0.03)
    for n in names:
        r = foot_r(n)
        for _ in range(300):
            a = rng.uniform(0, 2 * math.pi); d = rng.uniform(0, 0.04 + 0.022 * N)  # 隨物數放大,仍密
            x, y = cx + d * math.cos(a), cy + d * math.sin(a)
            if not in_ws(x, y):
                continue
            if all(math.hypot(x - px, y - py) > r + pr - 0.02 for px, py, pr in placed):  # 近觸(更密)
                placed.append((x, y, r)); out.append({"name": n,
                    "position_m": [round(x, 4), round(y, 4), round(rest_z(n), 4)]}); break
        else:
            return None
    return out if len(out) == N else None


def shared_viewpoints():
    """沿用現有 multi_scene_plan 的 12 共用視角(look-at 中心相同)。"""
    d = json.loads((PLANS / "multi_scene_plan.json").read_text())
    return d["scenes"][0]["viewpoints"]


def build(kind, per, n_list, seed, vps):
    rng = random.Random(seed)
    scenes = []
    for N in n_list:
        cnt = 0; tries = 0
        while cnt < per and tries < per * 50:
            tries += 1
            objs = gen_stack(rng, N) if kind == "stack" else gen_occ(rng, N)
            if objs:
                cnt += 1
                scenes.append({"scene_name": f"{kind}{N}_scene{cnt:04d}",
                               "objects": objs, "viewpoints": vps})
    return scenes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=20, help="每個物數每類場景數")
    ap.add_argument("--n", type=int, nargs="+", default=[3, 4, 5], dest="n_list")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    PLANS.mkdir(parents=True, exist_ok=True)
    vps = shared_viewpoints()
    for kind in ("stack", "occ"):
        scenes = build(kind, args.per, args.n_list, args.seed, vps)
        out = PLANS / f"{kind}_scene_plan.json"
        out.write_text(json.dumps({"scenes": scenes}, indent=2, ensure_ascii=False), encoding="utf-8")
        byN = {}
        for s in scenes:
            byN[s["objects"].__len__()] = byN.get(len(s["objects"]), 0) + 1
        print(f"[{kind}] {len(scenes)} 場景 (各物數 {byN}) → {out}")


if __name__ == "__main__":
    main()
