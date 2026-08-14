#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""gen_relation_balanced.py — 平衡關係場景:堆疊 stkb(產 on)+ 密集遮擋 occb(產 blocks_access)。
獨立新檔,不動舊 gen_relation_scenes.py。

★ 擺放機制逐字沿用舊 gen_relation_scenes.py:CENTER=(0.35,0)、WS_R=0.20 **2D 平圓盤**、
  foot_r=對角線/2、place_table 不重疊、gen_occ 密集群聚(d=0.04+0.022N,近觸)。只改「數量/需求」:
    · 物體池排除 GLOBAL_EXCLUDE(4)
    · N = 3/4/5/6(舊 3/4/5)
    · occb 物體平衡:每物在每 N 出現 R=8N 次(舊為隨機 sample);沿用舊 n 生成器的
      「零容忍移除迴圈」處理密集塞不下的巨物(移除後對乾淨池平衡)
    · stkb 用 mesh 池 STACK_BASE(19)/STACK_TOP(32)(舊為 STACKABLE 13 白名單);
      配對平衡:每個 (底,頂) 在每個 P 配置出現 K=P 次;P=1/2/3 疊(舊只有 1 疊)
    · 輸出 occb/stkb_scene_plan.json、場景名 occb{N}/stkb{N}(不覆蓋舊 occ/stack)
  頂物 fit 判定沿用舊 gen_stack 準則:top 兩邊 < base×0.85。
用法: ./srp/scene_gen/gen_relation_balanced.py [--n 3 4 5 6] [--seed 42] [--r-mult 8]
"""
import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "controllers" / "ycb_supervisor"))
import config as CFG                       # noqa: E402(純資料)

GEO = json.loads((REPO / "controllers" / "ycb_supervisor" / "ycb_geometries.json").read_text())
PLANS = REPO / "data" / "scene_plans"

CENTER = (0.35, 0.0)          # look-at 中心
WS_R = 0.35                    # 改(依指示):邊界改用半徑 0.35 的半球(含高度),取代舊 0.20 平圓盤

OBJS = [n for n in GEO if "size" in GEO[n] and n not in CFG.GLOBAL_EXCLUDE]   # 改:排除 4
BASE_POOL = [n for n in CFG.STACK_BASE if n in GEO]                            # 改:mesh 池
TOP_POOL = [n for n in CFG.STACK_TOP if n in GEO]


def size(n):
    s = GEO[n]["size"]; return s["x"], s["y"], s["z"]


def foot_r(n):
    sx, sy, _ = size(n); return 0.5 * math.hypot(sx, sy)   # 對角線/2


def half_h(n):
    return size(n)[2] / 2.0


def rest_z(n):
    return half_h(n)


def hemi_max_r(n):
    """物體(footprint 半徑 fr、頂高 h)中心離球心最大水平距離,使整體在 0.35 半球內:
    √((r+fr)²+h²)≤WS_R → r≤√(WS_R²−h²)−fr。<0=整體塞不進。"""
    fr = foot_r(n); h = size(n)[2]
    inner = WS_R ** 2 - h ** 2
    return math.sqrt(inner) - fr if inner > 0 else -1.0


def in_ws(x, y, n):
    """改:半球約束(含高度)。物體 n 中心在 (x,y),整體須在 0.35 半球內。"""
    d = math.hypot(x - CENTER[0], y - CENTER[1])
    return d <= hemi_max_r(n)


def place_table(rng, names, existing):
    """半球內不重疊(margin 0.02);每物在自身 hemi_max_r 內取樣(含高度)。舊擺放邏輯,僅邊界換半球。"""
    placed = list(existing); out = {}
    for n in names:
        r = foot_r(n); mr = hemi_max_r(n)
        if mr < 0:
            return None
        for _ in range(200):
            a = rng.uniform(0, 2 * math.pi); d = mr * math.sqrt(rng.random())   # d≤mr ⇒ 在半球內
            x, y = CENTER[0] + d * math.cos(a), CENTER[1] + d * math.sin(a)
            if all(math.hypot(x - px, y - py) > r + pr + 0.02 for px, py, pr in placed):
                placed.append((x, y, r)); out[n] = (x, y); break
        else:
            return None
    return out, placed


# ── 物體平衡組合(舊 n 生成器逐字)────────────────────────────────────────────
def build_combos(pool, n, appear, rng):
    if len(pool) < n:
        raise ValueError(f"物體數 {len(pool)} 不足以組 {n}")
    need = {o: appear for o in pool}; used = set(); combos = []
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
    return combos


# ── occb:密集群聚(舊 gen_occ 擺放機制,給定 names)────────────────────────────
def place_occ(rng, names):
    """密集群聚(遮擋);擺給定 names。群聚半徑 d 自適應(選項 a):
    小物 → 維持舊 0.04+0.022N 的緊密;大物 → 放大到剛好容得下 N 個 footprint(仍群聚)。失敗回 None。"""
    N = len(names); placed = []; out = []
    old_d = 0.04 + 0.022 * N
    # 能容 N 個 footprint 圓的半徑(近觸隨機擺放填充率~0.5,×1.15 餘裕)
    pack_d = 1.15 * math.sqrt(sum(foot_r(m) ** 2 for m in names) / 0.5)
    d_cluster = max(old_d, pack_d)
    cx = CENTER[0] + rng.uniform(-0.03, 0.03); cy = CENTER[1] + rng.uniform(-0.03, 0.03)
    for n in names:
        r = foot_r(n)
        for _ in range(300):
            a = rng.uniform(0, 2 * math.pi); d = rng.uniform(0, d_cluster)          # 自適應密集半徑
            x, y = cx + d * math.cos(a), cy + d * math.sin(a)
            if not in_ws(x, y, n):          # 半球邊界(含高度)
                continue
            if all(math.hypot(x - px, y - py) > r + pr - 0.02 for px, py, pr in placed):
                placed.append((x, y, r))
                out.append({"name": n, "position_m": [round(x, 4), round(y, 4), round(rest_z(n), 4)]})
                break
        else:
            return None
    return out if len(out) == N else None


def build_occb(n_list, rmult, seed, vps):
    """物體平衡 + 舊 n 生成器的零容忍移除迴圈(密集塞不下的巨物→全域移除→重生)。"""
    def once(pool):
        rng = random.Random(seed)
        scenes = defaultdict(list); failed = []
        for N in n_list:
            for names in build_combos(pool, N, rmult * N, rng):
                objs = place_occ(rng, list(names))
                if objs is None:
                    failed.append(names)
                else:
                    scenes[N].append(objs)
        return scenes, failed

    pool = list(OBJS); removed = []
    while True:
        scenes, failed = once(pool)
        if not failed:
            break
        victim = max({nm for names in failed for nm in names},
                     key=lambda nm: max(size(nm)[0], size(nm)[1]))
        removed.append(victim); pool.remove(victim)
    out = []
    for N in n_list:
        for i, objs in enumerate(scenes[N], 1):
            out.append({"scene_name": f"occb{N}_scene{i:04d}", "objects": objs, "viewpoints": vps})
    return out, removed, {N: len(scenes[N]) for N in n_list}, pool


# ── stkb:配對平衡 + 1/2/3 疊(舊 gen_stack 擺放機制,擴多疊)────────────────────
def fits_on(top, base):
    """舊 gen_stack 準則:top 兩邊 < base×0.85(不轉)。"""
    tx, ty, _ = size(top); bx, by, _ = size(base)
    return tx < bx * 0.85 and ty < by * 0.85


def pile_fits_hemi(base, top):
    """整疊(底全高+頂全高)含 footprint 要塞進 0.35 半球,否則頂會戳出半球/settle 時倒
    (如 cracker_box 21 + sugar_box 18 = 39cm > 35cm)。與所選半球邊界一致。"""
    H = size(base)[2] + size(top)[2]
    inner = WS_R ** 2 - H ** 2
    return inner > 0 and math.sqrt(inner) - foot_r(base) >= 0


def valid_pairs():
    return [(b, t) for b in BASE_POOL for t in TOP_POOL
            if t != b and fits_on(t, b) and pile_fits_hemi(b, t)]


def stack_scene_objects(rng, piles, n_table, tbl):
    """P 疊 + n_table 桌物,舊 place_table 擺放(2D 圓盤)。失敗回 None。"""
    bases = [b for b, _ in piles]
    pile_objs = set()
    for b, t in piles:
        pile_objs |= {b, t}
    table_names = tbl.take(n_table, pile_objs, rng) if n_table > 0 else []
    for _ in range(12):
        res = place_table(rng, bases + table_names, [])
        if res is not None:
            break
    else:
        return None
    pos, _ = res
    objs = []
    for b, t in piles:
        bx, by = pos[b]
        objs.append({"name": b, "position_m": [round(bx, 4), round(by, 4), round(rest_z(b), 4)]})
        objs.append({"name": t, "position_m": [round(bx, 4), round(by, 4),
                                               round(size(b)[2] + half_h(t), 4)]})
    for nm in table_names:
        x, y = pos[nm]
        objs.append({"name": nm, "position_m": [round(x, 4), round(y, 4), round(rest_z(nm), 4)]})
    return objs


class TableBalancer:
    def __init__(self, pool):
        self.need = Counter({o: 0 for o in pool}); self.pool = list(pool)

    def take(self, k, exclude, rng):
        cands = sorted((o for o in self.pool if o not in exclude),
                       key=lambda o: (self.need[o], rng.random()))
        pick = cands[:k]
        for o in pick:
            self.need[o] += 1
        return pick


def pack_into_scenes(pile_pool, P):
    pool = list(pile_pool); scenes = []
    while len(pool) >= P:
        used = set(); chosen = []; rest = []
        for pile in pool:
            b, t = pile
            if len(chosen) < P and b not in used and t not in used:
                chosen.append(pile); used |= {b, t}
            else:
                rest.append(pile)
        if len(chosen) == P:
            scenes.append(chosen); pool = rest
        else:
            break
    return scenes


def build_stkb(n_list, seed, vps):
    rng = random.Random(seed)
    pairs = valid_pairs(); M = len(pairs)
    tbl = TableBalancer(OBJS)
    by_N = defaultdict(list); cell = {}; pair_cnt = Counter()
    for P in (1, 2, 3):
        for N in n_list:
            if 2 * P > N:
                continue
            pile_pool = []
            for pr in pairs:
                pile_pool += [pr] * (2 * P)       # K=2P(拉齊三組~1650)
            rng.shuffle(pile_pool)
            made = 0
            for piles in pack_into_scenes(pile_pool, P):
                objs = stack_scene_objects(rng, piles, N - 2 * P, tbl)
                if objs is None:
                    continue
                for b, t in piles:
                    pair_cnt[(b, t)] += 1
                by_N[N].append({"n_piles": P, "pairs": [[b, t] for b, t in piles], "objects": objs})
                made += 1
            cell[(P, N)] = made
    scenes = []
    for N in n_list:
        for i, sc in enumerate(by_N[N], 1):
            scenes.append({"scene_name": f"stkb{N}_scene{i:04d}", "n_piles": sc["n_piles"],
                           "pairs": sc["pairs"], "objects": sc["objects"], "viewpoints": vps})
    return scenes, M, cell, pair_cnt


def shared_viewpoints():
    d = json.loads((PLANS / "multi_scene_plan.json").read_text())
    return d["scenes"][0]["viewpoints"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[3, 4, 5, 6], dest="n_list")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--r-mult", type=int, default=10, dest="rmult")   # occb x=10(拉齊三組~1650)
    ap.add_argument("--only", choices=["occb", "stkb"], default=None)
    args = ap.parse_args()
    PLANS.mkdir(parents=True, exist_ok=True)
    vps = shared_viewpoints()
    print(f"OBJS(全池−GLOBAL_EXCLUDE)={len(OBJS)}  BASE={len(BASE_POOL)}  TOP={len(TOP_POOL)}")

    if args.only != "stkb":
        scenes, removed, stats, pool = build_occb(args.n_list, args.rmult, args.seed, vps)
        (PLANS / "occb_scene_plan.json").write_text(
            json.dumps({"scenes": scenes}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[occb] {len(scenes)} 場景  最終池 {len(pool)}(密集移除 {len(removed)}: {', '.join(removed) or '無'})")
        for N in args.n_list:
            print(f"  occb{N}: {stats[N]} 場 (每物 {args.rmult*N} 次)")

    if args.only != "occb":
        scenes, M, cell, pair_cnt = build_stkb(args.n_list, args.seed, vps)
        (PLANS / "stkb_scene_plan.json").write_text(
            json.dumps({"scenes": scenes}, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[stkb] {len(scenes)} 場景  合法配對 M={M}")
        for P in (1, 2, 3):
            cs = [(N, cell[(P, N)]) for N in args.n_list if (P, N) in cell]
            print(f"  P={P}: " + " ".join(f"N{N}={c}" for N, c in cs) + f"  (K=2P={2*P},每格目標 {2*M})")
        if pair_cnt:
            v = list(pair_cnt.values())
            print(f"  配對覆蓋 {len(pair_cnt)}/{M},每對 {min(v)}~{max(v)} 次")


if __name__ == "__main__":
    main()
