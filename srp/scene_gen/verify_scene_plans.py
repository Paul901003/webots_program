#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""verify_scene_plans.py — 獨立稽核 nb/occb/stkb 場景計畫(重讀 JSON,不信生成器自印)。

檢查(每個違規都列出實例,不只給結論):
  1. 平衡:每物在每 N 的出現次數(min/max/目標);stkb 每配對次數 + 覆蓋。
  2. 半球:每個物體(位置 + footprint + 頂高)是否整個落在 0.35 半球內。
  3. 疊的幾何(stkb):頂與底同 (x,y)、頂 z = 底全高 + 頂半高、頂 footprint ≤ 底頂面。
  4. 水平重疊:z 區間重疊的兩物,footprint 圓不可相交(疊的頂在底之上、z 不重疊 → 不算)。
  5. 池歸屬:nb/occb 用全池;stkb 疊底∈BASE、疊頂∈TOP。
用法: ./verify_scene_plans.py
"""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "controllers" / "ycb_supervisor"))
import config as CFG   # noqa
GEO = json.loads((REPO / "controllers" / "ycb_supervisor" / "ycb_geometries.json").read_text())
PLANS = REPO / "data" / "scene_plans"

R = 0.35; CENTER = (0.35, 0.0)


def size(n):
    s = GEO[n]["size"]; return s["x"], s["y"], s["z"]
def foot_r(n):
    sx, sy, _ = size(n); return 0.5 * math.hypot(sx, sy)
def half_h(n):
    return size(n)[2] / 2.0


def load(name):
    p = PLANS / f"{name}_scene_plan.json"
    return json.loads(p.read_text())["scenes"] if p.is_file() else None


def check_hemi(scenes):
    """回傳越界實例 [(scene, obj, 超出量)]。物體最遠點 sqrt((ρ+fr)²+top²) 要 ≤ R。"""
    bad = []
    for s in scenes:
        for o in s["objects"]:
            n = o["name"]; x, y, z = o["position_m"]
            rho = math.hypot(x - CENTER[0], y - CENTER[1])
            top = z + half_h(n)                     # 物體最高點
            reach = math.hypot(rho + foot_r(n), top)
            if reach > R + 1e-6:
                bad.append((s["scene_name"], n, round(reach - R, 4)))
    return bad


def check_overlap(scenes):
    """z 區間重疊(同層)的兩物 footprint 圓不可相交。回傳違規實例。"""
    bad = []
    for s in scenes:
        objs = s["objects"]
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                a, b = objs[i], objs[j]
                na, nb = a["name"], b["name"]
                ax, ay, az = a["position_m"]; bx, by, bz = b["position_m"]
                za0, za1 = az - half_h(na), az + half_h(na)
                zb0, zb1 = bz - half_h(nb), bz + half_h(nb)
                z_overlap = min(za1, zb1) - max(za0, zb0)
                if z_overlap <= 1e-4:               # 不同層(疊)→ 不算水平重疊
                    continue
                d = math.hypot(ax - bx, ay - by)
                if d < foot_r(na) + foot_r(nb) - 1e-3:
                    bad.append((s["scene_name"], na, nb, round(foot_r(na) + foot_r(nb) - d, 4)))
    return bad


def balance(scenes):
    """每 N 的 per-物出現次數。"""
    byN = defaultdict(Counter)
    for s in scenes:
        byN[len(s["objects"])][None]  # touch
        for o in s["objects"]:
            byN[len(s["objects"])][o["name"]] += 1
    return byN


def report_balance(name, scenes, rmult=8):
    print(f"\n[{name}] {len(scenes)} 場景")
    byN = balance(scenes)
    for N in sorted(k for k in byN if k):
        c = {k: v for k, v in byN[N].items() if k}
        vals = sorted(c.values())
        tgt = rmult * N
        n_scene = sum(vals) // N
        print(f"  N={N}: {n_scene} 場, 出現物 {len(c)} 種, 目標{tgt}, 實際 {vals[0]}~{vals[-1]}, "
              f"{'✓齊' if vals[0]==vals[-1]==tgt else ('近齊' if vals[-1]-vals[0]<=1 else '⚠不齊')}")


def report_stack(scenes):
    print(f"\n[stkb 疊的幾何 + 配對平衡]")
    pair_cnt = Counter(); geo_bad = []; pool_bad = []
    BASE = set(CFG.STACK_BASE); TOP = set(CFG.STACK_TOP)
    byP = Counter()
    for s in scenes:
        P = s.get("n_piles"); byP[P] += 1
        pos = {o["name"]: o["position_m"] for o in s["objects"]}
        for b, t in s.get("pairs", []):
            pair_cnt[(b, t)] += 1
            if b not in BASE:
                pool_bad.append((s["scene_name"], f"底{b}∉BASE"))
            if t not in TOP:
                pool_bad.append((s["scene_name"], f"頂{t}∉TOP"))
            if b in pos and t in pos:
                bx, by, bz = pos[b]; tx, ty, tz = pos[t]
                if abs(bx - tx) > 1e-3 or abs(by - ty) > 1e-3:
                    geo_bad.append((s["scene_name"], f"{t}未置中於{b}"))
                z_expect = size(b)[2] + half_h(t)
                if abs(tz - z_expect) > 1e-3:
                    geo_bad.append((s["scene_name"], f"{t} z={tz}≠{round(z_expect,4)}"))
                tsx, tsy, _ = size(t); bsx, bsy, _ = size(b)
                if not ((tsx <= bsx and tsy <= bsy) or (tsy <= bsx and tsx <= bsy)):
                    geo_bad.append((s["scene_name"], f"{t} footprint>{b}頂面"))
    print(f"  疊數分布 P: " + " ".join(f"P{p}={byP[p]}" for p in sorted(byP)))
    vals = list(pair_cnt.values())
    print(f"  配對: 覆蓋 {len(pair_cnt)} 對, 每對 {min(vals)}~{max(vals)} 次 (平均 {sum(vals)/len(vals):.1f})")
    print(f"  幾何違規: {len(geo_bad)}  池歸屬違規: {len(pool_bad)}")
    for x in (geo_bad + pool_bad)[:8]:
        print(f"    {x}")


def main():
    for name in ("nb", "occb", "stkb"):
        scenes = load(name)
        if scenes is None:
            print(f"[{name}] 無 plan 檔"); continue
        report_balance(name, scenes)
        hemi = check_hemi(scenes)
        ov = check_overlap(scenes)
        print(f"  半球越界: {len(hemi)}  水平重疊: {len(ov)}")
        for x in (hemi[:4]):
            print(f"    越界 {x}")
        for x in (ov[:4]):
            print(f"    重疊 {x}")
        if name == "stkb":
            report_stack(scenes)


if __name__ == "__main__":
    main()
