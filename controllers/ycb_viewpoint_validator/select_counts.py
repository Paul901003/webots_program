#!/usr/bin/env python3
"""select_counts.py — 用 A-3 原方法(select_validated_viewpoints)對多種視角數各選一次。

完全重用 select_validated_viewpoints 的:合併去重、選半徑、強制群組(A/B/低仰角)、
最遠點補滿(select_spread_viewpoints)。對 COUNTS 中每個數量各跑一次,輸出
  data/viewpoints/selected_viewpoints_multi_n{count}_x{tag}.json
並印出各數量的視角角度 + 散開度(選集兩兩視線方向的最小夾角,越大越散開)。
用法: /usr/bin/python3 select_counts.py --multi --x-offset 0.35 [--counts 6 8 10 12]
"""
import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import select_validated_viewpoints as S   # 重用原 A-3 全部函式/常數

def az_normalized(v):
    return round(float(v.get("meta", {}).get("azimuth_deg", 0)))


# 方位角群組（正規化 0~360）：中心=180、A=[90,180) 左半、B=(180,270] 右半（互不重疊）
def el_deg(v):
    return abs(float(v.get("meta", {}).get("elevation_deg", 90)))


def in_az_center(v):
    return az_normalized(v) % 360 == 180


def in_az_a(v):
    return 90 <= az_normalized(v) % 360 < 180


def in_az_b(v):
    return 180 < az_normalized(v) % 360 <= 270


def load_validated(multi, x_offset):
    files = S._find_validated_files(multi, x_offset)
    if not files:
        sys.exit(f"找不到 x_offset={x_offset:+.3f} 的 A-2 輸出檔")
    validated, seen = [], set()
    for vf in files:
        for rec in json.loads(vf.read_text(encoding="utf-8")).get("validated", []):
            uid = (rec.get("id"), str(rec.get("source", "")))
            if uid not in seen:
                seen.add(uid); validated.append(rec)
    print(f"[selN] 合併 {len(validated)} 個通過視角(來自 {len(files)} 檔)")
    return validated, files


def pick_radius(validated, count):
    by_r = defaultdict(list)
    for v in validated:
        r = v.get("meta", {}).get("radius_m") or S.planner_config.HEMISPHERE_RADIUS_M
        by_r[round(float(r), 3)].append(v)

    def ok(pool):
        a = any(in_az_a(v) for v in pool)
        b = any(in_az_b(v) for v in pool)
        return a and b
    for r in sorted(by_r, reverse=True):
        if len(by_r[r]) >= count and ok(by_r[r]):
            return r, by_r[r]
    # fallback:視角最多的半徑
    r = max(by_r, key=lambda k: len(by_r[k]))
    return r, by_r[r]


def select_n(pool, count, target_m):
    """強制保留：天頂(el=90) + 中心(az=180,有才留) + 群組 A/B 各挑仰角最小 + 最遠點補滿。"""
    forced, used = [], set()

    def by_el_then_roll(v):
        return (el_deg(v), S.candidate_roll_error(v))   # 先比仰角(小=側視),並列再比 roll

    def force_one(filt, key):
        cand = [v for v in pool if filt(v) and id(v) not in used]
        if cand:
            p = min(cand, key=key)
            forced.append(p); used.add(id(p))

    force_one(lambda v: round(el_deg(v)) == 90, S.candidate_roll_error)  # ① 天頂
    force_one(in_az_center, by_el_then_roll)                             # ② 中心 180(有才留)
    force_one(in_az_a, by_el_then_roll)                                  # ③ 左半 仰角最小
    force_one(in_az_b, by_el_then_roll)                                  # ④ 右半 仰角最小

    rest_pool = [v for v in pool if id(v) not in used]
    rest = S.select_spread_viewpoints(rest_pool, count - len(forced), target_m)
    return forced + rest


def min_pair_angle(selected, target_m):
    """選集兩兩視線方向的最小夾角(度);散開度指標,越大越均勻。"""
    dirs = [S.norm(S.subtract(S.candidate_position(v), target_m)) for v in selected]
    m = 180.0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            m = min(m, S.angle_between(dirs[i], dirs[j]))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multi", action="store_true")
    ap.add_argument("--x-offset", type=float, default=0.0)
    ap.add_argument("--counts", type=int, nargs="+", default=[6, 8, 10, 12])
    args = ap.parse_args()

    oc = list(S.planner_config.OBJECT_CENTER_M)
    target_m = [oc[0] + args.x_offset, oc[1], oc[2]]
    validated, files = load_validated(args.multi, args.x_offset)
    x_tag = f"x{int(args.x_offset*100):+04d}"
    base = "selected_viewpoints_multi" if args.multi else "selected_viewpoints"

    summary = []
    for count in args.counts:
        r, pool = pick_radius(validated, count)
        selected = select_n(pool, count, target_m)
        disp = min_pair_angle(selected, target_m)
        # 輸出檔
        out = S.VIEWPOINTS_DIR / f"{base}_n{count}_{x_tag}.json"
        out.write_text(json.dumps({
            "sources": [str(f) for f in files], "target_m": target_m,
            "x_offset_m": args.x_offset, "radius_m": r,
            "requested_count": count, "selected_count": len(selected),
            "min_pair_angle_deg": disp, "selected": selected,
        }, indent=2), encoding="utf-8")
        # 角度列表
        angs = []
        for v in selected:
            p = S.candidate_position(v); d = S.subtract(p, target_m)
            dist = math.sqrt(S.dot(d, d))
            el = round(math.degrees(math.asin(max(-1, min(1, d[2]/max(dist, 1e-9))))))
            az = round(math.degrees(math.atan2(d[1], d[0])) % 360)
            angs.append((el, az))
        print(f"\n=== count={count}  半徑={r}m  實得 {len(selected)} 個  最小兩兩角距={disp:.1f}° ===")
        for el, az in sorted(angs):
            print(f"    el{el:>2}  az{az:>3}")
        print(f"  → {out.name}")
        summary.append((count, len(selected), disp))

    print(f"\n{'count':>6}{'實得':>6}{'最小兩兩角距(°)':>16}")
    for c, n, dsp in summary:
        print(f"{c:>6}{n:>6}{dsp:>16.1f}")


if __name__ == "__main__":
    main()
