#!/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
"""check_capture.py — 驗證平衡資料集 nb/occb/stkb 拍攝完整性 + 回報續跑點。

比對「該拍的(plan)」vs「已拍的(captures_fast)」,每場分類:
  ok       : 12 視角齊 + scene_manifest.json(視角名與 srp selected_view_names(12) 一致)
  partial  : manifest 存在但視角 <12 或名字不齊 → skip-guard 會誤跳,需 FORCE=1 重拍(★列出)
  broken   : 有 view 但無 manifest(拍到一半掛)→ 重跑會自動重拍(無需 FORCE)
  missing  : 還沒拍
輸出各組統計 + 要處理的清單(partial/broken)+ 建議續跑指令。
用法: ./check_capture.py [nb occb stkb ...]   (空=全部)
偵測停滯: --stall-min N(最新檔 N 分鐘沒更新 → 標記可能卡住,exit code 2)
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "srp" / "io"))
import viewpoints as VP   # noqa

PLANS = REPO / "data" / "scene_plans"
CAPS = REPO / "data" / "captures_fast"
WANT = VP.selected_view_names(12)   # 該有的 12 視角名


def scene_state(scene):
    group = scene.split("_")[0]
    d = CAPS / f"multi_{group}" / scene
    if not d.is_dir():
        return "missing", 0, 0.0
    views = {os.path.basename(p)[:-4] for p in glob.glob(str(d / "view_*.png")) if "depth" not in p}
    man = d / "scene_manifest.json"
    mtime = max([os.path.getmtime(p) for p in glob.glob(str(d / "*"))] or [0])
    if man.is_file():
        if views >= WANT:
            return "ok", len(views), mtime
        return "partial", len(views), mtime       # manifest 但視角不齊 → 需 FORCE
    if views:
        return "broken", len(views), mtime        # 有圖無 manifest → 重跑自動重拍
    return "missing", 0, mtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("groups", nargs="*", help="nb occb stkb(空=全部)")
    ap.add_argument("--stall-min", type=float, default=0, help="最新檔 N 分鐘沒更新 → 標記卡住")
    args = ap.parse_args()

    plan_files = {"nb": "nb_scene_plan.json", "occb": "occb_scene_plan.json", "stkb": "stkb_scene_plan.json"}
    want_groups = args.groups or list(plan_files)
    latest_all = 0.0
    total = {"ok": 0, "partial": 0, "broken": 0, "missing": 0}
    fix_partial, fix_broken = [], []

    for g in want_groups:
        pf = PLANS / plan_files[g]
        if not pf.is_file():
            print(f"[{g}] 無 plan {pf.name}"); continue
        scenes = [s["scene_name"] for s in json.loads(pf.read_text())["scenes"]]
        c = {"ok": 0, "partial": 0, "broken": 0, "missing": 0}
        for sc in scenes:
            st, nv, mt = scene_state(sc)
            c[st] += 1; total[st] += 1
            latest_all = max(latest_all, mt)
            if st == "partial":
                fix_partial.append(sc)
            elif st == "broken":
                fix_broken.append(sc)
        done = c["ok"]; n = len(scenes)
        print(f"[{g}] {n} 場: ✓ok {c['ok']}  ⚠partial {c['partial']}  ✗broken {c['broken']}  ·missing {c['missing']}"
              f"   ({done/n*100:.1f}% 完成)")

    print(f"\n合計: ok {total['ok']}  partial {total['partial']}  broken {total['broken']}  missing {total['missing']}")

    if fix_partial:
        print(f"\n★ partial(有 manifest 但視角不齊,skip-guard 會誤跳 → 需 FORCE 重拍){len(fix_partial)}:")
        for s in fix_partial[:20]:
            print(f"  {s}")
        if len(fix_partial) > 20:
            print(f"  …共 {len(fix_partial)} 個")
        print(f"  重拍指令: FORCE=1 run_capture_balanced.sh " + " ".join(sorted(set(fix_partial))))
    if fix_broken:
        print(f"\n✗ broken(有圖無 manifest → 直接重跑同組即自動重拍,無需 FORCE){len(fix_broken)}: "
              + ", ".join(fix_broken[:10]) + (" …" if len(fix_broken) > 10 else ""))

    # 續跑:missing/broken 直接重跑該組(skip-guard 跳過 ok);partial 要 FORCE
    remaining = total["missing"] + total["broken"]
    if remaining or fix_partial:
        print(f"\n▶ 續跑:直接重下 run_capture_balanced.sh {' '.join(want_groups)}"
              f"(自動跳過 {total['ok']} 個 ok、重拍 {remaining} 個 missing/broken)")
        if fix_partial:
            print(f"   partial 另需 FORCE=1 單獨重拍(見上)")
    else:
        print(f"\n✓ 全部完成,無需續跑。")

    # 停滯偵測
    if args.stall_min > 0 and latest_all > 0:
        idle_min = (time.time() - latest_all) / 60
        if idle_min > args.stall_min:
            print(f"\n⚠⚠ 停滯:最新檔 {idle_min:.1f} 分鐘沒更新(>{args.stall_min})——可能卡住,查 webots 是否還活著")
            sys.exit(2)
        else:
            print(f"\n最新檔 {idle_min:.1f} 分鐘前更新(活著)")


if __name__ == "__main__":
    main()
