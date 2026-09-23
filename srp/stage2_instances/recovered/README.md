# recovered/ — 從 job-tmp(session fe965b39)撿回的重要腳本(2026-09-16)

⚠ 這些是當初一次性跑的腳本(硬編路徑、從 repo 根執行、sys.path 指 srp/),
撿回落地避免遺失;**未整理成正式介面**,當「怎麼算的」的參考。正式版見上層目錄。

## 必救(指標/建置器)
- build_gtlabel_all.py — gtlabel 建置器原版(已被上層 `build_gtlabel.py` 取代、參數化)。
- perobj_leak.py / nosep_full.py / onpair_fix.py — per-object 洩漏% / 沒分開率 原版(已被 `stack_leak_nosep.py` 取代)。

## 中(可復現)
- morph_grid.py / morph_grid618.py / morph_open.py — 膨脹侵蝕掃描(產記憶 morphology-no-help-hull 的數字)。
- eval_solid_4lines.py / eval_1.py / eval_23.py — found@/mIoU 物理正確 eval(產「已作廢」的 RESULT_hull_vote_reassign_eval 表)。
- purity_metrics.py / gthull_pr.py / gtlabel_cmp.py — precision/recall vs GT 完美標籤 輔助指標。
