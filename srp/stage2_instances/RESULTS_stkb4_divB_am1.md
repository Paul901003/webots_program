# stkb4 divB θ.5(am1,無光雕)兩軸結果 — 2026-09-10

**重現指令(同一套 srp 腳本,只設 env;無 wrapper、無 monkeypatch)**:
```bash
PYW=/home/cho/.pyenv/versions/webots_visual_hull/bin/python3
S=$(ls -d data/captures_fast/multi_stkb4/stkb4_scene*/ | xargs -n1 basename)
# 過切軸
HULL_ROOT_NAME=srp_hull_mv2_v12_am1 $PYW srp/stage2_instances/eval_surface_iou.py \
  --roots srp_hull_semcluster_surf_donut,srp_hull_divB_t50_am1 $S
# 堆疊軸
HULL_ROOT_NAME=srp_hull_mv2_v12_am1 OBS_ONLY=1 $PYW srp/stage2_instances/d2_stack_clean.py \
  --roots srp_hull_semcluster_surf_donut,srp_hull_divB_t50_am1 $S
```
hull=srp_hull_mv2_v12_am1(非photo)、surf_donut 12視角/sem_thr0.4/DROP_ARM(canonical arm)/donut/min_vox50、merge θ.5。
⚠ GT OCC_THRESH:stkb GT 把重遮擋物(0 modal)算進分母,兩軸數字偏保守。

## 過切軸 eval_surface_iou(GT物體 1534)
| found@ | baseline surf_donut | divB θ.5 am1 |
|---|---|---|
| 0.3 | 0.960 | 0.868 |
| 0.5 | 0.816 | 0.824 |
| 0.7 | 0.531 | 0.676 |
| 配對mIoU | 0.688 | 0.690 |
| 預測inst / 幻影 | 2941 / 1412 | 1550 / 158 |

## 堆疊軸 d2_stack_clean(可評on 607,OBS_ONLY)
| success@ | baseline surf_donut | divB θ.5 am1 |
|---|---|---|
| 0.3 | 0.90 | 0.66 |
| 0.5 | 0.67 | 0.61 |
| 0.7 | 0.32 | 0.52 |
| 分開 | 567/607 | 403/607 |

**裁決**:divB 在 stkb4 過度合併(併掉 164 對堆疊 567→403),@0.5 堆疊反輸 baseline(0.67→0.61);過切收幻影(1412→158)但 found@0.3 退步。θ.5 在平衡集偏低,需重掃 θ 或改「不讓碎片當橋」。
