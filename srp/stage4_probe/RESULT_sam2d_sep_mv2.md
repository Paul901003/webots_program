# MobileSAMv2 遮罩在 2D 有沒有把堆疊上下物切開(IoU≥0.5)

- 建檔 2026-09-25;程式 `srp/stage4_probe/sam2d_sep_mv2.py`;可復現。
- SAM=`mobilesamv2_fast`(經 `kept_object_masks` 過濾,與管線一致);GT=labels/<場>/actual(modal);12 視角 A-3。
- 母體:29 個 on 對 × 12 視角,有效判定 **345** 個(視角,對);**排除 GEX**。
- 分開=T/B 最佳匹配是不同遮罩且兩者 IoU≥thr;併=同一個遮罩;漏=至少一方無 IoU≥thr 的遮罩。
- ⚠ 既有 `diag_stack_sep.py` 用舊 `sam_only` 遮罩,結果 84%/1%/15%(REPORT_relation_separation.md §6.1),基準不同。

## 整體

| 判定 | 次數 | 佔比 |
|---|---|---|
| 分開 | 300 | 87.0% |
| 併 | 0 | 0.0% |
| 漏 | 45 | 13.0% |

(另有 3 個(視角,對)因該視角缺 GT modal 遮罩而未判定)

## match_eval 判定「沒分開」的對(重點)

| 場景 | 上物 T | 下物 B | 分開 | 併 | 漏 | 無GT |
|---|---|---|---|---|---|---|
| stack3_scene0005 | foam_brick | gelatin_box | **11** | 0 | 1 | 0 |
| stack4_scene0007 | sugar_box | sponge | **12** | 0 | 0 | 0 |
| stack4_scene0010 | tuna_fish_can | master_chef_can | **9** | 0 | 3 | 0 |

## 逐對明細(依「分開」由少到多,前 15)

| 場景 | 上物 T | 下物 B | 分開 | 併 | 漏 | 無GT |
|---|---|---|---|---|---|---|
| stack5_scene0014 | sugar_box | cracker_box | 7 | 0 | 5 | 0 |
| stack5_scene0016 | tomato_soup_can | wood_block | 8 | 0 | 4 | 0 |
| stack4_scene0010 | tuna_fish_can | master_chef_can | 9 | 0 | 3 | 0 |
| stack4_scene0014 | sugar_box | sponge | 9 | 0 | 2 | 1 |
| stack4_scene0018 | tomato_soup_can | tuna_fish_can | 9 | 0 | 3 | 0 |
| stack3_scene0013 | foam_brick | sponge | 10 | 0 | 2 | 0 |
| stack4_scene0009 | master_chef_can | pudding_box | 10 | 0 | 1 | 1 |
| stack4_scene0017 | foam_brick | master_chef_can | 10 | 0 | 2 | 0 |
| stack5_scene0005 | tuna_fish_can | pudding_box | 10 | 0 | 2 | 0 |
| stack5_scene0011 | foam_brick | sponge | 10 | 0 | 1 | 1 |
| stack5_scene0013 | foam_brick | sponge | 10 | 0 | 2 | 0 |
| stack5_scene0019 | sugar_box | cracker_box | 10 | 0 | 2 | 0 |
| stack5_scene0020 | tuna_fish_can | pudding_box | 10 | 0 | 2 | 0 |
| stack3_scene0005 | foam_brick | gelatin_box | 11 | 0 | 1 | 0 |
| stack3_scene0006 | tomato_soup_can | tuna_fish_can | 11 | 0 | 1 | 0 |