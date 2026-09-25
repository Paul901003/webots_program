# 遮罩層級 voxel 重疊圖:相觸堆疊物能不能被切開(min_ov=1, gt_thr=0.5)

- 建檔 2026-09-25;程式 `srp/stage4_probe/mask_overlap_graph.py`;可復現。
- SAM=`mobilesamv2_fast`(kept_object_masks + DROP_ARM,與管線同源);hull=`srp_hull_mv2_v12_am1`;12 視角 A-3。
- 節點=遮罩,邊=兩遮罩認領的 voxel 集合交集 ≥ min_ov;取連通元件。**純空間,不含語意**。
- 母體:3 場、**3 個 on 對**(排 GEX);物體標籤由 GT modal 遮罩 IoU≥0.5 決定。

## 核心結果:on 對能不能被切開

| 判定 | 對數 | 佔比 |
|---|---|---|
| **不同元件(可切開)** | 0 | 0.0% |
| 同元件(切不開) | 3 | 100.0% |

## 過切風險:同一物體的遮罩被切成幾個元件

| 統計 | 值 |
|---|---|
| 物體數(非GEX) | 10 |
| 平均元件數 | 1.10 |
| 中位 | 1 |
| =1(完整一團) | 9 (90.0%) |
| ≥3 | 0 (0.0%) |
| 最大 | 2 |

## 逐對明細

| 場景 | 上物 T | 下物 B | T遮罩數 | B遮罩數 | T元件數 | B元件數 | 判定 |
|---|---|---|---|---|---|---|---|
| stack3_scene0005 | foam_brick | gelatin_box | 14 | 3 | 1 | 2 | 同元件(切不開) |
| stack4_scene0007 | sugar_box | sponge | 17 | 11 | 1 | 1 | 同元件(切不開) |
| stack4_scene0010 | tuna_fish_can | master_chef_can | 8 | 9 | 1 | 1 | 同元件(切不開) |