# 重投影 2D IoU(mode=direct):srp_hull_divB_t50_reNNcSd_am1

- 建檔 2026-09-23;程式 `reproj_iou.py --mode direct`;hull=`srp_hull_mv2_v12_am1`;排 GEX;303 多物;可復現。
- 投影=direct(整群footprint直接投影,不遮擋);每 instance 取對 GT modal 遮罩最佳 IoU;視角=instance 平均;每場=視角平均。

## 分組平均 IoU

| 組 | 場數 | 平均每場 IoU |
|---|---|---|
| n | 183 | 0.690 |
| occ | 60 | 0.690 |
| stack | 60 | 0.632 |
| all | 303 | 0.678 |