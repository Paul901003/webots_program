# 重投影 2D IoU(每 instance vs GT modal 遮罩,每場平均):srp_hull_divB_t50_reNNcSd_am1

- 建檔 2026-09-23;程式 `reproj_iou.py`;hull=`srp_hull_mv2_v12_am1`;排 GEX;303 多物;可復現。
- 每視角 zbuffer→pred群影像;每 instance 取對 GT 遮罩最佳 IoU;視角=instance 平均;每場=視角平均。

## 分組平均 IoU

| 組 | 場數 | 平均每場 IoU |
|---|---|---|
| n | 183 | 0.677 |
| occ | 60 | 0.684 |
| stack | 60 | 0.636 |
| all | 303 | 0.670 |