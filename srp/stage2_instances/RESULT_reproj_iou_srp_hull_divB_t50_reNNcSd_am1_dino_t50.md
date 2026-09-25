# 重投影 2D IoU + 溢出(mode=zbuffer):srp_hull_divB_t50_reNNcSd_am1_dino_t50

- 建檔 2026-09-23;程式 `reproj_iou.py --mode zbuffer`;hull=`srp_hull_mv2_v12_am1`;排 GEX;303 多物;可復現。
- 投影=zbuffer(modal);每 instance 對 GT modal 遮罩:IoU(最佳)、precision=|pm∩gt*|/|pm|、
  溢別物=溢到其他非GEX物體、溢背景=溢到所有物體(含GEX)之外。視角=instance 平均、每場=視角平均。

## 分組平均

| 組 | 場數 | IoU | precision | 溢別物% | 溢背景% |
|---|---|---|---|---|---|
| n | 183 | 0.714 | 80.9 | 0.1 | 11.4 |
| occ | 60 | 0.698 | 80.5 | 0.7 | 11.4 |
| stack | 60 | 0.634 | 73.3 | 3.7 | 11.4 |
| all | 303 | 0.695 | 79.3 | 1.0 | 11.4 |