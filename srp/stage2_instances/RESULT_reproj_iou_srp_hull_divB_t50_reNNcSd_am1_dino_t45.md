# 重投影 2D IoU + 溢出(mode=zbuffer):srp_hull_divB_t50_reNNcSd_am1_dino_t45

- 建檔 2026-09-23;程式 `reproj_iou.py --mode zbuffer`;hull=`srp_hull_mv2_v12_am1`;排 GEX;303 多物;可復現。
- 投影=zbuffer(modal);每 instance 對 GT modal 遮罩:IoU(最佳)、precision=|pm∩gt*|/|pm|、
  溢別物=溢到其他非GEX物體、溢背景=溢到所有物體(含GEX)之外。視角=instance 平均、每場=視角平均。

## 分組平均

| 組 | 場數 | IoU | precision | 溢別物% | 溢背景% |
|---|---|---|---|---|---|
| n | 183 | 0.688 | 80.5 | 0.1 | 11.9 |
| occ | 60 | 0.693 | 80.5 | 0.7 | 11.5 |
| stack | 60 | 0.629 | 73.2 | 3.4 | 11.4 |
| all | 303 | 0.677 | 79.1 | 0.9 | 11.7 |