# donut(去大遮罩) vs 原 semcluster 連通前語意群純度

組別: ['n3', 'n4', 'n5', 'occ3', 'occ4', 'occ5', 'stack3', 'stack4', 'stack5'] | 同 hull(srp_hull_mv2_v12_am1)+SAM(mobilesamv2_fast),基準一致

## 原 semcluster (srp_hull_semcluster_clip_am1) — 303 場
- homogeneity(群純度,↑好): min=0.36 中位=0.80 max=1.00 平均=0.79
- completeness(物體完整,↑好): min=0.16 中位=0.52 max=1.00 平均=0.52
- 每群含GT物體數(1=純,>1=混): min=1.00 中位=1.00 max=5.00 平均=1.37  混群(>1)佔 27%
- 每GT物體群數(1=不拆,>1=多群): min=1.00 中位=3.00 max=14.00 平均=3.46  被拆(>1)佔 83%

## donut 去大遮罩 (srp_hull_semcluster_donut) — 303 場
- homogeneity(群純度,↑好): min=0.30 中位=0.74 max=1.00 平均=0.74
- completeness(物體完整,↑好): min=0.16 中位=0.43 max=0.90 平均=0.44
- 每群含GT物體數(1=純,>1=混): min=1.00 中位=1.00 max=5.00 平均=1.45  混群(>1)佔 31%
- 每GT物體群數(1=不拆,>1=多群): min=1.00 中位=4.00 max=19.00 平均=4.69  被拆(>1)佔 90%
