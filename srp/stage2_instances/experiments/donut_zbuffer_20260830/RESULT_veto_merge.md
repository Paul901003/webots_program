# veto 組合(語意+表面深度)合併 — 試過,輸給現有 crossing(2026-08-31)

## 動機
平面階梯(前表面穩健平面階梯)在理想 GT 遮罩上分同/異物 per-scene AUC 0.972。想把它 + z重疊 + 空間連通組成 veto 式合併,把 CLIP 過切的同物併回。

## 方法(veto_merge.py)
對每個 CLIP 相鄰 instance 對,**預設 MERGE**,任一 veto 命中就 KEEP-SEPARATE:
① 空間不連通(voxel 3D 局部 bbox dilation 不相接)② z 高度分離(z重疊<0.3)③ 深度牆(平面階梯 min_step>40mm)。union-find 合併。

## 結果(全 303 多物場,同一 clean/split/lost 評分,排 GLOBAL_EXCLUDE)
| 方法 | 過切比 | clean | split | lost | 鬼影/混 |
|---|--:|--:|--:|--:|--:|
| CLIP-only 未合併 | 2.15 | 54% | 46% | 0% | 330 |
| **crossing 現有** | 1.12 | **90%** | 5% | 4% | 113 |
| veto 組合 | 1.26 | 86% | 13% | 2% | 129 |

## 結論(事實)
- **veto 組合 clean 86% < crossing 90%、split 13% > 5%,輸了。** crossing(群A voxel 投影落在群B遮罩比例,τ=0.1)更好也更簡單。
- veto 唯一略勝:lost 2%<4%(較不過度合併)。
- **敗因**:深度牆 veto(③)把「箱面/曲面自遮擋」的同物對擋住(群層級量到同物誤拆 42%)→ 欠合、split 高。平面階梯的強(理想 GT)搬到真實碎片就被自遮擋侵蝕。
- **裁決:接受 crossing 為目前最佳合併**(`srp_hull_semcluster_surf_am1photo_merged`)。veto 這條保留為負結果,`srp_hull_veto_merge` 資料留存供對照。

## 這串深度實驗的整體價值(留存)
- 平面階梯(`plane_step.py`)是很強的**成對**同/異物訊號(理想 GT per-scene 0.972、per-view 0.908,解長物誤拆 30→9%),但**在真實過切碎片的合併決策上不敵 crossing**。
- 深度表面訊號要對合併有用,卡在「曲面/自遮擋同物」與「相觸/同深度異物」兩個幾何死角(見 nonoverlap RESULT + group_pairs.csv)。
