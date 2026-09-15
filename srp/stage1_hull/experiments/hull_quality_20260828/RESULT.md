# Hull 品質評估:am0/am1 × 光度雕 × 語意重疊雕 vs GT 實心 mesh

日期 2026-08-28。60 場堆疊(stack3/4/5)。基準全用 **MobileSAMv2**(`mobilesamv2_fast`),與主線一致。

## 目的
比較各 hull 變體對 GT 實心 mesh 的品質,選定主線 hull。定義:
- **覆蓋** = 真 mesh 被 hull 蓋到的比例(TP/GT_total);越高越好。
- **切掉** = 真 mesh 沒被蓋到(FN/GT_total)= 1−覆蓋(hull 缺的洞)。
- **過估(ghost)** = hull 有但不在真 mesh 內(FP/hull_total);hull 憑空多出的體積。

## 結果(見 hull_quality.log / hull_quality_am1.log)

| 變體 | 覆蓋 | 切掉 | 過估 |
|------|:---:|:---:|:---:|
| am1(現行,軟 hull) | 89.5% | 10.5% | 19.4% |
| **am1+光度雕** | **88.1%** | 11.9% | **16.3%** |
| am1+語意重疊雕 | 31.2% | 68.8% | 18.9% |
| am1+光度+語意雕 | 28.6% | 71.4% | 15.9% |
| am0(硬交集) | 83.2% | 16.8% | 15.2% |
| am0+光度雕 | 81.2% | 18.8% | 12.9% |
| am0+語意重疊雕 | 25.9% | 74.1% | 17.5% |

## 結論
1. **最佳 = am1+光度雕**:覆蓋只掉 1.4%(89.5→88.1),ghost 卻少 3.1%(19.4→16.3)。近乎白賺的過估縮減(am1 的 ghost 多在遮罩外空曠處,光度雕削那些不太傷真 mesh)。→ 已建成主線 root `data/eval/srp_hull_mv2_v12_am1_photo`(367 場,build_am1_photo.py)。
2. **語意重疊雕(mixed-carve)淘汰**:不管接 am0 或 am1,覆蓋暴跌到 25–31%、切掉 ~70% 真 mesh。因為相觸/重疊區大片是真實表面,一雕就挖穿。ghost 也沒降多少。
3. **覆蓋↔ghost 是蹺蹺板**:堆疊場的 ghost(接觸面外擴)和真 mesh(接觸面本身)長在同處,雕不掉一個不傷另一個。與「相鄰=同物 vs 相觸分不開」同根因。

## 「切掉」來源拆解(cut_breakdown.py,am1,240 個 GT 物體)
- **整個不見的物體**:僅 1/240(0.4%),1/60 場。
- **切掉的 voxel 98.3% 來自「有蓋到的物體之表面薄皮/缺角」**,非整物消失。
- 成因:SAM 剪影邊緣稍內縮 → hull 表面被多雕一薄層。**hull 沒在丟物體。**

## 腳本
- `hull_quality.py` / `hull_quality_am1.py`:品質評估(am0 / am1 基底)。
- `cut_breakdown.py`:「切掉」是整物還是薄皮的拆解。
- `build_am1_photo.py`:定案主線 hull 建置(am1→光度雕→濾<20 碎塊→存 srp_hull_mv2_v12_am1_photo)。
env:`CAPTURES_ROOT=$PWD/data/captures_fast SAM_ROOT=$PWD/data/eval/mobilesamv2_fast`。
