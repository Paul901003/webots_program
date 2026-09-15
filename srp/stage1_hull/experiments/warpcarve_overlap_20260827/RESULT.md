# 實驗:自造 warped-NCC space carving 修 hull 過估 + 對堆疊分離有無幫助

日期:2026-08-27
場景:stack3/4/5 全 60 場(單場驗證用 stack3_scene0001)

## 動機
visual hull 過估體積(幽靈+凹面填實,冗餘 ~51%)。假設:堆疊物在交界處的幽靈體積使兩物
voxel 重疊 → 雕掉過估 → 降低堆疊物語意群間重疊 → 幫助分離 → 幫助物體關係。

## 方法:warped-NCC space carving(`srp/stage1_hull/photo_carve_warp.py`,自造、不跑 COLMAP)
以 visual hull 為基礎,借 COLMAP 光度一致性技術把冗餘 voxel 雕掉:
- 法向 = hull occupancy 梯度(免費表面法向,供單應性 warp)
- warp = 由 (voxel位置, 法向) 算 ref→src 平面單應性(HZ:H=Ks(R_sr + t_sr·n^T/d)Kr⁻¹),把 ref patch 對齊 src
- 一致性 = warped-patch NCC(去亮度),只跟視線夾角最小的 k 個鄰視角比
- **法向微搜尋**:每 voxel 試 9 個傾斜法向(±tilt),取 NCC 最高
- 正則化 = 只雕連通成團(≥min_cluster)的不一致表面 voxel;事後丟 <最大塊1% 的雜訊碎屑
- 純 RGB、免深度、不跑 COLMAP。定案參數:12視角 5mm，tilt20 ncc0.2 k4 msrc2 min_cluster3

前身(同資料夾外的 srp/stage1_hull/):
- `photo_carve.py`(route A:用現成 COLMAP 點雲 flood-carve;因點雲含手臂汙染未採)
- `photo_carve_b.py`（route B：naive 顏色/軸對齊NCC；寬基線下全是雜訊，失敗）

## 雕刻本身有效(單場 stack3_scene0001,vs GT mesh)
| | 覆蓋 | 冗餘(過估) |
|---|---|---|
| 原始 hull | 0.841 | 0.509 |
| 雕後(ncc0.1) | 0.749 | 0.402 |
→ 過估 51%→40%（降 ~11pp），2 主體乾淨不碎裂。**參數/法向微搜尋掃過,frontier 地板 ~冗餘0.39@覆蓋0.73,無法再突破**(剩餘為無紋理幽靈,光度法物理極限)。

## 但對堆疊分離「沒有幫助」(★核心結論,60 場)
語意群間 voxel 重疊率(該合=同物體群對、該分=堆疊物群對):

| hull | 同物體(該合)中位/均 | 堆疊物(該分)中位/均 |
|------|------------------|------------------|
| 原始 | 0.024 / 0.076 | 0.010 / 0.045 |
| 雕後 | 0.024 / 0.076 | 0.010 / 0.043 |

- **雕刻對 60 場重疊率幾乎零效果**(堆疊 0.010→0.010)。
- **單場 stack3_scene0001 的「降 32%(0.272→0.185)」是高離群值,不推廣**。
- 單場看到的「堆疊>同物體(方向反)」也是假象:60 場整體同物體(0.024)>堆疊(0.010)。

## 結論
1. warped-NCC 雕刻成立(降過估、不碎裂、免深度、不跑 COLMAP),但天花板在冗餘~0.39。
2. **雕掉過估對「用 voxel 重疊分堆疊物」無實質幫助**（60 場堆疊重疊 0.010→0.010）。過估不是堆疊分不開的根因。
3. 再次驗證 [[voxeloverlap-semantic-ceiling]]:堆疊分離需第三個訊號,非把 hull 修更準能解。
4. 教訓重申:**單場不可信**(stack3_scene0001 是離群,誤導了中途判斷)。

## 產物
- 方法:`srp/stage1_hull/photo_carve_warp.py`（正式元件,留原位）
- 本資料夾:`warp60_build.py`(建hull+雕+打包 60場)、`warp60_measure.py`(量重疊彙總)、`result_60scene.txt`
- 資料 root(data/eval,不進 git,可由腳本重生):`srp_hull_warp_{orig,carved}`、`semdonut_warp_{orig,carved}`
- 時間:建hull+雕 60 場 616s(~10s/場);分群 2.4s/場;60場全實驗 ~15 分。
