# 實驗:voxel 大小 vs 語意群間表面 voxel 重疊率(堆疊物 vs 同物體)

日期:2026-08-27
場景:stack3_scene0001(單場,GT=3:海綿/方塊/杯;海綿+方塊堆疊)

## 目的
測試「縮小 voxel 大小」能否讓堆疊物在語意群間的表面 voxel 重疊率下降到低於同物體,
以便用 voxel 重疊「合同物體、不合堆疊物」。voxel 10/7/5/3/1mm。

## 方法
1. 建 hull:`run_scene.py --voxel {V}`,1mm 用分批版 `carve_chunked.py`(CARVE_CHUNKED=1,避免 GPU OOM;結果與原版逐位元相同)。
2. 補 surface:`add_surface_mask.py`。
3. 語意分群:`voxel_sem_cluster_surf_donut.py`(surf+去大遮罩+CLIP,sem_thr=0.4)。
4. 群間重疊:每語意群=該群遮罩多數決歸屬的表面 voxel 聯集;群兩兩算 Jaccard;GT 貼標分同物體/堆疊相觸(水平<5cm)。

## 結果(stack3_scene0001,5語意群)

| voxel | 同物體群對(該合)中位Jac | 堆疊物群對(該分)中位Jac |
|-------|---------------------|--------------------|
| 10mm  | 0.064               | 0.257              |
| 7mm   | 0.056               | 0.303              |
| 5mm   | 0.053               | 0.263              |
| 3mm   | 0.054               | 0.232              |
| 1mm   | 0.032               | 0.188              |

(每尺寸:同物體 n=3 對、堆疊物 n=1 對)

建 hull 耗時:10/7/5/3mm ~1.9s,1mm 4.35s(分批)。voxel 數:606/1775/4884/22544/609619。

## 結論(單場,初步)
- **voxel 變小 → 堆疊物重疊率下降**(10mm 0.257 → 1mm 0.188,降 27%),符合「小 voxel 交界更精細、共享 voxel 更少」的預期。
- **但沒解決根本問題**:堆疊物(該分)重疊率(0.188~0.257)仍**遠高於**同物體(該合)(0.032~0.064)。方向仍錯——該分的比該合的還像。
- 同物體重疊也隨 voxel 縮小一起降(0.064→0.032),兩者一起降,**差距沒拉開**,voxel 縮小到 1mm 也無法翻轉方向。
- **樣本極少(單場、堆疊物 n=1)**,趨勢待 60 場驗證;但初步看縮小 voxel 幫助有限、不足以用 voxel 重疊分開堆疊物。

## 相關產物(都在本資料夾,自足可重現)
- `data/srp_hull_vox{10,7,5,3,1}mm/`:5 種 voxel 大小的 hull(stack3_scene0001 單場;含 surface)。
- `data/srp_semdonut_vox{10,7,5,3,1}mm/`:對應語意分群(instances.json 含 mask_clusters)。
- `voxsize_group_overlap.py`:分析腳本,讀本資料夾 data/,repo 根目錄執行即重現上表。
- `voxsize_compare.png`:5 種 voxel 大小 hull 對照圖。
- `run_voxsize_full.sh` / `run_voxelsize_exp.sh`:當初建 hull+surface+分群的批次腳本
  (原輸出到 data/eval/,已搬進本資料夾 data/;要重建須改路徑)。
- 分批雕程式:`srp/stage1_hull/carve_chunked.py`(讓 1mm 等大 grid 可跑,run_scene 用
  CARVE_CHUNKED=1 切換;為正式管線元件,留原位不搬)。
- 相關背景:[[voxeloverlap-semantic-ceiling]] — voxel重疊+語意天花板 AUC0.687/v0.674。
