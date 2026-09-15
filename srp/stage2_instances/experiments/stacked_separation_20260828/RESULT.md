# 堆疊分離訊號調查:nnratio + CLIP 能不能分開相觸物

日期 2026-08-28。60 場堆疊(stack3/4/5)。基準全用 **MobileSAMv2**(`mobilesamv2_fast`)+ **am0/am1 + 光度雕** hull。
問題:ConceptGraphs 式的幾何相鄰(nnratio)+ 外觀(CLIP)能不能可靠分開「相觸的不同物體」vs「同物體的不同塊」。

## 訊號定義
- **nnratio**(幾何相鄰):A 的 voxel 有幾成在 B 的 2.5cm 內(雙向取 max,cKDTree)。相鄰=1、離遠=0。
- **CLIP**(語意):donut 遮罩 CLIP 特徵去偏後 cosine 映射 (cos+1)/2。
- **AUC**:隨機一個「同物體對」分數 > 隨機一個「相觸異物對」的機率。1=完美可分,0.5=亂猜。

## 三種單位量測(全 am0/am1+光度雕,MobileSAMv2)

| 單位 | nnratio AUC | CLIP AUC | 說明 |
|------|:---:|:---:|------|
| **語意群 vs 語意群**(先 CLIP 分群)| 0.658 (am0) / 0.663 (am1+photo) | 0.575 | `nnratio_clip.py`、`nnratio_photo.log` |
| **遮罩 vs 遮罩**(全部對) | 0.622 | 0.678 | `mask_nn.py`、`mask_nn.log` |
| **遮罩 vs 遮罩**(只留有相鄰 voxel 的候選對) | 0.655 | — | 濾掉對向不相交對後,仍分不開 |

光度雕做不做幾乎無差(nnratio AUC 0.664→0.663)。

## 遮罩層級候選對分佈(圖:mask_nn_dist/counts/prop.png;資料:mask_nn_cand.npz)
- 兩類都**雙峰**塞在 0 和 1;`[0.9,1.0)` 帶:同物體 48.5% / 相觸 25.7% 都在此爆量。
- **每個分數帶,同物體都是多數(70–91%),相觸永遠少數(最高 30.4%)** → 沒有分數範圍能圈出「該分開」的相觸。
- 最佳門檻(0.47 全對 / 0.91 候選)平衡錯誤率仍 **~38–40%**(每 5 對錯 2)。
- Precision/Recall:**相觸類 precision 永遠 ~20%**(說「該分」時 80% 其實是同物體被誤切),不管門檻。

## 忠實 ConceptGraphs(mask 層級逐一 nnratio+CLIP 關聯,φ>1.1)
`faithful_cg.py`、`faithful_cg.log`:
- 平均關聯出 3.0 物體 / GT 4.0(**過切比 0.74 = 併過頭**)。
- **42% 場(25/60)出現錯併**(相觸物被融成一個);純度中位 1.00(分散物抓得對,相觸物才崩)。

## 定案結論
1. **不管單位取遮罩/群/累積物體,幾何相鄰(nnratio)對堆疊都卡在 AUC ~0.62–0.66,分不開。** 根因:相觸物的**大接觸面 = 高度相鄰 = 高 nnratio**,和同物體無法區分。
2. **CLIP 也救不了**(AUC ~0.58–0.68,兩類外觀分佈幾乎重疊)。
3. **為什麼「8.6% 誤合」也夠爛**:誤合火種集中在接觸面 + 合併有傳遞性,每對相觸物有幾十條高分錯連結,一條就融死一雙 → 42% 場錯併。
4. **要突破需第三訊號**(接縫/不連續偵測、支撐幾何)或**監督式 GNN 學何時該擋住合併**。

## 與舊實驗的關係
取代 `warpcarve_overlap_20260827`(那個 hull 建自 sam_only、與主線 MobileSAMv2 不一致);本調查全在 MobileSAMv2 基準上重驗,結論一致。相關 memory:[[voxeloverlap-semantic-ceiling]]、[[warpcarve-no-help-stacked]]。

## 腳本
- `nnratio_clip.py`:語意群層級 nnratio+CLIP AUC(am0/am1+photo 可切)。
- `mask_nn.py`:遮罩層級 nnratio+CLIP 分佈(存 mask_nn_scores.npz)。
- `plot_counts.py` / `plot_mask_nn.py`:直方圖(數量 / 密度 / 組成比例)。
- `faithful_cg.py`:忠實 ConceptGraphs 逐一關聯 → 過切比/純度/錯併。
env:`CAPTURES_ROOT=$PWD/data/captures_fast SAM_ROOT=$PWD/data/eval/mobilesamv2_fast`。
