# 整體演算法流程與實驗規劃

> 免深度 (RGB-only)、eye-in-hand 機械手臂、語意 visual hull → GNN 物件關係 → LLM 任務規劃。
> 本文件彙整目前為止確定的流程與實驗;與 `visual_hull_carving_spec.md` 配套 (該檔是 Stage 1 的實作規格)。

---

## 0. 定位 (一頁回顧)

**做什麼**:手臂多視角 RGB → 免深度 visual hull → 帶 instance 的佔據 → GNN 推物件間關係 → scene graph → LLM 規劃。

**三個賣點 (論文要證的 claim)**:
- **C-MOD 模組化可修正**:辨識/關係/規劃三段各有人類可讀介面,錯誤可定位、各段可獨立改良。
- **C-GEO 幾何 grounding**:關係 grounding 在 arm-pose hull 的可檢核度量幾何,而非 VLM 隱式推理。
- **C-DEP 免深度**:對 RGB-D 主打鏡面/反光金屬件 (深度直接失效),對 SfM/NeRF 主打無紋理表面 (特徵匹配崩潰)——兩條失效線皆能運作。

**與 SOTA 的區隔**:RoboRetriever / ConceptGraphs / VL-GRiP3 / UniManip 皆為 RGB-D + VLM/LLM 推關係;GVMRN 為單視角 2D GNN。**你 = 免深度 hull + 多視角 3D 幾何上的 GNN 關係**,此交集無人佔。

**範圍哲學**:架構完整、實作分層。骨架每段都打通能端到端跑,深度集中在 hull + GNN 兩段;其餘用最簡版,部分留 future work。

**已定範圍**:hull 固定軌跡不最佳化;關係 2 種原始 (`on` 支撐、`blocks_access` 遮擋),抓取順序為衍生;模擬環境 (外參 ground-truth,免校正)。containment 明確排除 (hull 無凹腔)。

**三場景評估架構 (難度遞增階梯)**:
- **場景 1 — 分開擺放**:無嚴重遮擋。baseline,證整鏈跑通。純 hull,開環即可。
- **場景 2 — 部分遮擋**:物體被擋但某些視角仍可見。**招牌貢獻**——多視角 hull 對部分遮擋天生強,打單視角 (GVMRN 2D / 單張 RGB-D) 的點。需符號層計畫檢查。
- **場景 3a — 可揭露的完全遮擋**:目標被完全擋住、所有視角不可見,但**移開遮擋物後可揭露**。需 **感知層重掃 + 揭露行為** (故 Stage 6 重掃進 core)。證明閉環價值。
- (放棄 **3b 容器內不可見**:visual hull 填實凹腔,結構性做不到。)

---

## 1. 整體流程 (Stage 0–6)

資料在各段之間的介面要釘死,這是「分段可修正」的物理載體。`[core]` = 這次做,`[v2]` = future work。

### Stage 0 — 擷取 (模擬環境)
- `[core]` 固定上半球軌跡,N 個視角 (N 由實驗 B1 收斂曲線決定);stop-and-shoot。
- `[core]` 模擬器提供 ground-truth 外參 (world→camera) + 內參;架構預留外參為可替換輸入 (未來上真機接校正)。
- `[core]` 每視角:RGB → SAM **實例遮罩** (保留 per-instance ID,不只合併前景)。
- **輸出介面**:`{RGB, K, [R|t], instance_masks}` × N 視角。

### Stage 1 — Visual hull carving  → 見 `visual_hull_carving_spec.md`
- `[core]` voxel 投影測試:每 voxel 投影到「看得到它的」視角取交集 (含可見性遮罩,frustum 外不投票)。
- `[core]` 支撐平面封底 (table_z)。
- `[v2 / 亮點]` soft/機率化雕刻:用「投前景票數比例」門檻,容忍少數視角漏檢 (抗 SAM false-negative)。
- `[core]` 輸出**可見性** `observed`:標記每 voxel 是否曾被任一視角看進去 → 區分「未觀測空間」與「確定為空」(場景 3a 揭露所需)。
- **輸出介面**:世界座標系的佔據網格 `occupancy + observed + grid_min + voxel_size`。

### Stage 2 — Instance 指派 (跨視角關聯)
- `[core]` 跨視角實例關聯:用精確外參 + voxel 當「橋」,把各視角的 local SAM ID 焊成 global instance (幾何一致性,非外觀)。
- `[core]` voxel 多數投票 → 每 voxel 一個 global instance 標籤。
- `[core]` 處理 SAM 過度分割 (合併共佔同一 voxel 群的碎片 ID)。
- **輸出介面**:帶 instance 標籤的佔據網格。**3D 不切割,只標記。**

### Stage 3 — 建圖
- `[core]` 節點 = 物件 (group by instance label):特徵 = OBB 參數、體積、質心、CLIP embedding。
- `[core]` 邊 = 物件對:邊特徵由「未切開時的物體對幾何」算 (相對位移、距離、垂直高差、尺寸比、接觸面、投影重疊、遮擋線索),**重力對齊座標系下的相對量** (泛化關鍵)。有向、multi-label。
- **輸出介面**:`graph = {nodes[特徵], directed_edges[幾何邊特徵]}`。

### Stage 4 — GNN 關係推理
- `[core]` 架構錨定 GVMRN / GGNN-VMRN (操作 domain),非 3DSSG。
- `[core]` edge-conditioned message passing (PyG NNConv/TransformerConv) 2–3 層;輸出 per-edge multi-label 關係 logits **+ confidence** (BCE/focal)。
- `[core]` hybrid:可解析算的關係 (上下左右遠近) 當邊先驗,GNN 只修正接觸/遮擋/順序等規則算不出的。
- **輸出介面**:scene graph (關係 + 信心值)。

### Stage 5 — Scene graph → LLM 規劃
- `[core]` 把圖序列化成**可讀文字 schema** 餵 LLM;LLM 輸出任務規劃 (任務零樣本)。
- `[core]` **翻譯層 (場景 3a)**:把 hull 的未觀測空間幾何降維成 `has_hidden_region` / `possibly_occludes` 謂詞 (+方位詞),再隨圖序列化給 LLM。幾何決定「誰是嫌疑遮擋物」,LLM 語意排序「先查誰」。見 `plan_check_schema.md` §7。
- 介面刻意 inspectable,這是 C-MOD 成立的關鍵。

### Stage 6 — 閉環 (規劃層計畫檢查 + 場景 3a 揭露重掃)
- `[core]` belief scene graph (帶信心值的世界模型),當作**計畫檢查用的模擬器**。
- `[core]` **規劃層回饋 (場景 1/2)**:LLM 出動作序列 → 用 belief graph 逐步檢查**前置條件可行性** (例:要抓 A,但 `on(B,A)` → 違反) → 違反原因回饋 LLM → replan。符號層跑,不動相機手臂。
- `[core for 3a]` **感知層揭露重掃**:任務目標**不在圖中** → 判定完全遮擋 → 由 `possibly_occludes` 篩嫌疑物、LLM 排序 → `reveal(B)` 移開 → **局部重掃** B 後方未觀測空間 → 更新 belief graph → 目標現身則重規劃,否則換下一嫌疑物。遮擋層數上限 1–2 層。
- `[v2]` 一般失敗的感知層回饋 (非揭露場景的關係糾錯重掃);**主動補視角 NBV** (低信心關係驅動手臂選視角)。
- **誠實盲區**:belief graph 是自己從 hull+GNN 估的、會錯。當錯誤源在**感知層** (GNN 把支撐判反),純規劃層檢查會拿錯圖驗證 → 擋對的、放錯的。場景 1/2 第一版先假設圖正確;論文須點明**規劃層閉環修不了感知層錯誤**。(場景 3a 的揭露重掃是感知層回饋的一個特例,正好示範感知層閉環的價值。)

**完整資料流**:
```
擷取 → hull(occupancy+observed) → 帶instance佔據 → graph → GNN(關係+信心)
  → scene graph (+未觀測空間翻譯成 has_hidden_region/possibly_occludes) → LLM規劃
  → belief graph前置條件檢查 → 不可行則回饋replan → 執行
  場景3a: 目標不在圖 → reveal(嫌疑物) → 局部重掃未觀測空間 → 更新圖 → 重規劃
```

---

## 2. 實驗總表 (各個實驗)

每個實驗標註:驗證哪個 claim、core/v2、對照組、指標、能否在純模擬完成。

| ID | 實驗 | 驗證 | core? | 對照組 | 主要指標 | 純模擬可? |
|---|---|---|---|---|---|---|
| **A1** | 表徵充分性 probing (oracle):特徵階梯 centroid→OBB→voxel→點雲→視角覆蓋,小 probe 預測各關係 | C-GEO | core | 各特徵層級 | 各關係 probe 準確率飽和曲線 | ✓ |
| **A2** | 充分性退化軸:hull 品質下降 (少視角/壞遮罩) 下關係可復原性 | C-GEO | core | 退化程度 | 關係復原率 vs 退化 | ✓ |
| **B1** | 視角數收斂:hull 體積 vs 視角數 → 定 N | — | core | 視角數 | 體積收斂飽和點 | ✓ |
| **B2** | voxel 解析度收斂:hull 誤差 vs voxel_size | — | core | 解析度 | 誤差 vs 記憶體/時間 | ✓ |
| **B3** | 遮罩品質:GT 遮罩 vs SAM 遮罩 對 hull 誤差 | C-DEP | core | 兩種遮罩 | hull IoU / 體積誤差 | ✓ |
| **B4** | 漏檢敏感度:抽掉一視角遮罩 → hull 體積損失 (證 soft hull 必要) | C-GEO | core | hard vs soft carving | 體積損失量 | ✓ |
| **C1** | carving 正確性:規格 T1–T7 (標準球/方向慣例/單調性/多物體/封底/對齊IoU/可重現) | C-GEO | core | — | 各測試通過 | ✓ |
| **C2** | soft hull:T8 單視角漏檢不雕穿物體 | C-GEO | v2 | hard hull | 物體保留率 | ✓ |
| **D1** | 跨視角實例關聯正確率 vs 模擬器 GT instance | C-GEO | core | — | 關聯準確率 | ✓ |
| **D2** | 接觸/堆疊物體:T9 兩接觸物分到不同 instance | C-GEO | core | 3D 連通元件 baseline | 實例分離正確率 | ✓ |
| **D3** | GT-instance vs SAM+關聯:關聯誤差傳到下游 | C-MOD | core | 兩種來源 | 下游關係準確率落差 | ✓ |
| **E1** | GNN 關係預測:各關係準確率/recall vs GT MRG | C-GEO | core | — | per-relation F1 (multi-label) | ✓ |
| **E2** | **關鍵對照**:GNN(3D) vs VLM 推關係 vs GVMRN(2D),聚焦遮擋順序案例 | C-GEO | core | VLM / 2D-GNN | 遮擋順序正確率 | 部分 |
| **E3** | GT-hull vs real-hull 輸入:hull 誤差傳到關係 | C-MOD | core | 兩種輸入 | 關係準確率落差 | ✓ |
| **F1** | **錯誤定位**:給失敗案例,能否歸因到 辨識/關係/規劃 哪段 | C-MOD | core | — | 歸因正確率 | ✓ |
| **F2** | **獨立修正**:抽換分割/修一條關係/換 LLM,他段不重訓 | C-MOD | core | — | 修正後改善 & 他段不變 | ✓ |
| **F3** | **模組化對照**:端到端 VLM baseline 同樣失敗無法局部修正 | C-MOD | core | 端到端 VLM | 可局部修正性對比 | 部分 |
| **F4** | belief graph 維護正確率 (SGH 式) | C-MOD | core | — | 世界模型更新正確率 | ✓ |
| **G1** | 端到端任務成功率,**分三場景各自報** (任務零樣本) | 全部 | core | 端到端 VLA/VLM | 各場景 task success rate | 部分 |
| **G2** | 計畫檢查 vs 無檢查:前置條件檢查 + replan 是否提升 (場景 2 為主) | C-MOD | core | 無檢查直接執行 | 成功率 / 不可行計畫攔截率 | 部分 |
| **G3** | **揭露行為 (場景 3a)**:目標完全遮擋下,reveal+重掃能否找到目標 | C-GEO, C-MOD | core | 無揭露 / 隨機移開 / 純LLM猜 | 揭露成功率 / 移開次數 | ✓ |
| **G4** | 未觀測空間嫌疑篩選:幾何篩選 vs 全物體窮舉 的效率 | C-GEO | core | 窮舉 baseline | 找到目標前的 reveal 次數 | ✓ |
| **H1a** | **免深度 vs RGB-D**:鏡面/反光金屬件 (主打) 上 hull vs RGB-D 重建 | C-DEP | core* | RGB-D (RoboRetriever 類) | 鏡面件重建/成功率 | **✗ 需真機/真實資料** |
| **H1b** | **免深度 vs SfM**:無紋理表面上 hull vs COLMAP/NeRF (第二賣點) | C-DEP | v2* | COLMAP / NeRF | 無紋理件重建成功率 | **✗ 需真實資料** |

\* H1 是 C-DEP 賣點的關鍵實證,**純模擬證不出** (模擬的深度與紋理都完美)。免深度有**兩條失效線**:對 RGB-D 主打**鏡面/反光金屬** (H1a,深度感測器直接失效);對 SfM/MVS/NeRF 主打**無紋理表面** (H1b,特徵匹配崩潰)。薄件/細結構可「也適用」帶過,不單獨展開。建議 **H1a 為 core (主賣點)、H1b 為 v2**,避免 H1 自己膨脹成一篇論文。硬體成本**不可**當主賣點 (一顆 RGB-D 很便宜,論點撐不起來)。兩者皆需真實資料/真機,獨立於模擬主線。

---

## 3. 誠實的開放風險

- **誤差累積**:四段串聯,各段 80% 乘起來約 40%。F 系列實驗正是要證明「至少壞了可定位可修」來緩解,而非宣稱不累積。
- **關係封閉 ≠ 全零樣本**:GNN 在固定關係集上訓練,物體/任務零樣本但關係不是。論文措辭守在「物體與任務零樣本、關係 grounding 在幾何」,別宣稱全零樣本。
- **sim-to-real gap**:免校正是延後不是消除。架構預留校正介面;真實 SAM 在鏡面件、真實光照的表現只有 H1 能驗。
- **實例關聯會錯**:D1 若低,兩物體 voxel 混淆直接污染關係,故 D1 必須當獨立指標量。
- **GNN 監督來源**:用 REGRAD 式物理模擬自動生 MRG 標籤;sim-to-real 為 v2。
- **scope**:全做是博士量。守住「骨架完整、hull+GNN 做深、其餘最簡」,否則最可能結局是某段卡死、端到端跑不起來。
- **場景 3a 加重 scope**:選 3a 把「感知層重掃 + 揭露行為 + 處理不完整世界模型」拉進 core,且 3a 是**最長、最易在 demo 翻車**的鏈 (同時依賴重掃與 LLM 揭露規劃都對)。**開發順序務必把 3a 放最後**;1+2 本身已是完整可發表的東西,3a 萬一時間不夠可降級為「初步結果 / 概念驗證」。
- **揭露的層數邊界**:reveal 只處理 1–2 層遮擋;更深堆埋或容器 (3b) 不在範圍,論文須明講。

---

## 4. 建議執行順序

> 按場景難度遞進:**先把場景 1+2 紮實做完 (已可發表),再衝場景 3a**。

1. **Stage 1 + C1**:先讓 carving 正確 (尤其 T6 對齊、T2 方向慣例),幾何主幹站穩;順帶輸出 `observed`。
2. **B1/B2**:定 N 與解析度。
3. **Stage 2 + D1/D2**:實例關聯打通,接縫補上。
4. **A1/A2 充分性 probing**:用 GT hull 決定 GNN 節點特徵規格 (再回頭定 Stage 3 節點維度)。
5. **Stage 3–4 + E 系列**:GNN 上線;E2 (vs VLM/2D) 是 C-GEO 的證明,最重要。
6. **Stage 5–6 (場景 1+2) + F/G1-G2**:接 LLM 與規劃層計畫檢查;F 系列是 C-MOD 的證明,不能只量 G1 成功率。**到這裡已是完整可發表成果。**
7. **場景 3a:翻譯層 + reveal + 揭露重掃 + G3/G4**:全鏈最難,放最後;做不完可降級概念驗證。
8. **H1a**:擇期補真機/真實資料的**鏡面件 vs RGB-D** 實驗,撐 C-DEP 主賣點 (H1b 無紋理 vs SfM 為 v2)。

> 鐵則:先用 **GT 遮罩 / GT instance** 把後段調通 (驗「完美輸入下整鏈成立」),再換成 **SAM + 你的關聯** 量退化。先 GT、再真實,貫穿 hull 段與 instance 段。
