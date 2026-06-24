# Stage 6 計畫檢查 — 謂詞 Schema (SayPlan 式)

> 用途:定義 belief scene graph 的關係謂詞,以及抓取/放置動作的**前置條件與效果**,
> 使規劃層計畫檢查器 (= SayPlan 的 scene graph simulator) 能用**純符號比對**驗證 LLM 計畫可行性。
> 對應 `pipeline_and_experiments.md` 的 Stage 6 `[core]`。配套檔:hull 規格、pipeline 總表。

---

## 0. 你的 3 種關係 → schema 謂詞的對應 (先對齊,確保與 GNN 一致)

| 你定的關係 | schema 謂詞 | 備註 |
|---|---|---|
| 支撐 (support) | `on(X, Y)` | X 置於 Y 之上 / X 由 Y 支撐 |
| 遮擋 (occlusion) | `blocks_access(X, Y)` | X 擋住抓取 Y 的接近路徑。**視覺遮擋是它的證據,但不等同**——見 §1 |
| 抓取順序 (grasp order) | **衍生,非原始謂詞** | 由 `on` + `blocks_access` 推出,見 §3。(若改用 REGRAD 式直接預測 parent 邊,則 grasp order 為原始,見 §5 註) |

GNN 只需預測 `on` 與 `blocks_access` 兩種邊;抓取順序由檢查器從前置條件自動推出,不必另外學。

---

## 1. 慣例 (先釘死,否則檢查器與 GNN 不一致 —— 同 hull 規格釘座標慣例的精神)

- **方向慣例 (最易錯,最該先定)**:`on(X, Y)` ≜ X 在上、Y 在下 (X 被 Y 支撐)。
  GNN 輸出的 support 邊必須照此方向寫入 belief graph;若 GNN 內部方向相反,寫入前先反轉。
- **遮擋 ≠ 接近阻擋**:相機視角的視覺遮擋,和「夾爪接近路徑被擋」不是同一件事。
  `blocks_access(X, Y)` 指的是**後者** (操作相關)。視覺遮擋只是它的證據之一。
  第一版可先用視覺遮擋近似 `blocks_access`,但論文須註明此近似的落差。

---

## 2. 型別與謂詞 (belief graph 的內容)

**型別**:`object` (來自 Stage 2 instance 標籤)、`gripper`、`location`。

**狀態謂詞 (fluents,會被動作改變)**:
- `on(X, Y)` — X 置於 Y 上                  [GNN 預測,帶 confidence]
- `blocks_access(X, Y)` — X 擋住抓取 Y 的路徑   [GNN 預測,帶 confidence]
- `clear(Y)` — 沒有物體在 Y 上              [衍生:`clear(Y) ⟺ ¬∃X. on(X,Y)`]
- `accessible(Y)` — 沒有物體擋住 Y           [衍生:`accessible(Y) ⟺ ¬∃X. blocks_access(X,Y)`]
- `in_gripper(X)` — X 被夾在手上
- `gripper_empty` — 手上沒東西
- `at(X, L)` — X 在位置 L

**揭露謂詞 (場景 3a 專用,來源是可見性幾何,非 GNN)**:
- `has_hidden_region(B)` — B 後方/下方有足夠藏住物體的**未觀測空間** (`occupancy=False & observed=False`)。由 hull 的 `observed` 輸出算出 (見 §7 翻譯層)。
- `possibly_occludes(B, target)` — 當任務目標 `target` **不在 belief graph 中** 且 `has_hidden_region(B)` 成立時為真。表示 B 是「可能藏著目標」的嫌疑遮擋物。
- 兩者各帶**粗略方位詞** (B 的未觀測空間在「後方/下方/內側…」),供 LLM 排序。

**每條 GNN 預測的邊附帶**:`relation_type`、`confidence ∈ [0,1]`、`last_updated`。
(`confidence` 是你跟 SayPlan 的關鍵差異,見 §5。)

---

## 3. 動作:前置條件 + 效果 (STRIPS 式) —— schema 的核心

檢查器逐步拿這些前置條件比對 belief graph。

```
action grasp(X):
  precond:
    gripper_empty
    clear(X)            # ¬∃Z. on(Z, X)          沒東西壓在 X 上
    accessible(X)       # ¬∃Z. blocks_access(Z,X)  沒東西擋住接近路徑
  effect:
    in_gripper(X);  ¬gripper_empty
    ∀Y: on(X,Y)            → ¬on(X,Y)             # X 離開 → Y 可能變 clear
    ∀Y: blocks_access(X,Y) → ¬blocks_access(X,Y)  # X 離開 → Y 可能變 accessible

action place(X, L):
  precond:
    in_gripper(X)
    valid_surface(L)
  effect:
    ¬in_gripper(X);  gripper_empty;  at(X, L)
    若 L 在物體 Z 之上 → 新增 on(X, Z)
```

**衍生動作** `remove(W)` ≜ `grasp(W)` + `place(W, 空位)`,用來解阻擋。

**抓取順序如何「浮現」**:要 `grasp(A)` 但 `on(B,A)` 成立 → `clear(A)` 為假 → 前置條件違反
→ 檢查器要求先 `remove(B)`。於是「先 B 後 A」的順序**從前置條件自動推出**,不需 GNN 直接給順序。

**揭露動作 (場景 3a)** —— 處理「目標完全遮擋、不在圖中」:
```
action reveal(B):                    # 移開嫌疑遮擋物 B 以揭露其後的未觀測空間
  precond:
    possibly_occludes(B, target)     # B 後方有未觀測空間且目標不在圖中
    clear(B);  accessible(B)          # B 本身要抓得到 (否則先解 B 的阻擋)
  effect:
    remove(B)                        # = grasp(B)+place(B,空位)
    觸發局部重掃 B 原本遮蔽的未觀測空間 → 更新 belief graph
    # 重掃後: target 可能首次出現在圖中 (成功),或仍不在 (換下一個嫌疑物)
```
**觸發器 (核心)**:任務指涉的 `target` **不在 belief graph 中** → 判定「目標可能被完全遮擋」
→ 蒐集所有 `possibly_occludes(_, target)` 的嫌疑物 → 由 LLM 用方位詞 + 語意排序 → 依序 `reveal`。
這是「**任務要的東西圖裡沒有**」這個 gap 驅動的,不需要無中生有偵測不可見物。

**邊界 (寫進論文)**:`reveal` 只處理「目標在單一可移開遮擋物後方」;**遮擋層數上限建議 1–2 層**。
更深的堆埋、或需「打開」的容器 (退回 3b),本方法不處理。

---

## 4. 檢查程序 (檢查器怎麼跑) —— 對應 SayPlan 的 scene graph simulator

```
輸入: LLM 計畫 = [a1, a2, ..., an]
state ← 當前 belief graph 投影成謂詞集合 (含衍生謂詞 clear / accessible)
for each action ai in 計畫:
    if ai.precond 在 state 不成立:
        return (INFEASIBLE, ai, 違反的前置條件, 違反它的物體)
    else:
        state ← apply ai.effect to state     # 模擬執行,推進狀態 (不碰真機)
return FEASIBLE
```

違反時,把原因回饋 LLM → replan,**步數上限 N (如 5)** 避免無限迴圈。
回饋訊息要具體,例:`"grasp(A) 違反 clear(A),因為 on(B,A);請先 remove(B)"`。

**範例**:場景 `on(B,A)`,任務「拿起 A」。
LLM 出 `[grasp(A)]` → 檢查 `clear(A)` 失敗 (因 `on(B,A)`) → 回饋 → LLM replan 成 `[remove(B), grasp(A)]`
→ 重檢:`remove(B)` 使 `on(B,A)` 消失 → `clear(A)` 成立 → `grasp(A)` 通過 → FEASIBLE。

---

## 5. 你跟 SayPlan 的關鍵差異:confidence (你的圖會錯,它的不會)

SayPlan 的圖是給定且乾淨的;你的關係是 GNN 估的、帶 confidence。
所以前置條件檢查是**三態**,不是二態:

- **SATISFIED**:決定可行性的關係高信心且滿足 → 通過。
- **VIOLATED**:高信心且違反 → 回饋 replan (同 SayPlan)。
- **UNCERTAIN**:關鍵關係 `confidence < 門檻`
  - `[core]` **保守處理**:當成可能違反,要求 LLM 把計畫排成「先驗證/先移開」(寧可多一步,不冒險)。
  - `[v2]` 觸發感知層**重掃**該區,重判該關係後再檢查。

> 註:若改採 REGRAD 式,讓 GNN **直接預測 manipulation parent 邊** (抓取順序為原始關係),
> 則檢查器改查 parent 邊;但前置條件式 (本 schema) 更貼近 SayPlan、更可讀可改,建議優先。

---

## 6. 誠實盲區 (寫進論文)

檢查器只驗證**「計畫是否符合圖」**,不驗證**「圖是否符合現實」**。
若 GNN 把 `on(B,A)` 判成 `on(A,B)` (方向判反),檢查器會用錯誤的圖放行錯計畫或擋下對計畫。
**純規劃層閉環偵測不到這類感知錯誤** → 這正是 `[v2]` 感知層重掃存在的理由,而非可有可無。
(對應 pipeline Stage 6 的盲區註記。)

---

## 7. 翻譯層:未觀測空間 → 謂詞 → 文字 (場景 3a 的幾何↔LLM 接縫)

**原則:不要把幾何 (voxel/座標) 給 LLM,要給它「對物體的語意後果」。** 三層,只有最上層碰 LLM。

**第一層 (幾何,LLM 看不到)**:用 hull 的 `observed` 輸出找未觀測空間 (`occupancy=False & observed=False`);
對每個物體 B,判斷其鄰接 (後方/下方) 是否有夠大的未觀測空間 (體積 > 目標尺寸門檻)。

**第二層 (翻譯成謂詞)**:把幾何結論寫成 belief graph 謂詞:
- 有足夠未觀測空間 → `has_hidden_region(B)` + 粗略方位詞。
- 目標不在圖中 → 對每個 `has_hidden_region` 的 B 標 `possibly_occludes(B, target)`。
- **方位詞**:由未觀測空間質心相對 B 質心的方向,量化成 6 方位之一 (後/下/左/右/前/上)。

**第三層 (序列化給 LLM,文字)**:belief graph 轉文字時這些謂詞變成句子。例:
```
場景物體: box_1, cup_2, large_panel_3
關係: on(cup_2, box_1)
      has_hidden_region(large_panel_3) @ 後方
任務: 拿起 scissors
備註: scissors 不在已觀測場景; possibly_occludes(large_panel_3, scissors)
```
LLM 據此推理:「目標不在場景、panel_3 後方有未觀測空間 → 先 reveal(panel_3) 再找」。

**職責分工 (對應揭露策略二/三)**:
- **幾何 (策略二)** 產生 `has_hidden_region` / `possibly_occludes` → 決定**「誰有資格當嫌疑犯」**。
- **LLM (策略三)** 拿方位詞 + 語意常識 → 排序**「先 reveal 哪個嫌疑犯」**(找剪刀→工具箱比畫框可疑)。
- 幾何負責「誰有可能」,LLM 負責「先賭誰」;LLM 的猜測被**限制在幾何篩出的候選內**,不亂猜全場。

這條接縫和你整篇哲學一致:幾何先 grounding 成符號謂詞,LLM 只在符號層推理,中間表徵可讀可查。
