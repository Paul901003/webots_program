# 合併方案評估(定案,2026-09)— modal 表面 3D-IoU + 堆疊分離兩軸

## 管線與問題
現行管線(單一條):**am1+光度雕 hull(`srp_hull_mv2_v12_am1_photo`)→ MobileSAMv2 遮罩 → CLIP 語意分群(`srp_hull_semcluster_surf_am1photo`)**,嚴重過切(1052 GT 物體 → 2347 個 instance)。
問題(plan D2):把過切碎片併回,同時把相觸堆疊物分到不同 instance。在測的方法**只差「怎麼合併過切的 CLIP 群」**:baseline(不併)/ crossing / veto / plane-step / span+3D閘門(span3d)。

## 鎖定的主指標:modal 表面 voxel 3D-IoU 找到率(看結果前定義,不再改)
- 共用 hull 表面 voxel;**GT**=每表面 voxel 由其**可見(zbuffer 最前)**視角投影落進哪個物體 **modal**(只可見部分)遮罩、多數決 → GTsurf(o)。
- **預測**=該 root 的 `instances.npz` labels 在表面 voxel 上(用 voxel labels,避開 instances.json 遮罩檔問題)。
- **配對**=預測 instance ↔ GT 物體 以表面 voxel 3D-IoU 做 Hungarian;IoU≥門檻算找到。
- 腳本 `eval_surface_iou.py`;per-object IoU 存 `eval_surface_iou_<root>.csv`。分母 1052 GT 物體(排 GLOBAL_EXCLUDE),4 方法同一批 GT。

### 主結果(303 場,GT 物體 1052)
| 方法 | 預測inst | 幻影 | found@0.3 | found@0.5 | found@0.7 | 全體mIoU |
|---|--:|--:|--:|--:|--:|--:|
| baseline(不併) | 2347 | 1295 | 0.985 | 0.818 | 0.644 | 0.750 |
| **span3d** | 1275 | 233 | 0.980 | **0.963** | **0.899** | **0.853** |
| veto | 1371 | 326 | 0.981 | 0.932 | 0.848 | 0.834 |
| crossing | 1218 | **191** | 0.957 | 0.928 | 0.852 | 0.822 |

### 依場景組(同指標切分;分母 n646/occ206/stack3-48/stack4-68/stack5-84)
found@0.7:
| 方法 | n | occ | stack3 | stack4 | stack5 |
|---|--:|--:|--:|--:|--:|
| baseline | 0.67 | 0.67 | 0.54 | 0.51 | 0.57 |
| span3d | 0.93 | 0.92 | 0.69 | 0.74 | 0.85 |
| veto | 0.88 | 0.88 | 0.71 | 0.71 | 0.68 |
| crossing | 0.90 | 0.84 | 0.62 | 0.65 | 0.77 |
→ 堆疊組每個方法都最難;此「找到率」指標下 span3d 對堆疊仍優於 baseline。

## 核心:兩個目標是取捨,沒有方法兩者兼得
把「純堆疊分離」與「過切合併」分開量(29 個可評 on 對,排 GLOBAL_EXCLUDE 31 個):

| 方法 | 純堆疊分離(上下物主群不同) | 過切合併(found@0.7) | 幻影 |
|---|--:|--:|--:|
| **baseline(不併)** | **0.97**(28/29) ← 分離最好 | 0.64 ← 合併最差 | 1295 |
| veto | 0.45(13/29) | 0.85 | 326 |
| span3d | 0.45(13/29) | **0.90** ← 合併最好 | 233 |
| crossing | 0.38(11/29) | 0.85 | 191 |

**單調取捨**:越敢併 → 過切合併越好、堆疊分離越差。baseline 不併=堆疊分離 0.97 但過切最差;span3d 最敢併=過切最好但堆疊分離掉到 0.45。**沒有方法同時在兩軸都高。**

（另:「乾淨分開」= 分開且上下物各自 recall≥0.5、precision≥0.5,d2_stack_clean.py:baseline 0.48 / veto 0.34 / crossing 0.28 / span3d 0.24。baseline 從 0.97 掉到 0.48 全因過切—28 個分開中 14 個主群 recall<0.5;precision≈1 幾乎不混。此指標把過切與分離綁一起,故 baseline 也被過切拖累。）

### per-stack 明細
每個堆疊在各方法的分離狀況見 `stack_separation_perpair.csv`(29 對 × 4 方法 ✓/✗)。baseline 分開 28/29(僅 stack5_scene0019 sugar_box/cracker_box 未分開,CLIP 本就同群);合併法各只分開 11–13 個。

## 為什麼兩者兼得不了(根本原因)
合併訊號(共視重疊 / 同一大遮罩框住 / 3D 連通 / 深度階梯)對「同物過切碎片」與「相觸堆疊兩物」反應相同——兩者都 3D 相連、被同一區域覆蓋。3D 連通閘門只擋得住「分開擺」的物體,擋不了「真相觸」的堆疊。**免深度、同材質下,沒有訊號能區分「該併的碎片」與「不該併的相觸物」——這是天花板。** 突破需新訊號:深度(排除)或能辨同/異物的外觀。

## 結論
- **要過切合併/重建完整 → span3d 最好**(found@0.5=0.963、@0.7=0.899、mIoU 0.853、幻影 233)。
- **要保住堆疊分離 → baseline 不併最好**(0.97),任何合併都掉到 ~0.4。
- **兩者兼得目前無解**;最平衡折衷 = veto。
- plane-step 平面假設從根上錯(前表面取到側面、堆疊是頂面 vs 側面方向不同、曲面外插失效),已棄。

## ★ 完整度合併(法向多樣性)—— 突破取捨(2026-09-09)
**訊號**:每群「表面法向多樣性」`div = 1 − ‖mean(unit normal)‖`,法向取自 `np.gradient(occupancy)`(正確軸序),在 `labels==k`(完整可見表面 voxel,已鎖定;100% observed)上算。已驗證:vs 可靠窗法向 cosine 0.89;碎片中位 0.32、完整物中位 0.66,AUC(完整>碎片)堆疊 0.884 / 分開 0.829。轉念:不看「兩相觸群該不該併」(接觸面填平無解),改看「每群本身是完整物體還是碎片」(單群性質,正交)。
**兩方法**(`merge_div.py`,θ=完整門檻,掃 0.4/0.5/0.6;3D 連通閘門):
- **A 純完整度閘門**:3D 連通相鄰對,兩群都 div≥θ(都完整)→ 不併;否則(至少一碎片)→ 併。
- **B span3d + 完整度否決**:取 span3d 邊,否決「兩群都完整」的邊。

### 兩軸結果(303 場;過切軸 1052 物體、堆疊軸 29 on 對)
| 方法 | 過切 found@0.7 | mIoU | 幻影 | 堆疊乾淨分離(τ.5) | 堆疊純分離 |
|---|--:|--:|--:|--:|--:|
| baseline(不併) | 0.644 | 0.750 | 1295 | 0.48 | 0.97 |
| span3d | 0.899 | 0.853 | 233 | 0.24 | 0.45 |
| divA θ.5 | 0.872 | 0.846 | 347 | **0.66** | 0.69 |
| divA θ.6 | 0.896 | 0.852 | 255 | 0.55 | — |
| divB θ.5 | 0.873 | 0.848 | 365 | **0.66** | 0.86 |
| **divB θ.6** | **0.900** | 0.857 | 283 | 0.62 | 0.83 |

**關鍵結論**:**div 同時贏兩軸,把取捨前緣整條往外推(Pareto 更好)**:
- vs baseline(舊堆疊最佳 0.48):div θ.5 兩軸皆贏(重建 0.87 vs 0.64、堆疊 0.66 vs 0.48)。
- vs span3d(舊重建最佳 0.90):**divB θ.6 重建打平 0.900、堆疊 0.62 是 span3d 0.24 的 2.6 倍**。
- 甜蜜點:重建優先→ divB θ.6;堆疊優先→ div θ.5。
機制:低 div 碎片併回(過切收好)、高 div 完整物(堆疊上下物)不互併(堆疊保住)。

### 失敗分析(divB θ.6,乾淨分開 18/29,失敗 11;0 個混物)
- **未分開 5**(div 仍把上下併掉):stack4_0007/0014(sugar_box/sponge)、stack4_0010(tuna_can/master_chef_can)、stack5_0011(foam_brick/sponge)、stack5_0019(sugar_box/cracker_box)。成因=**扁/相似物 diversity 低→假碎片**;stack5_0019 是 **CLIP 上游就把兩物併一群,任何合併法無解**。
- **過切 6**(分開但物體 recall<0.5):gelatin_box(0.23/0.30)、pudding_box(0.23)、sugar_box(0.30)、foam_brick/tuna_can(0.45)——盒子碎片沒收齊(θ.6 保守)。θ.5 救回 1 個→失敗 10。

### 誠實但書
- **堆疊軸只 29 on 對**(GLOBAL_EXCLUDE 砍掉積木堆),小樣本;0.66 vs 0.48 待在更多/新資料(平衡集 stkb)驗證。過切軸 1052 物體較穩。
- 失敗集中在扁物/軟塊/相似盒子(diversity 對它們弱);補救方向:diversity 再加尺寸/厚度線索。
- 只會合併不會拆分 → CLIP 上游欠切(如 stack5_0019)一律無解。

### 可視化 / provenance
roots：`srp_hull_div{A,B}_t{40,50,60}`(`merge_div.py`,build_meta 記 method/theta)。看:`SRP_VIZ_ARGS="<scene> 1 srp_hull_divB_t60" webots worlds/hull_viz.wbt`。法向可視化:`worlds/normal_viz.wbt` + `controllers/srp_normal_viz/` + `gen_normals_viz.py`。

## ★★ 指標修正(2026-09-09):堆疊 recall/precision 分母改「可觀測表面」
**Bug**:`d2_stack_clean.py` 原本 recall/precision 分母 = `surf ∩ GT 實心 mesh occ`(物體整圈表面殼),**含「任何相機都看不到最前面」的背面/底面/被壓住的 voxel**。實測 gelatin(堆疊下物)表面 570 中 **286(50%)永遠不可觀測** → 即使可見表面全標對,recall 上限也只 ~0.50 → **recall≥0.5 對堆疊下物幾乎不可能達成**。這使 baseline 堆疊分離被壓成假的 0.48。與鎖定的 `eval_surface_iou`(GT 用 modal 遮罩=可觀測)**定義不一致**,是本 session 建 d2 時的疏忽。
**修正**:`OBS_ONLY=1`(預設)——分母只算「≥1 視角 zbuffer 看得到」的表面 voxel,排掉永遠不可見的背面,與 eval_surface_iou 一致。

### 修正後的正確兩軸(取代上表堆疊欄)
| 方案 | 過切/重建 found@0.7 | 堆疊乾淨分離(可觀測,τ.5) |
|---|--:|--:|
| baseline(不併) | 0.644 | **0.66** |
| span3d | 0.899 | 0.28 |
| **divB θ.6** | **0.900** | **0.66** |

**修正後結論(誠實,取代先前「div 堆疊 0.62 > baseline 0.48」)**:
- baseline 堆疊分離 **0.48→0.66**(假低是壞分母造成)。
- **baseline 與 divB θ.6 堆疊打平(0.66)**——**div 先前的堆疊優勢大半是指標 artifact**。
- **div 的真正價值 = 堆疊分離不輸 baseline(0.66)、過切/重建大勝(0.900 vs 0.644);span3d 則過切好但堆疊崩(0.28)**。div 仍是唯一兩軸都在最佳角落者,但在堆疊上是「與 baseline 平手、且不像 span3d 搞砸」,非「贏 baseline」。
- **0.66 仍不足**(34% 堆疊未乾淨分開),尚有空間。

### 已否決的進一步修法(都刪除,含 build_meta 記錄)
拉高堆疊試過三種,**皆失敗**(放寬「同物該併」必連帶併「相觸不該併」):MIN_VOX 調小(小塊窗口太窄 14-16、調小則過度碎裂)、occ 連通閘門(連上相觸堆疊、兩軸皆退)、同群+occ(cc,CLIP 偶把跨物體塊分同群→誤併,堆疊 0.62→0.48)。roots 已刪。

## ★★★ θ 掃描 + 最佳操作點 + 根因天花板(2026-09-09 定案)

### θ 掃描(divB,可觀測指標)
`merge_div.py` 的 `THETAS` env 可設;掃 0.2–0.8:
| θ | 過切 found@0.7 | 堆疊(可觀測 τ.5) |
|---|--:|--:|
| baseline | 0.644 | 0.66 |
| 0.2 / 0.3 / 0.4 | 0.725 / 0.780 / 0.841 | 0.69 / 0.69 / 0.69 |
| **0.5** | 0.873 | **0.69** |
| 0.6 | **0.900** | 0.66 |
| 0.7 / 0.8 | 0.889 / 0.894 | 0.28 / 0.28 |
| span3d | 0.899 | 0.28 |

**堆疊封頂 0.69(θ.2–.5)、過切封頂 0.900(θ.6);兩邊界外延無改善(θ<.2→baseline、θ>.8→span3d)。**
**最佳操作點:θ.5(平衡:堆疊 0.69 最高 + 過切 0.873);θ.6(重建優先:0.900 / 0.66)。**

### 失敗分析(divB θ.5 可觀測:乾淨分開 20/29,失敗 9)
- **未分開 4**:(a)CLIP 接觸帶把兩物部分表面**分到同一群**(3:tuna/master 共群、sugar/sponge 共群、sugar/cracker 共群);(b)低-div 碎片橋接(1:stack4_0007)。
- **過切 5**:(a)同物切成多群、合併沒併回(foam_brick/gelatin×2/pudding——併回會連帶併堆疊,見 occ/cc 否決);(b)表面重度無label(sugar_box 60%、gelatin_0006 43%,接觸帶被夥伴 argmax 搶/MIN_VOX 丟/遮罩沒蓋)。

### ★ 根因天花板:CLIP 特徵「同物內差異 > 異物間差異」
實測 stack4_0012 gelatin(9 遮罩):**自己內部兩兩餘弦距離最大 0.665、中位 0.239;gelatin↔tomato_soup_can 只 0.264**(分群門檻 0.4)。
→ **要把 gelatin 散開的遮罩併成一群需門檻>0.665,但那也會併掉 gelatin+罐子(0.264)。沒有任何門檻能兩全。**
→ 過切(同物切散)與接觸帶共用群(兩物併一群)**都是這個「CLIP 特徵沒有乾淨每物一群結構」的表現**,調 sem_thr/div/θ/合併都動不了根。**突破需更強特徵(拉近同物不同面、推遠異物)或加幾何/位置訊號,不是調現有參數。**

### 已否決的拉高嘗試(全刪,實測無效)
| 嘗試 | 結果 | 為何失敗 |
|---|---|---|
| MIN_VOX 調小(20/30) | gelatin 沒改善 | 小塊在 14-16、窗口太窄,調小到能救則過度碎裂 |
| occ 連通閘門 | 兩軸皆退 | occ 連上相觸堆疊,完整度擋不住低-div 下物 |
| 同群+occ(cc) | 堆疊 0.62→0.48 | CLIP 偶把跨物體塊分同群→cc 誤併 |
| Gaussian 平滑法向 | AUC 0.888→~0.89 無改善;扁物 div 反降 | 扁物「可見面本就同向」是幾何、非噪;平滑讓它更同向→div 更低 |
| θ 掃 0.2–0.8 | 堆疊封頂 0.69 | 見上,θ 動不了 CLIP 特徵根因 |

### 檔案 / provenance
- **合併**:`merge_div.py`(θ via `THETAS` env;方法 A 純完整度閘門、B span+完整度否決;div=`group_div`)。
- **評估**:`eval_surface_iou.py`(過切,GT=modal 可觀測表面 3D-IoU)、`d2_stack_clean.py`(堆疊,`OBS_ONLY=1` 可觀測分母,已修 bug)。
- **輔助**:`eval_merge_rules.py`(signals:span/conn3d)、`eval_merge_methods.py`(BASE_ROOT env)。
- **roots**:`srp_hull_divB_t{20..80}`(build_meta 記 method/theta)。best=t50。
- **法向可視化**:`worlds/normal_viz.wbt`+`controllers/srp_normal_viz/`+`gen_normals_viz.py`。

## 作廢/取代
- 群對級 AUC(span 0.85 等):量訊號可分性非物體還原,母體被過切重複計+GLOBAL_EXCLUDE 扭曲 → **作廢**。
- 填實心 3D-IoU(`_solid`):方法只歸屬表面 voxel,填實心比實心 GT 是硬湊 → **作廢**,改用表面。
- 遮罩完全相等(9%):超嚴,一塊不差才算;保留為參考,主指標改用表面 3D-IoU。

## 腳本與 provenance
`eval_surface_iou.py`(主指標)、`d2_stack_clean.py`(乾淨分離)、`stack_separation_perpair.csv`(per-stack)、`materialize_merge.py`(span3d root)。GT=modal;預測=各 root instances.npz labels;hull=am1_photo;12 視角。

---

## ★ 舊資料兩軸確認 — divB θ.5(2026-09-09,重跑可復現)

**目的**:把 divB θ.5(`srp_hull_divB_t50`)在**舊資料**(n/occ/stack)對「合併過切」與「分開堆疊」兩軸的結果釘死並記錄。
**來源/一致**:baseline=`srp_hull_semcluster_surf_am1photo`;兩軸皆 hull=`srp_hull_mv2_v12_am1_photo`、12 視角、captures_fast、排 GLOBAL_EXCLUDE、baseline 與 divB 同一套 GT/hull 只差合併。

### ① 合併過切 `eval_surface_iou.py`(303 多物場,GT 物體 1052)
| found@ | baseline | divB θ.5 |
|---|---|---|
| 0.3 | 0.985 | 0.992 |
| 0.5 | 0.818 | **0.965** |
| 0.7 | 0.644 | **0.873** |
| 配對 mIoU | 0.750 | 0.848 |
| 預測 inst / 幻影 | 2347 / 1295 | **1414 / 365** |
→ 過切軸**大勝**:found@0.7 +0.229、幻影砍 930、預測 inst 逼近 1052。

### ② 分開堆疊 `d2_stack_clean.py`(29 可評 on 對,OBS_ONLY=1)
| success@ | baseline | divB θ.5 |
|---|---|---|
| 0.3 | 0.93 | 0.86 |
| 0.5 | 0.66 | **0.69** |
| 0.7 | 0.41 | **0.66** |
| 分開(τ0.5) | 28/29 | 25/29 |
→ @0.5 微升打平、@0.7 大勝;代價 @0.3 併掉 3 對本來分開的(28→25)。

### 9 對失敗逐條(divB θ.5,τ=0.5;`div_fail_list.py`)
**A. 未分開=誤併(4;真堆疊分離失敗)**:stack4_0007 sugar/**sponge**(P0.19)、stack4_0014 sugar/**sponge**(P0.19)、stack4_0010 tuna_can/master_chef_can(P0.27)、stack5_0019 sugar_box/cracker_box(P0.01)。→ **軟塊 sponge(2 場被吞)+ 同材質相疊(罐+罐、盒+盒)**。
**B. 已分開但一方 recall<0.5=過切/遮擋(5;precision≈1.0 無誤併)**:stack4_0006 gelatin R0.30、stack4_0012 gelatin R0.44、stack5_0020 pudding R0.35、stack3_0019 foam_brick R0.49(臨界)、stack5_0014 sugar_box R0.38。
**規律**:失敗集中在**軟塊/扁物(sponge/foam)+ 同材質盒罐 + gelatin 類**,非隨機;對應根因天花板(div 弱→假碎片、CLIP 同材質分不開)。
**provenance**:`div_fail_list.py`(scratchpad)、`d2_stack_clean.py`、`eval_surface_iou.py`;OBS_ONLY=1。

### 0.31 堆疊失敗原因逐對(divB θ.5,τ0.5,9/29;`div_fail_reason.py`+`trace_merge.py` 驗證)
**① div 誤併 3 對(完整度否決破口)**
- stack4_0010 tuna/master_chef_can:**直接誤併**——tuna_fish_can 自身 div=0.34<θ(扁罐像碎片)→ 直接邊(chef 0.73—tuna 0.34)未被否決→併。
- stack4_0007、0014 sugar/sponge:**傳遞誤併**——sugar/sponge 兩完整物(div 0.54/0.63、0.61/0.66,皆≥θ,直接邊被否決),但 sugar 被過切成低-div 碎片群(0.11~0.23);sponge 與碎片相觸被同一遮罩框住(span 0.8~1.0)→碎片不完整、否決不觸發→sponge併入碎片→碎片連回 sugar 主群,**union-find 透過碎片橋串成一塊**。
**② 上游 CLIP 早已同群 1 對**:stack5_0019 sugar/cracker(兩紙盒 baseline 同群3,div 只併不拆救不了)。
**③ 已分開但 recall<0.5 共 5 對**(分離對、失敗在覆蓋不足非合併):foam_brick 0.49(臨界)、gelatin×2、sugar_box(5_0014)、pudding_box。
**★ 統一根因**:完整度否決只擋「兩端都完整」的邊;**任一低-div 節點(真扁物如 tuna_can,或過切碎片)都是不設防的併接通道**,span 連通+union-find 把完整物透過它串起→堆疊被併。要補:對「跨 GT 物體/被否決端相鄰」的傳遞路徑加閘,或不讓碎片當橋。

### ★ photo 雕值不值:am1 vs am1+photo(divB θ.5,舊資料,2026-09-10)
同一套 GT、同分群(surf_donut),只差 hull(am1 vs am1+photo warp-NCC 雕);merge_div θ.5。
| 軸 | 指標 | am1+photo | am1(非photo) |
|---|---|---|---|
| 過切 | found@0.5 / @0.7 / mIoU / 幻影 | 0.965 / 0.873 / 0.848 / 365 | 0.964 / **0.887** / **0.855** / **325** |
| 堆疊 | success@0.3 / @0.5 / @0.7 | 0.86 / 0.69 / 0.66 | 0.79 / 0.69 / 0.62 |
**裁決**:差異全 1–4 百分點(29對/1052物雜訊級),過切軸 am1 反略勝、堆疊 @0.5 打平。**photo 雕(~12s/場,warp-NCC 15迭代)對 divB 兩軸無實質貢獻** → 新資料集(stkb)可省略,砍最大瓶頸 ~5–6h。管線=am1 hull→surf_donut→merge_div θ.5→eval。runner:`scratchpad/merge_am1.py`+`eval_am1.py`(monkeypatch HULL=am1,不動 repo);root=`srp_hull_divB_t50_am1`。

---

## ★★ 突破:connected(佔據連通拆)+ 守衛式 div = 兩軸兼得(2026-09-10)

**兩個新步驟(皆新檔,不動 surf_donut/merge_div)**:
1. **connected**(`voxel_sem_cluster_connected.py`)= surf_donut 唯一改連通:同語意群的表面 voxel 依 **hull occupancy 連通塊**分(取代原表面 26-鄰接);相觸/堆疊 hull 融一塊→不拆,分離→拆,單物表面破碎→不誤拆。build_meta `connectivity=hull_occupancy_26`。
2. **守衛式 merge**(`merge_div_guard.py`)= merge_div 唯一改 union-find:每 component 記「是否已含原始完整群(div≥θ)」,**兩塊都已含完整群→拒絕合併**(強 span 先併);擋掉「兩完整物經碎片橋接串起」。BASE 指 connected → connected→守衛div。

**四方兩軸(303 場;過切=eval_surface_iou found@,分母1052物;堆疊=d2_stack_clean success@ OBS_ONLY,分母29 on對)**

| am1_photo | 過切f@0.7 | mIoU | 幻影 | 堆疊s@0.5 | s@0.7 | 分開 |
|---|---|---|---|---|---|---|
| surf | 0.644 | 0.750 | 1295 | 0.66 | 0.41 | 28 |
| connected | 0.699 | 0.783 | 1369 | 0.83 | 0.38 | 28 |
| conn+div(無守衛) | 0.880 | 0.883 | 408 | 0.45 | 0.28 | 14 |
| **conn+div+守衛** | 0.855 | 0.876 | 525 | **0.83** | 0.52 | 26 |

| am1(無光雕) | 過切f@0.7 | mIoU | 幻影 | 堆疊s@0.5 | s@0.7 | 分開 |
|---|---|---|---|---|---|---|
| surf_donut | 0.640 | 0.749 | 1320 | 0.66 | 0.38 | 27 |
| connected | 0.700 | 0.784 | 1362 | 0.76 | 0.52 | 27 |
| conn+div(無守衛) | 0.890 | 0.889 | 377 | 0.48 | 0.45 | 16 |
| **conn+div+守衛** | 0.873 | 0.885 | 473 | **0.83** | **0.76** | 27 |

**裁決**:**單一最佳 = am1(無光雕)+ connected + div + 守衛**——過切 f@0.7 0.873/mIoU 0.885(近無守衛 div、遠勝 surf/connected),堆疊 s@0.5 0.83/s@0.7 0.76(救回 connected 水準)。守衛把無守衛 div 的堆疊崩(0.45/0.48)拉回 0.83,代價過切 f@0.7 僅 −0.02、幻影 +~120。分開對數 14→26/16→27(擋掉 ~10 對橋接誤併)。**剩死角**:扁物自身低div(tuna 罐,守衛不保護)、上游 CLIP 同群(sugar/cracker),手動 4 場只修 1。

**重現(同一套 srp 腳本 + env,無 wrapper/monkeypatch)**:
```bash
E="SAM_ROOT=$PWD/data/eval/mobilesamv2_fast CAPTURES_ROOT=$PWD/data/captures_fast"
S=$(for g in n3 n4 n5 occ3 occ4 occ5 stack3 stack4 stack5; do ls -d data/eval/srp_hull_mv2_v12_am1/${g}_scene*; done|xargs -n1 basename)
# connected(am1);報告另跑 am1_photo 版
env $E DROP_ARM=1 ARM_MASK_ROOT=$PWD/data/eval/srp_arm_masks HULL_ROOT=$PWD/data/eval/srp_hull_mv2_v12_am1 \
  OUT_ROOT=srp_hull_semcluster_connected_am1 voxel_sem_cluster_connected.py $S --n-views 12 --sem-thr 0.4
# 守衛式 div on connected
env $E THETAS=0.5 OUT_SUFFIX=_conn_guard_am1 BASE_ROOT=srp_hull_semcluster_connected_am1 \
  HULL_ROOT_NAME=srp_hull_mv2_v12_am1 merge_div_guard.py $S
# 兩軸
env $E HULL_ROOT_NAME=srp_hull_mv2_v12_am1 eval_surface_iou.py --roots srp_hull_divB_t50_conn_guard_am1
env $E HULL_ROOT_NAME=srp_hull_mv2_v12_am1 OBS_ONLY=1 d2_stack_clean.py --roots srp_hull_divB_t50_conn_guard_am1
```
roots:`srp_hull_semcluster_connected_{am1photo,am1}`、`srp_hull_divB_t50_conn{,_guard}_{am1photo,am1}`。arm 用 canonical 共用剪影。

---

## ★★ reassignNN+div+守衛:過切軸新高(2026-09-10)

**動機**:surf 乾淨底把 MIN_VOX 丟掉的散點也丟了 → 物體覆蓋不足 → div 低(如 sugar 0.41<0.5)→ 守衛認不出完整物 → 堆疊被併(surf+div+守衛 堆疊只 0.69)。**驗證(非推論)**:connected 的散點量測=60% 落在自己 GT 物體、11% 別的物體、28% 無物體(hull邊界)→ 散點主要是「投錯群但位置正確的物體真表面」,丟了才傷 div。無標籤組成(stack4_0007):A 從來不可見(遮擋)76% / B 可見無遮罩 1% / C 投票被MIN_VOX丟散點 23%。

**做法(`voxel_sem_cluster_reassign.py`,新檔)**:surf_donut 乾淨表面切塊,但**小塊(<MIN_VOX)不丟→併進「3D 最近的大實例」**(佔據可達,撐高該物 div;實例數不爆)。再接 `merge_div_guard`(θ.5)。root `srp_hull_divB_t50_reNN_guard`。

**全 303 兩軸(HULL=am1;`run_reNN_full.sh`)**:
| 方法 | 過切 found@0.7 | mIoU | 幻影 | 堆疊 s@0.5 | s@0.7 |
|---|---|---|---|---|---|
| surf_donut | 0.640 | 0.749 | 1320 | 0.66 | 0.38 |
| divB θ.5 am1(無守衛) | 0.887 | 0.855 | 325 | 0.69 | 0.66 |
| conn+div+守衛 | 0.825* | 0.850* | 84* | 0.83 | 0.76 |
| **reassignNN+div+守衛** | **0.911** | **0.930** | 357 | 0.76 | 0.72 |

(*conn+guard 過切數字為 60 stack 上量,非全 303)。**reNN = 全 303 過切最佳(mIoU 0.930、found@0.7 0.911);堆疊 0.76/0.72 > 舊 divB,略遜 conn+guard 0.83/0.76。**

**堆疊失敗(reNN,60 stack,τ0.5,7 對)**:未分開4(stack4_0007 sugar/sponge=reassign 跨接回歸、stack4_0010 tuna、stack5_0005 tuna、stack5_0014 sugar/cracker);分開但覆蓋不足3(stack4_0014 sponge P0.35、stack5_0019 cracker R0.46、stack5_0020 pudding R0.42)。**集中=扁罐 tuna(div天生低,reassign 也撐不成完整物)+ 同材質盒(上游CLIP同群/reassign跨接)。**

**兩變體區別(都 surf底→div+守衛,差在處理丟掉的 voxel)**:
- **bridge_guard**(`voxel_sem_cluster_bridge.py`,root `srp_hull_divB_t50_bridge_guard`):region-grow 時**沿表面跨無標籤縫**接(內建守衛擋兩完整物),保守只填縫、乾淨無散點。過切弱(f@0.7 0.590 on 60stack)、gelatin 類覆蓋不足救不起;堆疊 0.79/0.79。
- **reNN_guard**:把丟掉的散點**併進最近大實例**,積極回收覆蓋。過切最佳、gelatin 救起(stack4_0006/0012);但 11% 跨接偶爾傷堆疊(stack4_0007 回歸)。
**取捨**:reNN=過切最好+堆疊次佳;bridge=乾淨+堆疊穩但過切差。

**provenance**:新檔 `voxel_sem_cluster_reassign.py`(小塊併最近大實例)、`voxel_sem_cluster_bridge.py`(表面跨無標籤+守衛)、`merge_bridge_frag.py`(步驟4,未採用)、`merge_div_guard.py`(div+完整度守衛,guarded_union_find)。eval HULL 改吃 `HULL_ROOT_NAME` env。roots:`srp_hull_semcluster_reassignNN_am1`→`srp_hull_divB_t50_reNN_guard`。runner `scratchpad/run_reNN_full.sh`;失敗列 `reNN_fail.py`/`bg_fail.py`。
