# 定案指標結果表:胖瘦×中心/fp×三reassign(洩漏%/沒分開/純度)

- 建檔:2026-09-16。程式:`build_leak_tables.py`(復用 `stack_leak_nosep.py`,四錨點驗證過)。可復現:重跑本檔即得。
- 分母=同前景hull表面voxel(瘦=am1+gtlabel_am1、胖=am1fp+gtlabel_am1fp,**各自對齊、不跨hull比絕對值**);GT=gtlabel;**排GEX**(skillet_lid/windex/colored_wood_blocks/dice)。
- 量:per-object 洩漏%(無門檻)、沒分開率(主inst相同)、受污染inst純度。母體=303多物場;stack on對=29。
- 判準(不給總冠軍):單對『乾淨』=分開 且 leak低 且 純度高,三者缺一不算好;看表B逐對。

## 表A — 12 方法 × 分組概覽

| hull | 投票 | reassign | n洩漏% | occ洩漏% | stack洩漏% | stack最差洩漏%(哪對) | stack沒分開% |
|---|---|---|---|---|---|---|---|
| 瘦 | 中心 | 併最近 | 0.16 | 0.42 | 9.23 | 96.3 (stack4#0010 tuna_fish_can/master_chef_can) | 17.2 |
| 瘦 | 中心 | drop | 0.06 | 0.21 | 8.71 | 96.3 (stack4#0010 tuna_fish_can/master_chef_can) | 13.8 |
| 瘦 | 中心 | 侵蝕 | 0.06 | 0.21 | 8.64 | 96.3 (stack4#0010 tuna_fish_can/master_chef_can) | 13.8 |
| 瘦 | fp | 併最近 | 0.33 | 0.78 | 10.77 | 99.8 (stack4#0010 tuna_fish_can/master_chef_can) | 10.3 |
| 瘦 | fp | drop | 0.24 | 0.65 | 10.31 | 99.8 (stack4#0010 tuna_fish_can/master_chef_can) | 6.9 |
| 瘦 | fp | 侵蝕 | 0.24 | 0.65 | 10.21 | 99.8 (stack4#0010 tuna_fish_can/master_chef_can) | 6.9 |
| 胖 | 中心 | 併最近 | 0.20 | 1.61 | 9.71 | 97.9 (stack4#0010 tuna_fish_can/master_chef_can) | 13.8 |
| 胖 | 中心 | drop | 0.12 | 1.34 | 9.06 | 97.9 (stack4#0010 tuna_fish_can/master_chef_can) | 13.8 |
| 胖 | 中心 | 侵蝕 | 0.12 | 1.32 | 9.04 | 97.9 (stack4#0010 tuna_fish_can/master_chef_can) | 13.8 |
| 胖 | fp | 併最近 | 0.54 | 2.78 | 14.21 | 100.0 (stack4#0010 tuna_fish_can/master_chef_can) | 13.8 |
| 胖 | fp | drop | 0.36 | 2.43 | 13.97 | 100.0 (stack4#0010 tuna_fish_can/master_chef_can) | 13.8 |
| 胖 | fp | 侵蝕 | 0.36 | 2.41 | 13.96 | 100.0 (stack4#0010 tuna_fish_can/master_chef_can) | 13.8 |

## 表B — 堆疊 on 對逐對(cell = 該對最差 leak%;`✗`=沒分開;粗體=本對最佳)

| 場景 | 上物→下物 | 瘦c-S | 瘦c-Sd | 瘦c-Se | 瘦f-S | 瘦f-Sd | 瘦f-Se | 胖c-S | 胖c-Sd | 胖c-Se | 胖f-S | 胖f-Sd | 胖f-Se | 本對最佳(最低maxleak) | 最佳值 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stack3#0005 | foam_brick→gelatin_box | 20 | 76✗ | 76✗ | 6 | **3** | 3 | 18 | 17 | 17 | 9 | 12 | 12 | 瘦f-Sd | 3% |
| stack3#0006 | tomato_soup_can→tuna_fish_can | 5 | **3** | 3 | 32 | 32 | 32 | 5 | 5 | 5 | 32 | 32 | 32 | 瘦c-Sd | 3% |
| stack3#0011 | tomato_soup_can→tuna_fish_can | **4** | 4 | 4 | 27 | 27 | 27 | 5 | 5 | 5 | 28 | 28 | 28 | 瘦c-S | 4% |
| stack3#0013 | foam_brick→sponge | 12 | 8 | 8 | 3 | 3 | 3 | 17 | 13 | 13 | **2** | 2 | 2 | 胖f-S | 2% |
| stack3#0014 | tomato_soup_can→master_chef_can | 2 | 2 | 2 | 32 | 32 | 32 | 1 | **1** | 1 | 33 | 34 | 34 | 胖c-Sd | 1% |
| stack3#0015 | foam_brick→gelatin_box | 9 | 8 | 8 | 4 | 2 | 2 | 10 | 9 | 9 | **2** | 2 | 2 | 胖f-S | 2% |
| stack3#0016 | tomato_soup_can→tuna_fish_can | 3 | **3** | 3 | 28 | 28 | 28 | 4 | 4 | 4 | 29 | 29 | 29 | 瘦c-Sd | 3% |
| stack3#0017 | tomato_soup_can→master_chef_can | 1 | **1** | 1 | 2 | 2 | 2 | 2 | 1 | 1 | 3 | 3 | 3 | 瘦c-Sd | 1% |
| stack3#0019 | foam_brick→wood_block | 3 | 1 | 1 | 1 | 1 | 1 | **1** | 1 | 1 | 2 | 1 | 1 | 胖c-S | 1% |
| stack4#0002 | tomato_soup_can→tuna_fish_can | 5 | 3 | 3 | 28 | 28 | 28 | 4 | **3** | 3 | 31 | 31 | 31 | 胖c-Sd | 3% |
| stack4#0003 | master_chef_can→pudding_box | 3 | 2 | 2 | 7 | 10 | 10 | 4 | **2** | 2 | 14 | 14 | 14 | 胖c-Sd | 2% |
| stack4#0006 | tomato_soup_can→gelatin_box | 77✗ | 27 | 27 | 7 | 7 | 7 | 5 | **4** | 4 | 8 | 8 | 8 | 胖c-Sd | 4% |
| stack4#0007 | sugar_box→sponge | 64✗ | 57✗ | 57✗ | 99✗ | 92✗ | 92✗ | 64✗ | **57✗** | 57✗ | 99✗ | 83✗ | 83✗ | 胖c-Sd | 57% |
| stack4#0009 | master_chef_can→pudding_box | 5 | 3 | **3** | 16 | 15 | 15 | 24 | 22 | 22 | 15 | 15 | 15 | 瘦c-Se | 3% |
| stack4#0010 | tuna_fish_can→master_chef_can | **96✗** | 96✗ | 96✗ | 100✗ | 100✗ | 100✗ | 98✗ | 98✗ | 98✗ | 100✗ | 100✗ | 100✗ | 瘦c-S | 96% |
| stack4#0011 | tomato_soup_can→master_chef_can | 2 | 2 | 2 | 32 | 30 | 30 | **2** | 2 | 2 | 39 | 39 | 39 | 胖c-S | 2% |
| stack4#0012 | tomato_soup_can→gelatin_box | 5 | 5 | 5 | 3 | **3** | 3 | 6 | 6 | 6 | 98✗ | 97✗ | 96✗ | 瘦f-Sd | 3% |
| stack4#0014 | sugar_box→sponge | 75 | 73 | 73 | 26 | **25** | 25 | 70 | 66✗ | 66✗ | 98 | 98 | 98 | 瘦f-Sd | 25% |
| stack4#0017 | foam_brick→master_chef_can | 3 | 4 | 4 | 3 | 3 | 3 | 3 | 4 | 4 | 2 | **2** | 2 | 胖f-Sd | 2% |
| stack4#0018 | tomato_soup_can→tuna_fish_can | 4 | 3 | 3 | 2 | **2** | 2 | 5 | 5 | 5 | 5 | 5 | 5 | 瘦f-Sd | 2% |
| stack5#0005 | tuna_fish_can→pudding_box | **23** | 26 | 26 | 28 | 31 | 31 | 24 | 26 | 26 | 26 | 28 | 28 | 瘦c-S | 23% |
| stack5#0008 | tomato_soup_can→tuna_fish_can | 6 | **3** | 3 | 6 | 6 | 6 | 6 | 5 | 5 | 5 | 8 | 8 | 瘦c-Sd | 3% |
| stack5#0011 | foam_brick→sponge | 8 | 6 | 6 | **3** | 3 | 3 | 12 | 11 | 11 | 7 | 7 | 7 | 瘦f-S | 3% |
| stack5#0013 | foam_brick→sponge | 10 | 7 | 7 | **2** | 2 | 2 | 73✗ | 72✗ | 72✗ | 2 | 2 | 2 | 瘦f-S | 2% |
| stack5#0014 | sugar_box→cracker_box | 7 | 5 | 5 | 15 | 14 | 14 | 6 | **4** | 4 | 12 | 10 | 10 | 胖c-Sd | 4% |
| stack5#0016 | tomato_soup_can→wood_block | **2** | 2 | 2 | 3 | 3 | 3 | 2 | 2 | 2 | 3 | 3 | 3 | 瘦c-S | 2% |
| stack5#0018 | tomato_soup_can→gelatin_box | 3 | **1** | 1 | 4 | 4 | 4 | 3 | 3 | 3 | 5 | 5 | 5 | 瘦c-Sd | 1% |
| stack5#0019 | sugar_box→cracker_box | 32✗ | 25 | **21** | 48✗ | 35 | 30 | 40✗ | 31 | 30 | 52✗ | 51✗ | 51✗ | 瘦c-Se | 21% |
| stack5#0020 | tuna_fish_can→pudding_box | 35✗ | 37✗ | 37✗ | 31 | 30 | 30 | 30 | **30** | 30 | 32 | 32 | 32 | 胖c-Sd | 30% |

### 逐對『最佳可達 maxleak』完整排序(threshold-free;不設任何門檻,直接看連續分布與自然斷層)
| 名次 | 場景 | 上→下 | 最佳可達 maxleak% |
|---|---|---|---|
| 1 | stack4#0010 | tuna_fish_can→master_chef_can | 96 |
| 2 | stack4#0007 | sugar_box→sponge | 57 |
| 3 | stack5#0020 | tuna_fish_can→pudding_box | 30 |
| 4 | stack4#0014 | sugar_box→sponge | 25 |
| 5 | stack5#0005 | tuna_fish_can→pudding_box | 23 |
| 6 | stack5#0019 | sugar_box→cracker_box | 21 |
| 7 | stack5#0014 | sugar_box→cracker_box | 4 |
| 8 | stack4#0006 | tomato_soup_can→gelatin_box | 4 |
| 9 | stack3#0011 | tomato_soup_can→tuna_fish_can | 4 |
| 10 | stack3#0006 | tomato_soup_can→tuna_fish_can | 3 |
| 11 | stack5#0008 | tomato_soup_can→tuna_fish_can | 3 |
| 12 | stack3#0005 | foam_brick→gelatin_box | 3 |
| 13 | stack4#0002 | tomato_soup_can→tuna_fish_can | 3 |
| 14 | stack3#0016 | tomato_soup_can→tuna_fish_can | 3 |
| 15 | stack4#0012 | tomato_soup_can→gelatin_box | 3 |
| 16 | stack5#0011 | foam_brick→sponge | 3 |
| 17 | stack4#0009 | master_chef_can→pudding_box | 3 |
| 18 | stack4#0011 | tomato_soup_can→master_chef_can | 2 |
| 19 | stack4#0017 | foam_brick→master_chef_can | 2 |
| 20 | stack4#0018 | tomato_soup_can→tuna_fish_can | 2 |
| 21 | stack3#0015 | foam_brick→gelatin_box | 2 |
| 22 | stack5#0013 | foam_brick→sponge | 2 |
| 23 | stack5#0016 | tomato_soup_can→wood_block | 2 |
| 24 | stack4#0003 | master_chef_can→pudding_box | 2 |
| 25 | stack3#0013 | foam_brick→sponge | 2 |
| 26 | stack3#0014 | tomato_soup_can→master_chef_can | 1 |
| 27 | stack3#0017 | tomato_soup_can→master_chef_can | 1 |
| 28 | stack5#0018 | tomato_soup_can→gelatin_box | 1 |
| 29 | stack3#0019 | foam_brick→wood_block | 1 |

min=1% / 中位=3% / max=96%。**不設門檻**,嚴重程度由連續值與其間斷層自行呈現。