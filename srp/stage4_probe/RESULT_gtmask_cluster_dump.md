# GT 遮罩語意分群:逐場分群結果(可目視核對)

- 建檔 2026-09-23;程式 `srp/stage4_probe/gtmask_cluster_dump.py`;可復現。
- 資料 `data/eval/gt_mask_feats_v2/`(60 場 stack3/4/5、12 視角 A-3 selected)。
- 門檻用各特徵「排除 GEX」下的最佳值:clip_debias=0.35、clip_raw=0.15、dino_cw=0.40、dino_ce=0.40。
- 群組成寫法 `物體×n` = 該物體有 n 個視角被分進這一群(每物體最多 12)。
- **過切** = 同一物體散在 >1 群;**混群** = 同一群含 >1 物體。GEX 物體以 `*` 標註。

> ⚠ GT 完美遮罩下的結果,**不可**與 `srp_hull_cluster_mv2_noarm`(MobileSAMv2 遮罩、303 場)相比。

### stack3_scene0001 — 3 物體 × 12 視角 = 36 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 3 | 0 | 0 | C1: sponge×12 ｜ C2: cups×12 ｜ C3: colored_wood_blocks*×12 |
| clip_raw | 3 | 0 | 0 | C1: cups×12 ｜ C2: sponge×12 ｜ C3: colored_wood_blocks*×12 |
| dino_cw | 2 | 0 | 1 | **C1**: sponge×12 + colored_wood_blocks*×12 ｜ C2: cups×12 |
| dino_ce | 2 | 0 | 1 | **C1**: sponge×12 + colored_wood_blocks*×12 ｜ C2: cups×12 |

### stack3_scene0002 — 3 物體 × 12 視角 = 36 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 5 | 2 | 0 | C1: colored_wood_blocks*×12 ｜ C2: potted_meat_can×2 ｜ C3: potted_meat_can×10 ｜ C4: gelatin_box×11 ｜ C5: gelatin_box×1 |
| clip_raw | 8 | 2 | 0 | C1: gelatin_box×9 ｜ C2: gelatin_box×1 ｜ C3: gelatin_box×1 ｜ C4: potted_meat_can×2 ｜ C5: colored_wood_blocks*×12 ｜ C6: potted_meat_can×3 ｜ C7: potted_meat_can×7 ｜ C8: gelatin_box×1 |
| dino_cw | 4 | 2 | 1 | C1: potted_meat_can×11 ｜ C2: gelatin_box×12 ｜ C3: colored_wood_blocks*×11 ｜ **C4**: potted_meat_can×1 + colored_wood_blocks*×1 |
| dino_ce | 3 | 2 | 2 | **C1**: potted_meat_can×1 + colored_wood_blocks*×1 ｜ C2: gelatin_box×12 ｜ **C3**: potted_meat_can×11 + colored_wood_blocks*×11 |

### stack3_scene0003 — 3 物體 × 12 視角 = 36 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 5 | 2 | 0 | C1: potted_meat_can×2 ｜ C2: tennis_ball×11 ｜ C3: tennis_ball×1 ｜ C4: colored_wood_blocks*×12 ｜ C5: potted_meat_can×10 |
| clip_raw | 6 | 2 | 0 | C1: potted_meat_can×2 ｜ C2: tennis_ball×11 ｜ C3: colored_wood_blocks*×12 ｜ C4: tennis_ball×1 ｜ C5: potted_meat_can×9 ｜ C6: potted_meat_can×1 |
| dino_cw | 4 | 1 | 0 | C1: tennis_ball×12 ｜ C2: potted_meat_can×12 ｜ C3: colored_wood_blocks*×11 ｜ C4: colored_wood_blocks*×1 |
| dino_ce | 4 | 2 | 1 | C1: tennis_ball×12 ｜ **C2**: potted_meat_can×1 + colored_wood_blocks*×1 ｜ C3: colored_wood_blocks*×11 ｜ C4: potted_meat_can×11 |

### stack3_scene0004 — 3 物體 × 12 視角 = 36 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 4 | 1 | 0 | C1: sugar_box×9 ｜ C2: baseball×12 ｜ C3: colored_wood_blocks*×12 ｜ C4: sugar_box×3 |
| clip_raw | 5 | 1 | 0 | C1: sugar_box×4 ｜ C2: sugar_box×5 ｜ C3: baseball×12 ｜ C4: colored_wood_blocks*×12 ｜ C5: sugar_box×3 |
| dino_cw | 4 | 2 | 1 | C1: baseball×12 ｜ **C2**: sugar_box×1 + colored_wood_blocks*×1 ｜ C3: colored_wood_blocks*×11 ｜ C4: sugar_box×11 |
| dino_ce | 4 | 2 | 2 | C1: baseball×12 ｜ **C2**: sugar_box×1 + colored_wood_blocks*×1 ｜ C3: sugar_box×10 ｜ **C4**: colored_wood_blocks*×11 + sugar_box×1 |

### stack3_scene0005 — 3 物體 × 12 視角 = 36 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 4 | 1 | 0 | C1: gelatin_box×2 ｜ C2: gelatin_box×10 ｜ C3: racquetball×12 ｜ C4: foam_brick×12 |
| clip_raw | 5 | 2 | 0 | C1: gelatin_box×2 ｜ C2: gelatin_box×10 ｜ C3: racquetball×12 ｜ C4: foam_brick×10 ｜ C5: foam_brick×2 |
| dino_cw | 3 | 0 | 0 | C1: racquetball×12 ｜ C2: foam_brick×12 ｜ C3: gelatin_box×12 |
| dino_ce | 2 | 0 | 1 | C1: racquetball×12 ｜ **C2**: gelatin_box×12 + foam_brick×12 |

### stack3_scene0006 — 3 物體 × 12 視角 = 36 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 4 | 2 | 1 | **C1**: tuna_fish_can×11 + tomato_soup_can×3 ｜ C2: tuna_fish_can×1 ｜ C3: tomato_soup_can×9 ｜ C4: bleach_cleanser×12 |
| clip_raw | 7 | 3 | 0 | C1: tomato_soup_can×3 ｜ C2: tuna_fish_can×10 ｜ C3: tuna_fish_can×1 ｜ C4: tuna_fish_can×1 ｜ C5: tomato_soup_can×9 ｜ C6: bleach_cleanser×7 ｜ C7: bleach_cleanser×5 |
| dino_cw | 2 | 0 | 1 | C1: bleach_cleanser×12 ｜ **C2**: tuna_fish_can×12 + tomato_soup_can×12 |
| dino_ce | 3 | 2 | 2 | C1: bleach_cleanser×12 ｜ **C2**: tuna_fish_can×1 + tomato_soup_can×1 ｜ **C3**: tuna_fish_can×11 + tomato_soup_can×11 |

### stack3_scene0007 — 3 物體 × 12 視角 = 36 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 5 | 2 | 0 | C1: cracker_box×4 ｜ C2: cracker_box×8 ｜ C3: cups×5 ｜ C4: cups×7 ｜ C5: colored_wood_blocks*×12 |
| clip_raw | 7 | 2 | 0 | C1: cracker_box×4 ｜ C2: cracker_box×4 ｜ C3: cracker_box×2 ｜ C4: cracker_box×2 ｜ C5: cups×5 ｜ C6: colored_wood_blocks*×12 ｜ C7: cups×7 |
| dino_cw | 5 | 2 | 1 | **C1**: cracker_box×12 + cups×1 ｜ C2: colored_wood_blocks*×11 ｜ C3: colored_wood_blocks*×1 ｜ C4: cups×10 ｜ C5: cups×1 |
| dino_ce | 5 | 2 | 1 | **C1**: cracker_box×12 + cups×1 ｜ C2: colored_wood_blocks*×11 ｜ C3: colored_wood_blocks*×1 ｜ C4: cups×10 ｜ C5: cups×1 |

### stack3_scene0008 — 2 物體 × 12 視角 = 35 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 6 | 2 | 0 | C1: colored_wood_blocks*×2 ｜ C2: colored_wood_blocks*×9 ｜ C3: colored_wood_blocks*×1 ｜ C4: cups×2 ｜ C5: colored_wood_blocks*×12 ｜ C6: cups×9 |
| clip_raw | 6 | 2 | 0 | C1: cups×2 ｜ C2: colored_wood_blocks*×12 ｜ C3: cups×9 ｜ C4: colored_wood_blocks*×2 ｜ C5: colored_wood_blocks*×9 ｜ C6: colored_wood_blocks*×1 |
| dino_cw | 4 | 2 | 1 | C1: cups×8 ｜ C2: cups×2 ｜ C3: colored_wood_blocks*×12 ｜ **C4**: colored_wood_blocks*×12 + cups×1 |
| dino_ce | 4 | 2 | 1 | C1: cups×8 ｜ C2: colored_wood_blocks*×12 ｜ C3: cups×2 ｜ **C4**: colored_wood_blocks*×12 + cups×1 |

### stack3_scene0009 — 3 物體 × 12 視角 = 36 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 5 | 1 | 0 | C1: potted_meat_can×2 ｜ C2: potted_meat_can×3 ｜ C3: potted_meat_can×7 ｜ C4: peach×12 ｜ C5: colored_wood_blocks*×12 |
| clip_raw | 5 | 1 | 0 | C1: potted_meat_can×2 ｜ C2: potted_meat_can×3 ｜ C3: potted_meat_can×7 ｜ C4: peach×12 ｜ C5: colored_wood_blocks*×12 |
| dino_cw | 4 | 2 | 1 | C1: peach×12 ｜ C2: potted_meat_can×11 ｜ C3: colored_wood_blocks*×11 ｜ **C4**: potted_meat_can×1 + colored_wood_blocks*×1 |
| dino_ce | 3 | 2 | 2 | C1: peach×12 ｜ **C2**: potted_meat_can×2 + colored_wood_blocks*×2 ｜ **C3**: potted_meat_can×10 + colored_wood_blocks*×10 |

### stack3_scene0010 — 3 物體 × 12 視角 = 35 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 5 | 2 | 0 | C1: colored_wood_blocks*×12 ｜ C2: master_chef_can×1 ｜ C3: master_chef_can×10 ｜ C4: cracker_box×4 ｜ C5: cracker_box×8 |
| clip_raw | 7 | 2 | 0 | C1: cracker_box×4 ｜ C2: cracker_box×2 ｜ C3: cracker_box×6 ｜ C4: colored_wood_blocks*×12 ｜ C5: master_chef_can×1 ｜ C6: master_chef_can×9 ｜ C7: master_chef_can×1 |
| dino_cw | 4 | 2 | 1 | C1: master_chef_can×10 ｜ C2: cracker_box×12 ｜ **C3**: colored_wood_blocks*×11 + master_chef_can×1 ｜ C4: colored_wood_blocks*×1 |
| dino_ce | 4 | 2 | 1 | C1: master_chef_can×10 ｜ **C2**: colored_wood_blocks*×11 + master_chef_can×1 ｜ C3: cracker_box×12 ｜ C4: colored_wood_blocks*×1 |

### stack3_scene0011 — 3 物體 × 12 視角 = 36 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 3 | 1 | 1 | C1: flat_screwdriver×12 ｜ C2: tomato_soup_can×9 ｜ **C3**: tuna_fish_can×12 + tomato_soup_can×3 |
| clip_raw | 4 | 1 | 0 | C1: flat_screwdriver×12 ｜ C2: tomato_soup_can×9 ｜ C3: tomato_soup_can×3 ｜ C4: tuna_fish_can×12 |
| dino_cw | 2 | 0 | 1 | C1: flat_screwdriver×12 ｜ **C2**: tuna_fish_can×12 + tomato_soup_can×12 |
| dino_ce | 3 | 2 | 2 | C1: flat_screwdriver×12 ｜ **C2**: tuna_fish_can×1 + tomato_soup_can×1 ｜ **C3**: tuna_fish_can×11 + tomato_soup_can×11 |

### stack3_scene0012 — 3 物體 × 12 視角 = 36 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 4 | 1 | 0 | C1: potted_meat_can×2 ｜ C2: sponge×12 ｜ C3: colored_wood_blocks*×12 ｜ C4: potted_meat_can×10 |
| clip_raw | 5 | 1 | 0 | C1: potted_meat_can×2 ｜ C2: potted_meat_can×7 ｜ C3: potted_meat_can×3 ｜ C4: sponge×12 ｜ C5: colored_wood_blocks*×12 |
| dino_cw | 4 | 2 | 1 | C1: sponge×12 ｜ C2: potted_meat_can×10 ｜ C3: colored_wood_blocks*×11 ｜ **C4**: potted_meat_can×2 + colored_wood_blocks*×1 |
| dino_ce | 3 | 2 | 2 | C1: sponge×12 ｜ **C2**: potted_meat_can×1 + colored_wood_blocks*×1 ｜ **C3**: potted_meat_can×11 + colored_wood_blocks*×11 |

### stack3_scene0013 — 3 物體 × 12 視角 = 36 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 3 | 0 | 0 | C1: foam_brick×12 ｜ C2: phillips_screwdriver×12 ｜ C3: sponge×12 |
| clip_raw | 3 | 0 | 0 | C1: foam_brick×12 ｜ C2: sponge×12 ｜ C3: phillips_screwdriver×12 |
| dino_cw | 3 | 0 | 0 | C1: phillips_screwdriver×12 ｜ C2: sponge×12 ｜ C3: foam_brick×12 |
| dino_ce | 2 | 0 | 1 | C1: phillips_screwdriver×12 ｜ **C2**: sponge×12 + foam_brick×12 |

### stack3_scene0014 — 3 物體 × 12 視角 = 36 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 5 | 3 | 2 | **C1**: tomato_soup_can×9 + master_chef_can×8 ｜ C2: orange×2 ｜ C3: orange×10 ｜ **C4**: master_chef_can×3 + tomato_soup_can×3 ｜ C5: master_chef_can×1 |
| clip_raw | 9 | 3 | 0 | C1: tomato_soup_can×2 ｜ C2: tomato_soup_can×7 ｜ C3: master_chef_can×4 ｜ C4: master_chef_can×6 ｜ C5: tomato_soup_can×2 ｜ C6: tomato_soup_can×1 ｜ C7: orange×10 ｜ C8: orange×2 ｜ C9: master_chef_can×2 |
| dino_cw | 3 | 3 | 2 | C1: orange×10 ｜ **C2**: master_chef_can×10 + tomato_soup_can×9 + orange×2 ｜ **C3**: tomato_soup_can×3 + master_chef_can×2 |
| dino_ce | 3 | 3 | 2 | C1: orange×10 ｜ **C2**: master_chef_can×2 + tomato_soup_can×2 ｜ **C3**: master_chef_can×10 + tomato_soup_can×10 + orange×2 |

### stack3_scene0015 — 3 物體 × 12 視角 = 36 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 3 | 0 | 0 | C1: adjustable_wrench×12 ｜ C2: foam_brick×12 ｜ C3: gelatin_box×12 |
| clip_raw | 3 | 0 | 0 | C1: adjustable_wrench×12 ｜ C2: gelatin_box×12 ｜ C3: foam_brick×12 |
| dino_cw | 3 | 0 | 0 | C1: adjustable_wrench×12 ｜ C2: foam_brick×12 ｜ C3: gelatin_box×12 |
| dino_ce | 2 | 0 | 1 | C1: adjustable_wrench×12 ｜ **C2**: gelatin_box×12 + foam_brick×12 |

### stack3_scene0016 — 3 物體 × 12 視角 = 36 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 3 | 1 | 1 | C1: pudding_box×12 ｜ C2: tomato_soup_can×9 ｜ **C3**: tuna_fish_can×12 + tomato_soup_can×3 |
| clip_raw | 6 | 3 | 0 | C1: pudding_box×7 ｜ C2: pudding_box×5 ｜ C3: tomato_soup_can×9 ｜ C4: tuna_fish_can×6 ｜ C5: tomato_soup_can×3 ｜ C6: tuna_fish_can×6 |
| dino_cw | 3 | 2 | 2 | C1: pudding_box×12 ｜ **C2**: tomato_soup_can×3 + tuna_fish_can×1 ｜ **C3**: tuna_fish_can×11 + tomato_soup_can×9 |
| dino_ce | 3 | 2 | 2 | C1: pudding_box×12 ｜ **C2**: tuna_fish_can×1 + tomato_soup_can×1 ｜ **C3**: tuna_fish_can×11 + tomato_soup_can×11 |

### stack3_scene0017 — 3 物體 × 12 視角 = 36 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 4 | 2 | 2 | C1: baseball×12 ｜ **C2**: tomato_soup_can×9 + master_chef_can×8 ｜ **C3**: master_chef_can×3 + tomato_soup_can×3 ｜ C4: master_chef_can×1 |
| clip_raw | 6 | 2 | 0 | C1: baseball×12 ｜ C2: tomato_soup_can×3 ｜ C3: master_chef_can×7 ｜ C4: master_chef_can×1 ｜ C5: master_chef_can×4 ｜ C6: tomato_soup_can×9 |
| dino_cw | 3 | 2 | 2 | C1: baseball×12 ｜ **C2**: master_chef_can×10 + tomato_soup_can×9 ｜ **C3**: tomato_soup_can×3 + master_chef_can×2 |
| dino_ce | 3 | 2 | 2 | C1: baseball×12 ｜ **C2**: master_chef_can×10 + tomato_soup_can×9 ｜ **C3**: tomato_soup_can×3 + master_chef_can×2 |

### stack3_scene0018 — 3 物體 × 12 視角 = 36 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 4 | 1 | 0 | C1: colored_wood_blocks*×12 ｜ C2: sugar_box×1 ｜ C3: gelatin_box×12 ｜ C4: sugar_box×11 |
| clip_raw | 6 | 2 | 0 | C1: gelatin_box×10 ｜ C2: gelatin_box×1 ｜ C3: gelatin_box×1 ｜ C4: sugar_box×7 ｜ C5: colored_wood_blocks*×12 ｜ C6: sugar_box×5 |
| dino_cw | 4 | 2 | 1 | **C1**: sugar_box×1 + colored_wood_blocks*×1 ｜ C2: gelatin_box×12 ｜ C3: sugar_box×11 ｜ C4: colored_wood_blocks*×11 |
| dino_ce | 4 | 2 | 1 | **C1**: sugar_box×1 + colored_wood_blocks*×1 ｜ C2: gelatin_box×12 ｜ C3: sugar_box×11 ｜ C4: colored_wood_blocks*×11 |

### stack3_scene0019 — 3 物體 × 12 視角 = 36 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 5 | 1 | 0 | C1: wood_block×8 ｜ C2: wood_block×3 ｜ C3: wood_block×1 ｜ C4: mug×12 ｜ C5: foam_brick×12 |
| clip_raw | 4 | 1 | 0 | C1: wood_block×12 ｜ C2: mug×12 ｜ C3: foam_brick×11 ｜ C4: foam_brick×1 |
| dino_cw | 3 | 0 | 0 | C1: mug×12 ｜ C2: wood_block×12 ｜ C3: foam_brick×12 |
| dino_ce | 3 | 1 | 1 | C1: mug×12 ｜ C2: foam_brick×11 ｜ **C3**: wood_block×12 + foam_brick×1 |

### stack3_scene0020 — 3 物體 × 12 視角 = 36 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 3 | 0 | 0 | C1: colored_wood_blocks*×12 ｜ C2: cups×12 ｜ C3: tuna_fish_can×12 |
| clip_raw | 3 | 0 | 0 | C1: colored_wood_blocks*×12 ｜ C2: cups×12 ｜ C3: tuna_fish_can×12 |
| dino_cw | 4 | 1 | 0 | C1: colored_wood_blocks*×12 ｜ C2: tuna_fish_can×12 ｜ C3: cups×11 ｜ C4: cups×1 |
| dino_ce | 2 | 1 | 1 | C1: cups×11 ｜ **C2**: tuna_fish_can×12 + colored_wood_blocks*×12 + cups×1 |

### stack4_scene0001 — 4 物體 × 12 視角 = 47 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 5 | 1 | 1 | C1: cups×12 ｜ **C2**: colored_wood_blocks*×12 + racquetball×11 ｜ C3: wood_block×10 ｜ C4: wood_block×1 ｜ C5: wood_block×1 |
| clip_raw | 5 | 1 | 1 | C1: cups×12 ｜ **C2**: colored_wood_blocks*×12 + racquetball×11 ｜ C3: wood_block×10 ｜ C4: wood_block×1 ｜ C5: wood_block×1 |
| dino_cw | 4 | 0 | 0 | C1: wood_block×12 ｜ C2: colored_wood_blocks*×12 ｜ C3: cups×12 ｜ C4: racquetball×11 |
| dino_ce | 4 | 0 | 0 | C1: wood_block×12 ｜ C2: colored_wood_blocks*×12 ｜ C3: cups×12 ｜ C4: racquetball×11 |

### stack4_scene0002 — 4 物體 × 12 視角 = 48 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 5 | 2 | 1 | C1: knife×12 ｜ C2: colored_wood_blocks*×2 ｜ C3: colored_wood_blocks*×10 ｜ C4: tomato_soup_can×10 ｜ **C5**: tuna_fish_can×12 + tomato_soup_can×2 |
| clip_raw | 6 | 3 | 1 | C1: knife×12 ｜ C2: tomato_soup_can×10 ｜ C3: colored_wood_blocks*×2 ｜ C4: colored_wood_blocks*×10 ｜ C5: tuna_fish_can×7 ｜ **C6**: tuna_fish_can×5 + tomato_soup_can×2 |
| dino_cw | 4 | 1 | 1 | C1: knife×12 ｜ C2: colored_wood_blocks*×9 ｜ **C3**: tuna_fish_can×12 + tomato_soup_can×12 ｜ C4: colored_wood_blocks*×3 |
| dino_ce | 5 | 3 | 2 | C1: knife×12 ｜ C2: colored_wood_blocks*×10 ｜ C3: colored_wood_blocks*×2 ｜ **C4**: tuna_fish_can×1 + tomato_soup_can×1 ｜ **C5**: tuna_fish_can×11 + tomato_soup_can×11 |

### stack4_scene0003 — 4 物體 × 12 視角 = 48 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 6 | 2 | 0 | C1: pudding_box×9 ｜ C2: pudding_box×3 ｜ C3: adjustable_wrench×12 ｜ C4: master_chef_can×11 ｜ C5: apple×12 ｜ C6: master_chef_can×1 |
| clip_raw | 7 | 2 | 0 | C1: master_chef_can×7 ｜ C2: master_chef_can×4 ｜ C3: apple×12 ｜ C4: master_chef_can×1 ｜ C5: pudding_box×9 ｜ C6: pudding_box×3 ｜ C7: adjustable_wrench×12 |
| dino_cw | 6 | 2 | 0 | C1: adjustable_wrench×12 ｜ C2: apple×12 ｜ C3: pudding_box×1 ｜ C4: master_chef_can×1 ｜ C5: master_chef_can×11 ｜ C6: pudding_box×11 |
| dino_ce | 4 | 2 | 2 | C1: adjustable_wrench×12 ｜ C2: apple×12 ｜ **C3**: pudding_box×1 + master_chef_can×1 ｜ **C4**: pudding_box×11 + master_chef_can×11 |

### stack4_scene0004 — 4 物體 × 12 視角 = 46 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 9 | 3 | 0 | C1: bleach_cleanser×2 ｜ C2: bleach_cleanser×7 ｜ C3: bleach_cleanser×1 ｜ C4: wood_block×11 ｜ C5: wood_block×1 ｜ C6: rubiks_cube×12 ｜ C7: colored_wood_blocks*×2 ｜ C8: colored_wood_blocks*×7 ｜ C9: colored_wood_blocks*×3 |
| clip_raw | 10 | 3 | 0 | C1: colored_wood_blocks*×2 ｜ C2: rubiks_cube×12 ｜ C3: wood_block×10 ｜ C4: wood_block×1 ｜ C5: wood_block×1 ｜ C6: colored_wood_blocks*×3 ｜ C7: colored_wood_blocks*×7 ｜ C8: bleach_cleanser×2 ｜ C9: bleach_cleanser×6 ｜ C10: bleach_cleanser×2 |
| dino_cw | 7 | 3 | 0 | C1: bleach_cleanser×9 ｜ C2: bleach_cleanser×1 ｜ C3: rubiks_cube×12 ｜ C4: colored_wood_blocks*×9 ｜ C5: colored_wood_blocks*×3 ｜ C6: wood_block×11 ｜ C7: wood_block×1 |
| dino_ce | 7 | 3 | 0 | C1: bleach_cleanser×9 ｜ C2: bleach_cleanser×1 ｜ C3: rubiks_cube×12 ｜ C4: colored_wood_blocks*×11 ｜ C5: colored_wood_blocks*×1 ｜ C6: wood_block×11 ｜ C7: wood_block×1 |

### stack4_scene0005 — 4 物體 × 12 視角 = 48 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 4 | 0 | 0 | C1: fork×12 ｜ C2: sponge×12 ｜ C3: colored_wood_blocks*×12 ｜ C4: cups×12 |
| clip_raw | 4 | 0 | 0 | C1: fork×12 ｜ C2: cups×12 ｜ C3: sponge×12 ｜ C4: colored_wood_blocks*×12 |
| dino_cw | 3 | 0 | 1 | C1: fork×12 ｜ C2: cups×12 ｜ **C3**: sponge×12 + colored_wood_blocks*×12 |
| dino_ce | 3 | 0 | 1 | C1: fork×12 ｜ C2: cups×12 ｜ **C3**: sponge×12 + colored_wood_blocks*×12 |

### stack4_scene0006 — 4 物體 × 12 視角 = 48 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 7 | 3 | 0 | C1: tomato_soup_can×3 ｜ C2: tomato_soup_can×9 ｜ C3: gelatin_box×2 ｜ C4: gelatin_box×10 ｜ C5: cups×12 ｜ C6: colored_wood_blocks*×2 ｜ C7: colored_wood_blocks*×10 |
| clip_raw | 7 | 3 | 0 | C1: tomato_soup_can×3 ｜ C2: tomato_soup_can×9 ｜ C3: gelatin_box×2 ｜ C4: gelatin_box×10 ｜ C5: cups×12 ｜ C6: colored_wood_blocks*×2 ｜ C7: colored_wood_blocks*×10 |
| dino_cw | 5 | 1 | 0 | C1: cups×12 ｜ C2: colored_wood_blocks*×2 ｜ C3: colored_wood_blocks*×10 ｜ C4: gelatin_box×12 ｜ C5: tomato_soup_can×12 |
| dino_ce | 5 | 3 | 2 | C1: cups×12 ｜ **C2**: gelatin_box×2 + tomato_soup_can×2 ｜ **C3**: gelatin_box×10 + tomato_soup_can×10 ｜ C4: colored_wood_blocks*×2 ｜ C5: colored_wood_blocks*×10 |

### stack4_scene0007 — 4 物體 × 12 視角 = 48 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 6 | 2 | 0 | C1: tomato_soup_can×12 ｜ C2: sugar_box×2 ｜ C3: sugar_box×10 ｜ C4: sponge×12 ｜ C5: lemon×10 ｜ C6: lemon×2 |
| clip_raw | 8 | 3 | 0 | C1: sugar_box×4 ｜ C2: sugar_box×6 ｜ C3: tomato_soup_can×2 ｜ C4: tomato_soup_can×10 ｜ C5: lemon×10 ｜ C6: sugar_box×2 ｜ C7: lemon×2 ｜ C8: sponge×12 |
| dino_cw | 6 | 3 | 2 | C1: lemon×10 ｜ C2: tomato_soup_can×12 ｜ **C3**: sugar_box×10 + lemon×1 ｜ C4: lemon×1 ｜ C5: sponge×10 ｜ **C6**: sponge×2 + sugar_box×2 |
| dino_ce | 5 | 3 | 2 | C1: lemon×10 ｜ C2: tomato_soup_can×12 ｜ **C3**: sugar_box×8 + lemon×2 ｜ C4: sponge×8 ｜ **C5**: sponge×4 + sugar_box×4 |

### stack4_scene0008 — 4 物體 × 12 視角 = 48 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 4 | 1 | 1 | C1: master_chef_can×11 ｜ **C2**: cups×12 + master_chef_can×1 ｜ C3: orange×12 ｜ C4: colored_wood_blocks*×12 |
| clip_raw | 4 | 1 | 1 | C1: master_chef_can×9 ｜ **C2**: cups×12 + master_chef_can×3 ｜ C3: orange×12 ｜ C4: colored_wood_blocks*×12 |
| dino_cw | 5 | 1 | 0 | C1: orange×12 ｜ C2: cups×12 ｜ C3: colored_wood_blocks*×12 ｜ C4: master_chef_can×10 ｜ C5: master_chef_can×2 |
| dino_ce | 4 | 0 | 0 | C1: orange×12 ｜ C2: cups×12 ｜ C3: master_chef_can×12 ｜ C4: colored_wood_blocks*×12 |

### stack4_scene0009 — 4 物體 × 12 視角 = 47 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 8 | 3 | 0 | C1: mustard_bottle×6 ｜ C2: mustard_bottle×6 ｜ C3: master_chef_can×11 ｜ C4: master_chef_can×1 ｜ C5: knife×12 ｜ C6: pudding_box×3 ｜ C7: pudding_box×7 ｜ C8: pudding_box×1 |
| clip_raw | 10 | 3 | 0 | C1: knife×12 ｜ C2: pudding_box×4 ｜ C3: pudding_box×6 ｜ C4: pudding_box×1 ｜ C5: mustard_bottle×6 ｜ C6: mustard_bottle×3 ｜ C7: mustard_bottle×3 ｜ C8: master_chef_can×3 ｜ C9: master_chef_can×8 ｜ C10: master_chef_can×1 |
| dino_cw | 7 | 3 | 0 | C1: knife×9 ｜ C2: knife×3 ｜ C3: mustard_bottle×4 ｜ C4: mustard_bottle×8 ｜ C5: pudding_box×10 ｜ C6: master_chef_can×12 ｜ C7: pudding_box×1 |
| dino_ce | 7 | 3 | 0 | C1: knife×9 ｜ C2: knife×3 ｜ C3: mustard_bottle×4 ｜ C4: mustard_bottle×8 ｜ C5: pudding_box×10 ｜ C6: master_chef_can×12 ｜ C7: pudding_box×1 |

### stack4_scene0010 — 4 物體 × 12 視角 = 46 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 7 | 3 | 0 | C1: windex_bottle*×6 ｜ C2: windex_bottle*×6 ｜ C3: tuna_fish_can×12 ｜ C4: master_chef_can×9 ｜ C5: lemon×9 ｜ C6: master_chef_can×3 ｜ C7: lemon×1 |
| clip_raw | 8 | 4 | 1 | C1: windex_bottle*×6 ｜ C2: lemon×9 ｜ **C3**: master_chef_can×3 + lemon×1 ｜ C4: windex_bottle*×6 ｜ C5: master_chef_can×4 ｜ C6: master_chef_can×5 ｜ C7: tuna_fish_can×6 ｜ C8: tuna_fish_can×6 |
| dino_cw | 5 | 2 | 1 | C1: lemon×2 ｜ C2: lemon×7 ｜ C3: windex_bottle*×12 ｜ **C4**: tuna_fish_can×12 + master_chef_can×11 + lemon×1 ｜ C5: master_chef_can×1 |
| dino_ce | 5 | 3 | 2 | C1: lemon×2 ｜ C2: lemon×7 ｜ C3: windex_bottle*×12 ｜ **C4**: master_chef_can×1 + tuna_fish_can×1 ｜ **C5**: master_chef_can×11 + tuna_fish_can×11 + lemon×1 |

### stack4_scene0011 — 4 物體 × 12 視角 = 48 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 5 | 3 | 2 | C1: peach×11 ｜ C2: peach×1 ｜ C3: sponge×12 ｜ **C4**: master_chef_can×3 + tomato_soup_can×2 ｜ **C5**: tomato_soup_can×10 + master_chef_can×9 |
| clip_raw | 7 | 3 | 0 | C1: tomato_soup_can×10 ｜ C2: master_chef_can×9 ｜ C3: peach×2 ｜ C4: peach×10 ｜ C5: sponge×12 ｜ C6: master_chef_can×3 ｜ C7: tomato_soup_can×2 |
| dino_cw | 5 | 3 | 2 | C1: sponge×12 ｜ **C2**: master_chef_can×1 + tomato_soup_can×1 ｜ **C3**: master_chef_can×11 + tomato_soup_can×11 ｜ C4: peach×11 ｜ C5: peach×1 |
| dino_ce | 5 | 3 | 2 | C1: sponge×12 ｜ **C2**: master_chef_can×1 + tomato_soup_can×1 ｜ **C3**: master_chef_can×11 + tomato_soup_can×11 ｜ C4: peach×11 ｜ C5: peach×1 |

### stack4_scene0012 — 4 物體 × 12 視角 = 48 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 7 | 3 | 1 | C1: tomato_soup_can×9 ｜ C2: gelatin_box×10 ｜ C3: gelatin_box×1 ｜ C4: banana×12 ｜ C5: spoon×10 ｜ C6: tomato_soup_can×3 ｜ **C7**: spoon×2 + gelatin_box×1 |
| clip_raw | 7 | 3 | 1 | C1: tomato_soup_can×9 ｜ C2: banana×12 ｜ C3: gelatin_box×10 ｜ **C4**: spoon×2 + gelatin_box×1 ｜ C5: gelatin_box×1 ｜ C6: tomato_soup_can×3 ｜ C7: spoon×10 |
| dino_cw | 5 | 1 | 1 | C1: banana×12 ｜ C2: gelatin_box×12 ｜ **C3**: tomato_soup_can×12 + spoon×1 ｜ C4: spoon×10 ｜ C5: spoon×1 |
| dino_ce | 5 | 3 | 2 | C1: banana×12 ｜ **C2**: gelatin_box×2 + tomato_soup_can×2 ｜ **C3**: gelatin_box×10 + tomato_soup_can×10 + spoon×1 ｜ C4: spoon×9 ｜ C5: spoon×2 |

### stack4_scene0013 — 4 物體 × 12 視角 = 47 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 5 | 1 | 0 | C1: cracker_box×12 ｜ C2: cups×2 ｜ C3: colored_wood_blocks*×12 ｜ C4: pear×12 ｜ C5: cups×9 |
| clip_raw | 8 | 2 | 1 | C1: cracker_box×4 ｜ C2: cracker_box×4 ｜ C3: cracker_box×2 ｜ C4: cracker_box×2 ｜ C5: colored_wood_blocks*×12 ｜ C6: cups×6 ｜ C7: cups×2 ｜ **C8**: pear×12 + cups×3 |
| dino_cw | 8 | 3 | 1 | C1: pear×9 ｜ C2: pear×3 ｜ C3: cups×8 ｜ C4: cups×1 ｜ **C5**: colored_wood_blocks*×6 + cups×1 ｜ C6: colored_wood_blocks*×6 ｜ C7: cracker_box×12 ｜ C8: cups×1 |
| dino_ce | 7 | 2 | 1 | C1: pear×9 ｜ C2: cups×8 ｜ C3: cups×1 ｜ C4: pear×3 ｜ C5: cracker_box×12 ｜ **C6**: colored_wood_blocks*×12 + cups×1 ｜ C7: cups×1 |

### stack4_scene0014 — 4 物體 × 12 視角 = 44 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 8 | 3 | 0 | C1: rubiks_cube×1 ｜ C2: sugar_box×1 ｜ C3: sugar_box×10 ｜ C4: sugar_box×1 ｜ C5: pitcher_base×12 ｜ C6: sponge×10 ｜ C7: rubiks_cube×8 ｜ C8: sponge×1 |
| clip_raw | 9 | 3 | 1 | C1: sugar_box×4 ｜ C2: sugar_box×3 ｜ C3: sugar_box×3 ｜ C4: pitcher_base×12 ｜ C5: rubiks_cube×8 ｜ C6: sponge×10 ｜ **C7**: rubiks_cube×1 + sugar_box×1 ｜ C8: sponge×1 ｜ C9: sugar_box×1 |
| dino_cw | 7 | 4 | 3 | C1: pitcher_base×6 ｜ C2: pitcher_base×6 ｜ **C3**: sponge×1 + sugar_box×1 ｜ **C4**: sponge×1 + sugar_box×1 ｜ C5: sponge×9 ｜ C6: rubiks_cube×8 ｜ **C7**: sugar_box×10 + rubiks_cube×1 |
| dino_ce | 6 | 4 | 3 | C1: pitcher_base×6 ｜ C2: pitcher_base×6 ｜ **C3**: sponge×1 + sugar_box×1 ｜ C4: rubiks_cube×8 ｜ **C5**: sugar_box×10 + rubiks_cube×1 ｜ **C6**: sponge×10 + sugar_box×1 |

### stack4_scene0015 — 4 物體 × 12 視角 = 44 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 7 | 4 | 1 | C1: tomato_soup_can×9 ｜ C2: pudding_box×10 ｜ C3: colored_wood_blocks*×3 ｜ C4: colored_wood_blocks*×9 ｜ C5: tomato_soup_can×3 ｜ **C6**: pudding_box×1 + padlock×1 ｜ C7: padlock×8 |
| clip_raw | 11 | 4 | 1 | C1: pudding_box×8 ｜ C2: pudding_box×1 ｜ C3: pudding_box×1 ｜ C4: tomato_soup_can×7 ｜ C5: tomato_soup_can×2 ｜ C6: colored_wood_blocks*×3 ｜ C7: colored_wood_blocks*×7 ｜ C8: colored_wood_blocks*×2 ｜ C9: tomato_soup_can×3 ｜ **C10**: pudding_box×1 + padlock×1 ｜ C11: padlock×8 |
| dino_cw | 6 | 3 | 1 | C1: pudding_box×10 ｜ C2: colored_wood_blocks*×12 ｜ C3: padlock×4 ｜ **C4**: padlock×5 + pudding_box×1 ｜ C5: tomato_soup_can×11 ｜ C6: tomato_soup_can×1 |
| dino_ce | 6 | 4 | 2 | C1: pudding_box×10 ｜ C2: padlock×4 ｜ C3: padlock×4 ｜ **C4**: colored_wood_blocks*×1 + tomato_soup_can×1 ｜ C5: tomato_soup_can×11 ｜ **C6**: colored_wood_blocks*×11 + pudding_box×1 + padlock×1 |

### stack4_scene0016 — 4 物體 × 12 視角 = 46 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 7 | 2 | 0 | C1: banana×12 ｜ C2: cups×3 ｜ C3: colored_wood_blocks*×12 ｜ C4: cups×7 ｜ C5: sugar_box×8 ｜ C6: sugar_box×3 ｜ C7: sugar_box×1 |
| clip_raw | 7 | 2 | 0 | C1: sugar_box×3 ｜ C2: sugar_box×5 ｜ C3: banana×12 ｜ C4: sugar_box×4 ｜ C5: cups×7 ｜ C6: cups×3 ｜ C7: colored_wood_blocks*×12 |
| dino_cw | 5 | 3 | 2 | C1: banana×12 ｜ C2: cups×8 ｜ **C3**: sugar_box×2 + colored_wood_blocks*×2 ｜ C4: sugar_box×10 ｜ **C5**: colored_wood_blocks*×10 + cups×2 |
| dino_ce | 5 | 3 | 2 | C1: banana×12 ｜ C2: cups×8 ｜ **C3**: colored_wood_blocks*×4 + sugar_box×2 ｜ C4: sugar_box×10 ｜ **C5**: colored_wood_blocks*×8 + cups×2 |

### stack4_scene0017 — 4 物體 × 12 視角 = 48 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 9 | 3 | 0 | C1: golf_ball×1 ｜ C2: golf_ball×1 ｜ C3: cracker_box×11 ｜ C4: cracker_box×1 ｜ C5: master_chef_can×4 ｜ C6: master_chef_can×7 ｜ C7: foam_brick×12 ｜ C8: golf_ball×10 ｜ C9: master_chef_can×1 |
| clip_raw | 11 | 3 | 0 | C1: cracker_box×6 ｜ C2: cracker_box×1 ｜ C3: cracker_box×2 ｜ C4: cracker_box×2 ｜ C5: cracker_box×1 ｜ C6: master_chef_can×4 ｜ C7: master_chef_can×7 ｜ C8: golf_ball×2 ｜ C9: foam_brick×12 ｜ C10: golf_ball×10 ｜ C11: master_chef_can×1 |
| dino_cw | 6 | 3 | 2 | C1: golf_ball×10 ｜ **C2**: master_chef_can×1 + foam_brick×1 ｜ C3: foam_brick×11 ｜ C4: golf_ball×1 ｜ C5: cracker_box×12 ｜ **C6**: master_chef_can×11 + golf_ball×1 |
| dino_ce | 5 | 3 | 3 | C1: golf_ball×10 ｜ **C2**: master_chef_can×1 + foam_brick×1 ｜ **C3**: foam_brick×11 + golf_ball×1 ｜ C4: cracker_box×12 ｜ **C5**: master_chef_can×11 + golf_ball×1 |

### stack4_scene0018 — 4 物體 × 12 視角 = 48 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 5 | 2 | 1 | C1: cracker_box×4 ｜ C2: cracker_box×8 ｜ C3: foam_brick×12 ｜ C4: tomato_soup_can×8 ｜ **C5**: tuna_fish_can×12 + tomato_soup_can×4 |
| clip_raw | 9 | 3 | 0 | C1: cracker_box×8 ｜ C2: cracker_box×2 ｜ C3: cracker_box×1 ｜ C4: cracker_box×1 ｜ C5: tomato_soup_can×8 ｜ C6: foam_brick×12 ｜ C7: tomato_soup_can×4 ｜ C8: tuna_fish_can×2 ｜ C9: tuna_fish_can×10 |
| dino_cw | 3 | 0 | 1 | C1: foam_brick×12 ｜ C2: cracker_box×12 ｜ **C3**: tuna_fish_can×12 + tomato_soup_can×12 |
| dino_ce | 3 | 0 | 1 | C1: foam_brick×12 ｜ C2: cracker_box×12 ｜ **C3**: tuna_fish_can×12 + tomato_soup_can×12 |

### stack4_scene0019 — 4 物體 × 12 視角 = 48 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 6 | 2 | 0 | C1: wood_block×10 ｜ C2: wood_block×2 ｜ C3: medium_clamp×2 ｜ C4: medium_clamp×10 ｜ C5: strawberry×12 ｜ C6: colored_wood_blocks*×12 |
| clip_raw | 7 | 2 | 0 | C1: wood_block×10 ｜ C2: wood_block×1 ｜ C3: wood_block×1 ｜ C4: medium_clamp×2 ｜ C5: colored_wood_blocks*×12 ｜ C6: medium_clamp×10 ｜ C7: strawberry×12 |
| dino_cw | 6 | 2 | 0 | C1: strawberry×12 ｜ C2: wood_block×12 ｜ C3: colored_wood_blocks*×11 ｜ C4: colored_wood_blocks*×1 ｜ C5: medium_clamp×2 ｜ C6: medium_clamp×10 |
| dino_ce | 6 | 3 | 1 | C1: strawberry×12 ｜ C2: medium_clamp×10 ｜ C3: medium_clamp×2 ｜ **C4**: wood_block×2 + colored_wood_blocks*×1 ｜ C5: wood_block×10 ｜ C6: colored_wood_blocks*×11 |

### stack4_scene0020 — 4 物體 × 12 視角 = 48 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 5 | 1 | 0 | C1: potted_meat_can×2 ｜ C2: golf_ball×12 ｜ C3: colored_wood_blocks*×12 ｜ C4: cups×12 ｜ C5: potted_meat_can×10 |
| clip_raw | 6 | 1 | 0 | C1: potted_meat_can×2 ｜ C2: potted_meat_can×9 ｜ C3: potted_meat_can×1 ｜ C4: cups×12 ｜ C5: colored_wood_blocks*×12 ｜ C6: golf_ball×12 |
| dino_cw | 5 | 1 | 0 | C1: golf_ball×12 ｜ C2: cups×12 ｜ C3: potted_meat_can×12 ｜ C4: colored_wood_blocks*×11 ｜ C5: colored_wood_blocks*×1 |
| dino_ce | 5 | 2 | 1 | C1: golf_ball×12 ｜ C2: cups×12 ｜ **C3**: potted_meat_can×1 + colored_wood_blocks*×1 ｜ C4: colored_wood_blocks*×11 ｜ C5: potted_meat_can×11 |

### stack5_scene0001 — 5 物體 × 12 視角 = 58 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 9 | 4 | 1 | C1: master_chef_can×10 ｜ C2: cracker_box×4 ｜ C3: cracker_box×8 ｜ C4: flat_screwdriver×10 ｜ C5: master_chef_can×2 ｜ C6: flat_screwdriver×1 ｜ **C7**: colored_wood_blocks*×12 + flat_screwdriver×1 ｜ C8: apple×9 ｜ C9: apple×1 |
| clip_raw | 10 | 4 | 2 | C1: cracker_box×4 ｜ C2: cracker_box×4 ｜ C3: cracker_box×2 ｜ C4: cracker_box×2 ｜ C5: master_chef_can×6 ｜ C6: master_chef_can×4 ｜ C7: flat_screwdriver×10 ｜ **C8**: master_chef_can×2 + apple×1 + flat_screwdriver×1 ｜ C9: apple×9 ｜ **C10**: colored_wood_blocks*×12 + flat_screwdriver×1 |
| dino_cw | 10 | 3 | 0 | C1: apple×9 ｜ C2: flat_screwdriver×9 ｜ C3: flat_screwdriver×2 ｜ C4: apple×1 ｜ C5: master_chef_can×12 ｜ C6: flat_screwdriver×1 ｜ C7: cracker_box×12 ｜ C8: colored_wood_blocks*×6 ｜ C9: colored_wood_blocks*×5 ｜ C10: colored_wood_blocks*×1 |
| dino_ce | 9 | 3 | 0 | C1: apple×9 ｜ C2: flat_screwdriver×9 ｜ C3: flat_screwdriver×2 ｜ C4: flat_screwdriver×1 ｜ C5: apple×1 ｜ C6: master_chef_can×12 ｜ C7: colored_wood_blocks*×11 ｜ C8: cracker_box×12 ｜ C9: colored_wood_blocks*×1 |

### stack5_scene0002 — 5 物體 × 12 視角 = 60 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 7 | 3 | 1 | C1: fork×12 ｜ **C2**: cups×12 + master_chef_can×1 ｜ C3: colored_wood_blocks*×11 ｜ C4: colored_wood_blocks*×1 ｜ C5: master_chef_can×11 ｜ C6: bleach_cleanser×2 ｜ C7: bleach_cleanser×10 |
| clip_raw | 9 | 2 | 1 | C1: fork×12 ｜ C2: bleach_cleanser×1 ｜ C3: bleach_cleanser×1 ｜ **C4**: cups×12 + master_chef_can×1 ｜ C5: colored_wood_blocks*×12 ｜ C6: bleach_cleanser×2 ｜ C7: bleach_cleanser×8 ｜ C8: master_chef_can×4 ｜ C9: master_chef_can×7 |
| dino_cw | 7 | 2 | 0 | C1: fork×12 ｜ C2: cups×12 ｜ C3: bleach_cleanser×11 ｜ C4: bleach_cleanser×1 ｜ C5: colored_wood_blocks*×12 ｜ C6: master_chef_can×11 ｜ C7: master_chef_can×1 |
| dino_ce | 7 | 2 | 0 | C1: fork×12 ｜ C2: cups×12 ｜ C3: bleach_cleanser×11 ｜ C4: bleach_cleanser×1 ｜ C5: colored_wood_blocks*×12 ｜ C6: master_chef_can×11 ｜ C7: master_chef_can×1 |

### stack5_scene0003 — 5 物體 × 12 視角 = 60 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 6 | 1 | 1 | **C1**: medium_clamp×12 + windex_bottle*×1 ｜ C2: windex_bottle*×4 ｜ C3: windex_bottle*×7 ｜ C4: peach×12 ｜ C5: colored_wood_blocks*×12 ｜ C6: sponge×12 |
| clip_raw | 6 | 1 | 1 | C1: windex_bottle*×4 ｜ C2: windex_bottle*×7 ｜ **C3**: medium_clamp×12 + windex_bottle*×1 ｜ C4: peach×12 ｜ C5: colored_wood_blocks*×12 ｜ C6: sponge×12 |
| dino_cw | 6 | 1 | 1 | C1: peach×12 ｜ C2: windex_bottle*×10 ｜ C3: windex_bottle*×1 ｜ **C4**: sponge×12 + colored_wood_blocks*×12 ｜ C5: medium_clamp×12 ｜ C6: windex_bottle*×1 |
| dino_ce | 6 | 1 | 1 | C1: windex_bottle*×10 ｜ C2: windex_bottle*×1 ｜ C3: medium_clamp×12 ｜ C4: windex_bottle*×1 ｜ C5: peach×12 ｜ **C6**: sponge×12 + colored_wood_blocks*×12 |

### stack5_scene0004 — 5 物體 × 12 視角 = 60 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 6 | 1 | 0 | C1: sugar_box×8 ｜ C2: large_clamp×12 ｜ C3: apple×12 ｜ C4: strawberry×12 ｜ C5: colored_wood_blocks*×12 ｜ C6: sugar_box×4 |
| clip_raw | 6 | 1 | 0 | C1: sugar_box×8 ｜ C2: large_clamp×12 ｜ C3: sugar_box×4 ｜ C4: colored_wood_blocks*×12 ｜ C5: apple×12 ｜ C6: strawberry×12 |
| dino_cw | 7 | 3 | 1 | C1: apple×12 ｜ C2: strawberry×12 ｜ C3: large_clamp×11 ｜ C4: large_clamp×1 ｜ **C5**: sugar_box×1 + colored_wood_blocks*×1 ｜ C6: sugar_box×11 ｜ C7: colored_wood_blocks*×11 |
| dino_ce | 6 | 3 | 2 | C1: strawberry×12 ｜ C2: apple×12 ｜ C3: large_clamp×11 ｜ C4: large_clamp×1 ｜ **C5**: sugar_box×1 + colored_wood_blocks*×1 ｜ **C6**: sugar_box×11 + colored_wood_blocks*×11 |

### stack5_scene0005 — 4 物體 × 12 視角 = 60 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 4 | 1 | 1 | C1: pudding_box×11 ｜ C2: pudding_box×1 ｜ C3: rubiks_cube×12 ｜ **C4**: cups×24 + tuna_fish_can×12 |
| clip_raw | 5 | 1 | 0 | C1: pudding_box×11 ｜ C2: pudding_box×1 ｜ C3: rubiks_cube×12 ｜ C4: cups×24 ｜ C5: tuna_fish_can×12 |
| dino_cw | 4 | 0 | 0 | C1: rubiks_cube×12 ｜ C2: cups×24 ｜ C3: tuna_fish_can×12 ｜ C4: pudding_box×12 |
| dino_ce | 4 | 0 | 0 | C1: rubiks_cube×12 ｜ C2: cups×24 ｜ C3: tuna_fish_can×12 ｜ C4: pudding_box×12 |

### stack5_scene0006 — 5 物體 × 12 視角 = 60 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 8 | 2 | 0 | C1: sugar_box×8 ｜ C2: sugar_box×3 ｜ C3: sugar_box×1 ｜ C4: extra_large_clamp×12 ｜ C5: bowl×12 ｜ C6: colored_wood_blocks*×12 ｜ C7: tennis_ball×11 ｜ C8: tennis_ball×1 |
| clip_raw | 8 | 2 | 0 | C1: sugar_box×4 ｜ C2: sugar_box×4 ｜ C3: extra_large_clamp×12 ｜ C4: bowl×12 ｜ C5: tennis_ball×9 ｜ C6: tennis_ball×3 ｜ C7: colored_wood_blocks*×12 ｜ C8: sugar_box×4 |
| dino_cw | 7 | 3 | 1 | C1: extra_large_clamp×12 ｜ C2: bowl×12 ｜ C3: tennis_ball×9 ｜ C4: tennis_ball×3 ｜ **C5**: sugar_box×1 + colored_wood_blocks*×1 ｜ C6: sugar_box×11 ｜ C7: colored_wood_blocks*×11 |
| dino_ce | 6 | 3 | 2 | C1: extra_large_clamp×12 ｜ C2: bowl×12 ｜ **C3**: sugar_box×11 + colored_wood_blocks*×11 + tennis_ball×1 ｜ **C4**: sugar_box×1 + colored_wood_blocks*×1 ｜ C5: tennis_ball×9 ｜ C6: tennis_ball×2 |

### stack5_scene0007 — 5 物體 × 12 視角 = 58 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 7 | 2 | 0 | C1: apple×11 ｜ C2: colored_wood_blocks*×11 ｜ C3: cups×12 ｜ C4: tuna_fish_can×11 ｜ C5: sugar_box×10 ｜ C6: sugar_box×2 ｜ C7: tuna_fish_can×1 |
| clip_raw | 10 | 2 | 0 | C1: sugar_box×3 ｜ C2: sugar_box×4 ｜ C3: sugar_box×3 ｜ C4: sugar_box×2 ｜ C5: tuna_fish_can×1 ｜ C6: apple×11 ｜ C7: colored_wood_blocks*×11 ｜ C8: cups×12 ｜ C9: tuna_fish_can×10 ｜ C10: tuna_fish_can×1 |
| dino_cw | 6 | 1 | 0 | C1: sugar_box×11 ｜ C2: sugar_box×1 ｜ C3: apple×11 ｜ C4: colored_wood_blocks*×11 ｜ C5: tuna_fish_can×12 ｜ C6: cups×12 |
| dino_ce | 6 | 1 | 0 | C1: sugar_box×11 ｜ C2: sugar_box×1 ｜ C3: apple×11 ｜ C4: cups×12 ｜ C5: tuna_fish_can×12 ｜ C6: colored_wood_blocks*×11 |

### stack5_scene0008 — 5 物體 × 12 視角 = 60 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 6 | 3 | 2 | **C1**: tomato_soup_can×9 + potted_meat_can×9 + tuna_fish_can×5 ｜ C2: potted_meat_can×2 ｜ C3: potted_meat_can×1 ｜ **C4**: tuna_fish_can×7 + tomato_soup_can×3 ｜ C5: golf_ball×12 ｜ C6: cups×12 |
| clip_raw | 11 | 3 | 0 | C1: golf_ball×12 ｜ C2: cups×12 ｜ C3: tomato_soup_can×3 ｜ C4: tuna_fish_can×6 ｜ C5: tuna_fish_can×1 ｜ C6: potted_meat_can×1 ｜ C7: potted_meat_can×1 ｜ C8: tuna_fish_can×5 ｜ C9: tomato_soup_can×9 ｜ C10: potted_meat_can×9 ｜ C11: potted_meat_can×1 |
| dino_cw | 5 | 3 | 2 | C1: golf_ball×12 ｜ C2: cups×12 ｜ C3: potted_meat_can×2 ｜ **C4**: potted_meat_can×10 + tuna_fish_can×9 + tomato_soup_can×9 ｜ **C5**: tuna_fish_can×3 + tomato_soup_can×3 |
| dino_ce | 4 | 1 | 1 | C1: golf_ball×12 ｜ C2: cups×12 ｜ C3: potted_meat_can×2 ｜ **C4**: tuna_fish_can×12 + tomato_soup_can×12 + potted_meat_can×10 |

### stack5_scene0009 — 5 物體 × 12 視角 = 59 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 9 | 3 | 0 | C1: potted_meat_can×2 ｜ C2: wood_block×11 ｜ C3: wood_block×1 ｜ C4: colored_wood_blocks*×12 ｜ C5: potted_meat_can×4 ｜ C6: potted_meat_can×6 ｜ C7: knife×11 ｜ C8: windex_bottle*×6 ｜ C9: windex_bottle*×6 |
| clip_raw | 9 | 2 | 0 | C1: potted_meat_can×2 ｜ C2: knife×11 ｜ C3: wood_block×12 ｜ C4: windex_bottle*×6 ｜ C5: windex_bottle*×6 ｜ C6: colored_wood_blocks*×12 ｜ C7: potted_meat_can×4 ｜ C8: potted_meat_can×5 ｜ C9: potted_meat_can×1 |
| dino_cw | 9 | 4 | 1 | C1: wood_block×12 ｜ C2: knife×10 ｜ C3: knife×1 ｜ C4: windex_bottle*×9 ｜ C5: windex_bottle*×2 ｜ C6: windex_bottle*×1 ｜ C7: potted_meat_can×11 ｜ **C8**: potted_meat_can×1 + colored_wood_blocks*×1 ｜ C9: colored_wood_blocks*×11 |
| dino_ce | 10 | 5 | 2 | C1: wood_block×11 ｜ C2: wood_block×1 ｜ C3: knife×10 ｜ C4: knife×1 ｜ C5: windex_bottle*×9 ｜ C6: windex_bottle*×2 ｜ C7: windex_bottle*×1 ｜ **C8**: potted_meat_can×1 + colored_wood_blocks*×1 ｜ C9: colored_wood_blocks*×10 ｜ **C10**: potted_meat_can×11 + colored_wood_blocks*×1 |

### stack5_scene0010 — 4 物體 × 12 視角 = 58 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 12 | 3 | 0 | C1: fork×9 ｜ C2: fork×2 ｜ C3: potted_meat_can×1 ｜ C4: fork×1 ｜ C5: colored_wood_blocks*×1 ｜ C6: potted_meat_can×2 ｜ C7: colored_wood_blocks*×2 ｜ C8: colored_wood_blocks*×4 ｜ C9: colored_wood_blocks*×6 ｜ C10: colored_wood_blocks*×10 ｜ C11: pitcher_base×12 ｜ C12: potted_meat_can×8 |
| clip_raw | 11 | 3 | 2 | C1: potted_meat_can×2 ｜ C2: colored_wood_blocks*×2 ｜ C3: fork×9 ｜ **C4**: colored_wood_blocks*×1 + fork×1 ｜ **C5**: fork×2 + potted_meat_can×1 ｜ C6: colored_wood_blocks*×4 ｜ C7: colored_wood_blocks*×6 ｜ C8: pitcher_base×12 ｜ C9: colored_wood_blocks*×10 ｜ C10: potted_meat_can×3 ｜ C11: potted_meat_can×5 |
| dino_cw | 12 | 4 | 2 | C1: fork×9 ｜ C2: fork×2 ｜ C3: fork×1 ｜ C4: pitcher_base×3 ｜ C5: pitcher_base×9 ｜ **C6**: colored_wood_blocks*×5 + potted_meat_can×1 ｜ C7: colored_wood_blocks*×4 ｜ C8: colored_wood_blocks*×3 ｜ C9: potted_meat_can×9 ｜ C10: colored_wood_blocks*×9 ｜ **C11**: potted_meat_can×1 + colored_wood_blocks*×1 ｜ C12: colored_wood_blocks*×1 |
| dino_ce | 10 | 4 | 3 | C1: fork×9 ｜ C2: fork×2 ｜ C3: pitcher_base×3 ｜ C4: pitcher_base×9 ｜ **C5**: potted_meat_can×1 + colored_wood_blocks*×1 ｜ **C6**: potted_meat_can×9 + colored_wood_blocks*×9 ｜ **C7**: colored_wood_blocks*×5 + potted_meat_can×1 + fork×1 ｜ C8: colored_wood_blocks*×3 ｜ C9: colored_wood_blocks*×4 ｜ C10: colored_wood_blocks*×1 |

### stack5_scene0011 — 5 物體 × 12 視角 = 57 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 8 | 3 | 0 | C1: wood_block×12 ｜ C2: tennis_ball×10 ｜ C3: foam_brick×1 ｜ C4: tennis_ball×1 ｜ C5: foam_brick×10 ｜ C6: sponge×11 ｜ C7: padlock×11 ｜ C8: padlock×1 |
| clip_raw | 7 | 3 | 1 | C1: tennis_ball×10 ｜ C2: wood_block×12 ｜ C3: foam_brick×8 ｜ C4: foam_brick×2 ｜ C5: padlock×11 ｜ **C6**: padlock×1 + foam_brick×1 + tennis_ball×1 ｜ C7: sponge×11 |
| dino_cw | 6 | 3 | 2 | C1: tennis_ball×10 ｜ C2: wood_block×12 ｜ **C3**: sponge×1 + foam_brick×1 + tennis_ball×1 ｜ C4: padlock×12 ｜ C5: sponge×9 ｜ **C6**: foam_brick×10 + sponge×1 |
| dino_ce | 5 | 3 | 2 | C1: tennis_ball×10 ｜ C2: wood_block×12 ｜ **C3**: sponge×10 + foam_brick×10 ｜ **C4**: sponge×1 + foam_brick×1 + tennis_ball×1 ｜ C5: padlock×12 |

### stack5_scene0012 — 5 物體 × 12 視角 = 60 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 5 | 1 | 1 | C1: phillips_screwdriver×12 ｜ C2: gelatin_box×11 ｜ C3: gelatin_box×1 ｜ C4: colored_wood_blocks*×12 ｜ **C5**: orange×12 + plum×12 |
| clip_raw | 7 | 1 | 1 | C1: gelatin_box×7 ｜ C2: gelatin_box×3 ｜ C3: gelatin_box×1 ｜ C4: gelatin_box×1 ｜ C5: phillips_screwdriver×12 ｜ C6: colored_wood_blocks*×12 ｜ **C7**: orange×12 + plum×12 |
| dino_cw | 3 | 0 | 2 | C1: phillips_screwdriver×12 ｜ **C2**: orange×12 + plum×12 ｜ **C3**: gelatin_box×12 + colored_wood_blocks*×12 |
| dino_ce | 3 | 0 | 2 | C1: phillips_screwdriver×12 ｜ **C2**: orange×12 + plum×12 ｜ **C3**: gelatin_box×12 + colored_wood_blocks*×12 |

### stack5_scene0013 — 5 物體 × 12 視角 = 60 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 6 | 1 | 0 | C1: colored_wood_blocks*×3 ｜ C2: colored_wood_blocks*×9 ｜ C3: sugar_box×12 ｜ C4: sponge×12 ｜ C5: padlock×12 ｜ C6: foam_brick×12 |
| clip_raw | 11 | 3 | 0 | C1: sugar_box×3 ｜ C2: sugar_box×7 ｜ C3: sugar_box×1 ｜ C4: sugar_box×1 ｜ C5: foam_brick×11 ｜ C6: foam_brick×1 ｜ C7: padlock×12 ｜ C8: sponge×12 ｜ C9: colored_wood_blocks*×3 ｜ C10: colored_wood_blocks*×3 ｜ C11: colored_wood_blocks*×6 |
| dino_cw | 6 | 2 | 1 | C1: padlock×12 ｜ C2: foam_brick×11 ｜ **C3**: sponge×11 + foam_brick×1 ｜ C4: sponge×1 ｜ C5: sugar_box×12 ｜ C6: colored_wood_blocks*×12 |
| dino_ce | 5 | 2 | 2 | C1: padlock×12 ｜ **C2**: sponge×1 + foam_brick×1 ｜ **C3**: sponge×11 + foam_brick×11 ｜ C4: sugar_box×12 ｜ C5: colored_wood_blocks*×12 |

### stack5_scene0014 — 5 物體 × 12 視角 = 57 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 10 | 3 | 0 | C1: cracker_box×9 ｜ C2: cracker_box×3 ｜ C3: sugar_box×3 ｜ C4: sugar_box×8 ｜ C5: sugar_box×1 ｜ C6: baseball×7 ｜ C7: plum×12 ｜ C8: pear×12 ｜ C9: baseball×1 ｜ C10: baseball×1 |
| clip_raw | 14 | 3 | 0 | C1: baseball×7 ｜ C2: plum×12 ｜ C3: pear×12 ｜ C4: baseball×1 ｜ C5: baseball×1 ｜ C6: sugar_box×3 ｜ C7: sugar_box×7 ｜ C8: sugar_box×1 ｜ C9: cracker_box×3 ｜ C10: cracker_box×4 ｜ C11: cracker_box×1 ｜ C12: cracker_box×3 ｜ C13: cracker_box×1 ｜ C14: sugar_box×1 |
| dino_cw | 5 | 3 | 2 | **C1**: cracker_box×1 + sugar_box×1 ｜ **C2**: cracker_box×11 + sugar_box×11 + baseball×1 ｜ C3: baseball×8 ｜ C4: pear×12 ｜ C5: plum×12 |
| dino_ce | 5 | 3 | 2 | **C1**: cracker_box×1 + sugar_box×1 ｜ **C2**: cracker_box×11 + sugar_box×11 + baseball×1 ｜ C3: baseball×8 ｜ C4: plum×12 ｜ C5: pear×12 |

### stack5_scene0015 — 5 物體 × 12 視角 = 60 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 7 | 2 | 0 | C1: tennis_ball×12 ｜ C2: pear×12 ｜ C3: bleach_cleanser×1 ｜ C4: bleach_cleanser×11 ｜ C5: wood_block×12 ｜ C6: colored_wood_blocks*×9 ｜ C7: colored_wood_blocks*×3 |
| clip_raw | 11 | 3 | 0 | C1: bleach_cleanser×2 ｜ C2: bleach_cleanser×8 ｜ C3: bleach_cleanser×1 ｜ C4: tennis_ball×12 ｜ C5: pear×12 ｜ C6: bleach_cleanser×1 ｜ C7: wood_block×11 ｜ C8: wood_block×1 ｜ C9: colored_wood_blocks*×2 ｜ C10: colored_wood_blocks*×7 ｜ C11: colored_wood_blocks*×3 |
| dino_cw | 8 | 4 | 1 | **C1**: tennis_ball×12 + pear×4 ｜ C2: pear×8 ｜ C3: bleach_cleanser×11 ｜ C4: bleach_cleanser×1 ｜ C5: colored_wood_blocks*×11 ｜ C6: colored_wood_blocks*×1 ｜ C7: wood_block×11 ｜ C8: wood_block×1 |
| dino_ce | 7 | 3 | 1 | **C1**: tennis_ball×12 + pear×4 ｜ C2: pear×8 ｜ C3: bleach_cleanser×11 ｜ C4: bleach_cleanser×1 ｜ C5: colored_wood_blocks*×12 ｜ C6: wood_block×11 ｜ C7: wood_block×1 |

### stack5_scene0016 — 5 物體 × 12 視角 = 60 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 8 | 3 | 0 | C1: strawberry×12 ｜ C2: wood_block×11 ｜ C3: wood_block×1 ｜ C4: fork×12 ｜ C5: tomato_soup_can×11 ｜ C6: mustard_bottle×10 ｜ C7: mustard_bottle×2 ｜ C8: tomato_soup_can×1 |
| clip_raw | 9 | 2 | 0 | C1: strawberry×12 ｜ C2: tomato_soup_can×1 ｜ C3: tomato_soup_can×2 ｜ C4: tomato_soup_can×8 ｜ C5: tomato_soup_can×1 ｜ C6: mustard_bottle×2 ｜ C7: mustard_bottle×10 ｜ C8: fork×12 ｜ C9: wood_block×12 |
| dino_cw | 8 | 3 | 0 | C1: wood_block×11 ｜ C2: wood_block×1 ｜ C3: tomato_soup_can×11 ｜ C4: tomato_soup_can×1 ｜ C5: mustard_bottle×2 ｜ C6: mustard_bottle×10 ｜ C7: fork×12 ｜ C8: strawberry×12 |
| dino_ce | 7 | 2 | 0 | C1: wood_block×11 ｜ C2: wood_block×1 ｜ C3: tomato_soup_can×12 ｜ C4: mustard_bottle×2 ｜ C5: mustard_bottle×10 ｜ C6: fork×12 ｜ C7: strawberry×12 |

### stack5_scene0017 — 5 物體 × 12 視角 = 60 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 6 | 2 | 1 | C1: sugar_box×8 ｜ C2: sugar_box×4 ｜ C3: colored_wood_blocks*×12 ｜ C4: padlock×12 ｜ C5: master_chef_can×9 ｜ **C6**: cups×12 + master_chef_can×3 |
| clip_raw | 7 | 2 | 0 | C1: sugar_box×8 ｜ C2: master_chef_can×9 ｜ C3: master_chef_can×3 ｜ C4: cups×12 ｜ C5: sugar_box×4 ｜ C6: colored_wood_blocks*×12 ｜ C7: padlock×12 |
| dino_cw | 5 | 0 | 0 | C1: cups×12 ｜ C2: sugar_box×12 ｜ C3: master_chef_can×12 ｜ C4: padlock×12 ｜ C5: colored_wood_blocks*×12 |
| dino_ce | 5 | 0 | 0 | C1: sugar_box×12 ｜ C2: master_chef_can×12 ｜ C3: colored_wood_blocks*×12 ｜ C4: padlock×12 ｜ C5: cups×12 |

### stack5_scene0018 — 5 物體 × 12 視角 = 60 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 7 | 2 | 1 | C1: tomato_soup_can×10 ｜ C2: gelatin_box×12 ｜ C3: rubiks_cube×12 ｜ **C4**: tomato_soup_can×2 + cups×1 ｜ C5: cups×10 ｜ C6: golf_ball×12 ｜ C7: cups×1 |
| clip_raw | 7 | 2 | 0 | C1: gelatin_box×12 ｜ C2: tomato_soup_can×10 ｜ C3: tomato_soup_can×2 ｜ C4: rubiks_cube×12 ｜ C5: cups×2 ｜ C6: cups×10 ｜ C7: golf_ball×12 |
| dino_cw | 6 | 2 | 1 | C1: golf_ball×12 ｜ **C2**: tomato_soup_can×12 + cups×1 ｜ C3: gelatin_box×9 ｜ C4: gelatin_box×3 ｜ C5: rubiks_cube×12 ｜ C6: cups×11 |
| dino_ce | 5 | 3 | 2 | C1: golf_ball×12 ｜ **C2**: gelatin_box×3 + tomato_soup_can×3 ｜ **C3**: gelatin_box×9 + tomato_soup_can×9 + cups×2 ｜ C4: rubiks_cube×12 ｜ C5: cups×10 |

### stack5_scene0019 — 5 物體 × 12 視角 = 60 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 10 | 5 | 1 | C1: cracker_box×4 ｜ C2: cracker_box×8 ｜ C3: sugar_box×2 ｜ C4: sugar_box×8 ｜ C5: sugar_box×1 ｜ C6: orange×11 ｜ C7: bowl×9 ｜ **C8**: bowl×3 + sugar_box×1 + orange×1 ｜ C9: phillips_screwdriver×2 ｜ C10: phillips_screwdriver×10 |
| clip_raw | 12 | 4 | 1 | **C1**: bowl×3 + orange×2 + sugar_box×1 ｜ C2: phillips_screwdriver×12 ｜ C3: orange×10 ｜ C4: bowl×9 ｜ C5: sugar_box×2 ｜ C6: sugar_box×4 ｜ C7: sugar_box×4 ｜ C8: cracker_box×4 ｜ C9: cracker_box×4 ｜ C10: cracker_box×2 ｜ C11: cracker_box×2 ｜ C12: sugar_box×1 |
| dino_cw | 7 | 3 | 1 | C1: phillips_screwdriver×2 ｜ C2: phillips_screwdriver×10 ｜ **C3**: cracker_box×12 + sugar_box×12 ｜ C4: orange×2 ｜ C5: bowl×3 ｜ C6: orange×10 ｜ C7: bowl×9 |
| dino_ce | 7 | 3 | 1 | C1: phillips_screwdriver×2 ｜ C2: phillips_screwdriver×10 ｜ **C3**: cracker_box×12 + sugar_box×12 + bowl×2 ｜ C4: orange×2 ｜ C5: bowl×1 ｜ C6: orange×10 ｜ C7: bowl×9 |

### stack5_scene0020 — 5 物體 × 12 視角 = 60 筆

| 特徵 | 群數 | 過切 | 混群 | 各群組成 |
|---|---|---|---|---|
| clip_debias | 5 | 0 | 0 | C1: softball×12 ｜ C2: pudding_box×12 ｜ C3: pear×12 ｜ C4: cups×12 ｜ C5: tuna_fish_can×12 |
| clip_raw | 6 | 1 | 0 | C1: softball×12 ｜ C2: cups×12 ｜ C3: tuna_fish_can×6 ｜ C4: tuna_fish_can×6 ｜ C5: pear×12 ｜ C6: pudding_box×12 |
| dino_cw | 5 | 0 | 0 | C1: softball×12 ｜ C2: pear×12 ｜ C3: cups×12 ｜ C4: tuna_fish_can×12 ｜ C5: pudding_box×12 |
| dino_ce | 4 | 0 | 1 | C1: softball×12 ｜ C2: pear×12 ｜ C3: cups×12 ｜ **C4**: pudding_box×12 + tuna_fish_can×12 |

## 逐物體彙總:被過切 / 被混群的場次(60 場)

| 物體 | clip_debias 過切/混群 | clip_raw 過切/混群 | dino_cw 過切/混群 | dino_ce 過切/混群 | GEX |
|---|---|---|---|---|---|
| tomato_soup_can | 14 / 10 | 15 / 1 | 7 / 11 | 11 / 13 | 否 |
| master_chef_can | 12 / 6 | 12 / 4 | 9 / 6 | 8 / 7 | 否 |
| sugar_box | 11 / 1 | 12 / 2 | 9 / 9 | 9 / 9 | 否 |
| colored_wood_blocks | 9 / 2 | 8 / 3 | 22 / 17 | 21 / 22 | 是 |
| tuna_fish_can | 3 / 7 | 8 / 1 | 2 / 7 | 5 / 9 | 否 |
| cups | 5 / 5 | 5 / 3 | 6 / 5 | 6 / 6 | 否 |
| potted_meat_can | 8 / 1 | 8 / 1 | 6 / 6 | 8 / 8 | 否 |
| cracker_box | 7 / 0 | 8 / 0 | 1 / 3 | 1 / 3 | 否 |
| wood_block | 6 / 0 | 4 / 0 | 3 / 0 | 5 / 2 | 否 |
| gelatin_box | 5 / 1 | 6 / 1 | 1 / 1 | 3 / 6 | 否 |
| pudding_box | 4 / 1 | 5 / 1 | 3 / 1 | 3 / 3 | 否 |
| windex_bottle | 3 / 1 | 3 / 1 | 2 / 0 | 2 / 0 | 是 |
| orange | 2 / 2 | 2 / 2 | 2 / 2 | 2 / 2 | 否 |
| bleach_cleanser | 3 / 0 | 4 / 0 | 3 / 0 | 3 / 0 | 否 |
| padlock | 2 / 1 | 2 / 2 | 1 / 1 | 1 / 1 | 否 |
| tennis_ball | 3 / 0 | 3 / 1 | 2 / 2 | 2 / 3 | 否 |
| medium_clamp | 1 / 1 | 1 / 1 | 1 / 0 | 1 / 0 | 否 |
| mustard_bottle | 2 / 0 | 2 / 0 | 2 / 0 | 2 / 0 | 否 |
| bowl | 1 / 1 | 1 / 1 | 1 / 0 | 1 / 1 | 否 |
| lemon | 2 / 0 | 2 / 1 | 2 / 2 | 2 / 2 | 否 |
| spoon | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 否 |
| flat_screwdriver | 1 / 1 | 1 / 1 | 1 / 0 | 1 / 0 | 否 |
| fork | 1 / 0 | 1 / 1 | 1 / 0 | 1 / 1 | 否 |
| sponge | 1 / 0 | 1 / 0 | 4 / 7 | 4 / 8 | 否 |
| plum | 0 / 1 | 0 / 1 | 0 / 1 | 0 / 1 | 否 |
| phillips_screwdriver | 1 / 0 | 0 / 0 | 1 / 0 | 1 / 0 | 否 |
| foam_brick | 1 / 0 | 4 / 1 | 3 / 3 | 4 / 7 | 否 |
| peach | 1 / 0 | 1 / 0 | 1 / 0 | 1 / 0 | 否 |
| rubiks_cube | 1 / 0 | 1 / 1 | 1 / 1 | 1 / 1 | 否 |
| baseball | 1 / 0 | 1 / 0 | 1 / 1 | 1 / 1 | 否 |
| golf_ball | 1 / 0 | 1 / 0 | 1 / 1 | 1 / 1 | 否 |
| apple | 1 / 0 | 1 / 1 | 1 / 0 | 1 / 0 | 否 |
| racquetball | 0 / 1 | 0 / 1 | 0 / 0 | 0 / 0 | 否 |
| pitcher_base | 0 / 0 | 0 / 0 | 2 / 0 | 2 / 0 | 否 |
| knife | 0 / 0 | 0 / 0 | 2 / 0 | 2 / 0 | 否 |
| large_clamp | 0 / 0 | 0 / 0 | 1 / 0 | 1 / 0 | 否 |
| pear | 0 / 0 | 0 / 1 | 2 / 1 | 2 / 1 | 否 |
