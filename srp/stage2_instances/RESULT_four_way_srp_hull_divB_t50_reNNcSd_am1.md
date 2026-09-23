# 四分帳(歸對/洩漏/漏標/過切):srp_hull_divB_t50_reNNcSd_am1

- 建檔 2026-09-23;程式 `four_way_breakdown.py`;gt=`srp_hull_gtlabel_am1`;排 GEX;303 多物;可復現(重跑本檔)。
- 每物體 voxel:correct+leak+unassigned=100%;fragment=同物被切散(過切)。合併看洩漏、過切看fragment、分離看歸對、漏看unassigned。

## 分組(每物體平均%)

| 組 | 物體數 | 歸對% | 洩漏%(合併) | 漏標% | 過切% |
|---|---|---|---|---|---|
| n | 646 | 80.8 | 0.1 | 19.1 | 4.6 |
| occ | 206 | 79.7 | 0.2 | 20.1 | 3.2 |
| stack | 58 | 74.0 | 8.7 | 17.2 | 1.8 |
| all | 910 | 80.1 | 0.6 | 19.2 | 4.1 |

## stack 逐 on 對(上/下物 各自 歸對/洩漏/漏標/過切 %)

| 場景 | 上→下 | 物 | 歸對 | 洩漏 | 漏標 | 過切 |
|---|---|---|---|---|---|---|
| stack3#0005 | foam_brick→gelatin_box | foam_brick | 96 | 0 | 4 | 0 |
| stack3#0005 | foam_brick→gelatin_box | gelatin_box | 0 | 76 | 24 | 0 |
| stack3#0006 | tomato_soup_can→tuna_fish_can | tomato_soup_can | 89 | 0 | 11 | 0 |
| stack3#0006 | tomato_soup_can→tuna_fish_can | tuna_fish_can | 74 | 3 | 23 | 0 |
| stack3#0011 | tomato_soup_can→tuna_fish_can | tomato_soup_can | 90 | 0 | 9 | 0 |
| stack3#0011 | tomato_soup_can→tuna_fish_can | tuna_fish_can | 70 | 4 | 26 | 0 |
| stack3#0013 | foam_brick→sponge | foam_brick | 97 | 0 | 2 | 0 |
| stack3#0013 | foam_brick→sponge | sponge | 70 | 8 | 22 | 0 |
| stack3#0014 | tomato_soup_can→master_chef_can | tomato_soup_can | 85 | 2 | 14 | 0 |
| stack3#0014 | tomato_soup_can→master_chef_can | master_chef_can | 78 | 1 | 21 | 0 |
| stack3#0015 | foam_brick→gelatin_box | foam_brick | 95 | 0 | 4 | 0 |
| stack3#0015 | foam_brick→gelatin_box | gelatin_box | 72 | 8 | 20 | 0 |
| stack3#0016 | tomato_soup_can→tuna_fish_can | tomato_soup_can | 88 | 0 | 11 | 0 |
| stack3#0016 | tomato_soup_can→tuna_fish_can | tuna_fish_can | 68 | 3 | 29 | 0 |
| stack3#0017 | tomato_soup_can→master_chef_can | tomato_soup_can | 86 | 1 | 13 | 0 |
| stack3#0017 | tomato_soup_can→master_chef_can | master_chef_can | 75 | 1 | 24 | 0 |
| stack3#0019 | foam_brick→wood_block | foam_brick | 97 | 1 | 3 | 41 |
| stack3#0019 | foam_brick→wood_block | wood_block | 87 | 1 | 11 | 0 |
| stack4#0002 | tomato_soup_can→tuna_fish_can | tomato_soup_can | 92 | 0 | 8 | 0 |
| stack4#0002 | tomato_soup_can→tuna_fish_can | tuna_fish_can | 71 | 3 | 26 | 0 |
| stack4#0003 | master_chef_can→pudding_box | master_chef_can | 89 | 0 | 11 | 0 |
| stack4#0003 | master_chef_can→pudding_box | pudding_box | 64 | 2 | 34 | 0 |
| stack4#0006 | tomato_soup_can→gelatin_box | tomato_soup_can | 84 | 0 | 16 | 0 |
| stack4#0006 | tomato_soup_can→gelatin_box | gelatin_box | 47 | 27 | 26 | 0 |
| stack4#0007 | sugar_box→sponge | sugar_box | 82 | 0 | 18 | 0 |
| stack4#0007 | sugar_box→sponge | sponge | 0 | 57 | 43 | 0 |
| stack4#0009 | master_chef_can→pudding_box | master_chef_can | 87 | 1 | 12 | 0 |
| stack4#0009 | master_chef_can→pudding_box | pudding_box | 58 | 3 | 39 | 0 |
| stack4#0010 | tuna_fish_can→master_chef_can | tuna_fish_can | 0 | 96 | 4 | 0 |
| stack4#0010 | tuna_fish_can→master_chef_can | master_chef_can | 80 | 0 | 20 | 0 |
| stack4#0011 | tomato_soup_can→master_chef_can | tomato_soup_can | 89 | 2 | 9 | 0 |
| stack4#0011 | tomato_soup_can→master_chef_can | master_chef_can | 79 | 0 | 21 | 0 |
| stack4#0012 | tomato_soup_can→gelatin_box | tomato_soup_can | 89 | 0 | 11 | 0 |
| stack4#0012 | tomato_soup_can→gelatin_box | gelatin_box | 75 | 5 | 20 | 0 |
| stack4#0014 | sugar_box→sponge | sugar_box | 91 | 1 | 8 | 44 |
| stack4#0014 | sugar_box→sponge | sponge | 0 | 73 | 27 | 0 |
| stack4#0017 | foam_brick→master_chef_can | foam_brick | 96 | 1 | 3 | 0 |
| stack4#0017 | foam_brick→master_chef_can | master_chef_can | 76 | 4 | 20 | 0 |
| stack4#0018 | tomato_soup_can→tuna_fish_can | tomato_soup_can | 92 | 0 | 7 | 0 |
| stack4#0018 | tomato_soup_can→tuna_fish_can | tuna_fish_can | 76 | 3 | 21 | 0 |
| stack5#0005 | tuna_fish_can→pudding_box | tuna_fish_can | 68 | 26 | 6 | 0 |
| stack5#0005 | tuna_fish_can→pudding_box | pudding_box | 81 | 0 | 19 | 0 |
| stack5#0008 | tomato_soup_can→tuna_fish_can | tomato_soup_can | 92 | 0 | 8 | 0 |
| stack5#0008 | tomato_soup_can→tuna_fish_can | tuna_fish_can | 71 | 3 | 26 | 0 |
| stack5#0011 | foam_brick→sponge | foam_brick | 92 | 1 | 8 | 0 |
| stack5#0011 | foam_brick→sponge | sponge | 72 | 6 | 21 | 0 |
| stack5#0013 | foam_brick→sponge | foam_brick | 91 | 1 | 8 | 0 |
| stack5#0013 | foam_brick→sponge | sponge | 60 | 7 | 33 | 0 |
| stack5#0014 | sugar_box→cracker_box | sugar_box | 82 | 5 | 13 | 0 |
| stack5#0014 | sugar_box→cracker_box | cracker_box | 89 | 1 | 10 | 0 |
| stack5#0016 | tomato_soup_can→wood_block | tomato_soup_can | 83 | 2 | 15 | 0 |
| stack5#0016 | tomato_soup_can→wood_block | wood_block | 88 | 0 | 12 | 0 |
| stack5#0018 | tomato_soup_can→gelatin_box | tomato_soup_can | 89 | 1 | 10 | 0 |
| stack5#0018 | tomato_soup_can→gelatin_box | gelatin_box | 65 | 1 | 35 | 0 |
| stack5#0019 | sugar_box→cracker_box | sugar_box | 18 | 25 | 57 | 0 |
| stack5#0019 | sugar_box→cracker_box | cracker_box | 78 | 0 | 22 | 0 |
| stack5#0020 | tuna_fish_can→pudding_box | tuna_fish_can | 96 | 1 | 3 | 0 |
| stack5#0020 | tuna_fish_can→pudding_box | pudding_box | 42 | 37 | 22 | 19 |