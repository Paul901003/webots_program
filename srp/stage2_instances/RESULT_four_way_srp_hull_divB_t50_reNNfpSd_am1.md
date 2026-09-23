# 四分帳(歸對/洩漏/漏標/過切):srp_hull_divB_t50_reNNfpSd_am1

- 建檔 2026-09-23;程式 `four_way_breakdown.py`;gt=`srp_hull_gtlabel_am1`;排 GEX;303 多物;可復現(重跑本檔)。
- 每物體 voxel:correct+leak+unassigned=100%;fragment=同物被切散(過切)。合併看洩漏、過切看fragment、分離看歸對、漏看unassigned。

## 分組(每物體平均%)

| 組 | 物體數 | 歸對% | 洩漏%(合併) | 漏標% | 過切% |
|---|---|---|---|---|---|
| n | 646 | 98.1 | 0.2 | 1.7 | 6.4 |
| occ | 206 | 97.5 | 0.7 | 1.9 | 4.6 |
| stack | 58 | 88.7 | 10.3 | 1.0 | 1.6 |
| all | 910 | 97.3 | 1.0 | 1.7 | 5.7 |

## stack 逐 on 對(上/下物 各自 歸對/洩漏/漏標/過切 %)

| 場景 | 上→下 | 物 | 歸對 | 洩漏 | 漏標 | 過切 |
|---|---|---|---|---|---|---|
| stack3#0005 | foam_brick→gelatin_box | foam_brick | 100 | 0 | 0 | 0 |
| stack3#0005 | foam_brick→gelatin_box | gelatin_box | 95 | 3 | 2 | 0 |
| stack3#0006 | tomato_soup_can→tuna_fish_can | tomato_soup_can | 68 | 32 | 0 | 0 |
| stack3#0006 | tomato_soup_can→tuna_fish_can | tuna_fish_can | 97 | 1 | 2 | 0 |
| stack3#0011 | tomato_soup_can→tuna_fish_can | tomato_soup_can | 73 | 27 | 0 | 0 |
| stack3#0011 | tomato_soup_can→tuna_fish_can | tuna_fish_can | 97 | 1 | 2 | 0 |
| stack3#0013 | foam_brick→sponge | foam_brick | 99 | 0 | 0 | 0 |
| stack3#0013 | foam_brick→sponge | sponge | 96 | 3 | 1 | 0 |
| stack3#0014 | tomato_soup_can→master_chef_can | tomato_soup_can | 68 | 32 | 0 | 0 |
| stack3#0014 | tomato_soup_can→master_chef_can | master_chef_can | 99 | 0 | 1 | 0 |
| stack3#0015 | foam_brick→gelatin_box | foam_brick | 99 | 1 | 0 | 0 |
| stack3#0015 | foam_brick→gelatin_box | gelatin_box | 96 | 2 | 2 | 0 |
| stack3#0016 | tomato_soup_can→tuna_fish_can | tomato_soup_can | 72 | 28 | 0 | 0 |
| stack3#0016 | tomato_soup_can→tuna_fish_can | tuna_fish_can | 98 | 1 | 1 | 0 |
| stack3#0017 | tomato_soup_can→master_chef_can | tomato_soup_can | 98 | 2 | 0 | 0 |
| stack3#0017 | tomato_soup_can→master_chef_can | master_chef_can | 99 | 0 | 1 | 0 |
| stack3#0019 | foam_brick→wood_block | foam_brick | 99 | 1 | 0 | 42 |
| stack3#0019 | foam_brick→wood_block | wood_block | 99 | 0 | 0 | 0 |
| stack4#0002 | tomato_soup_can→tuna_fish_can | tomato_soup_can | 72 | 28 | 0 | 0 |
| stack4#0002 | tomato_soup_can→tuna_fish_can | tuna_fish_can | 99 | 1 | 0 | 0 |
| stack4#0003 | master_chef_can→pudding_box | master_chef_can | 99 | 1 | 0 | 0 |
| stack4#0003 | master_chef_can→pudding_box | pudding_box | 89 | 10 | 1 | 0 |
| stack4#0006 | tomato_soup_can→gelatin_box | tomato_soup_can | 92 | 7 | 1 | 0 |
| stack4#0006 | tomato_soup_can→gelatin_box | gelatin_box | 94 | 2 | 4 | 0 |
| stack4#0007 | sugar_box→sponge | sugar_box | 100 | 0 | 0 | 0 |
| stack4#0007 | sugar_box→sponge | sponge | 0 | 92 | 8 | 0 |
| stack4#0009 | master_chef_can→pudding_box | master_chef_can | 98 | 2 | 0 | 0 |
| stack4#0009 | master_chef_can→pudding_box | pudding_box | 83 | 15 | 1 | 0 |
| stack4#0010 | tuna_fish_can→master_chef_can | tuna_fish_can | 0 | 100 | 0 | 0 |
| stack4#0010 | tuna_fish_can→master_chef_can | master_chef_can | 99 | 0 | 1 | 0 |
| stack4#0011 | tomato_soup_can→master_chef_can | tomato_soup_can | 69 | 30 | 1 | 0 |
| stack4#0011 | tomato_soup_can→master_chef_can | master_chef_can | 100 | 0 | 0 | 0 |
| stack4#0012 | tomato_soup_can→gelatin_box | tomato_soup_can | 97 | 3 | 0 | 0 |
| stack4#0012 | tomato_soup_can→gelatin_box | gelatin_box | 97 | 2 | 1 | 0 |
| stack4#0014 | sugar_box→sponge | sugar_box | 75 | 25 | 0 | 0 |
| stack4#0014 | sugar_box→sponge | sponge | 93 | 4 | 3 | 0 |
| stack4#0017 | foam_brick→master_chef_can | foam_brick | 97 | 3 | 0 | 0 |
| stack4#0017 | foam_brick→master_chef_can | master_chef_can | 99 | 0 | 0 | 0 |
| stack4#0018 | tomato_soup_can→tuna_fish_can | tomato_soup_can | 99 | 1 | 0 | 0 |
| stack4#0018 | tomato_soup_can→tuna_fish_can | tuna_fish_can | 96 | 2 | 2 | 0 |
| stack5#0005 | tuna_fish_can→pudding_box | tuna_fish_can | 68 | 31 | 1 | 0 |
| stack5#0005 | tuna_fish_can→pudding_box | pudding_box | 99 | 0 | 1 | 0 |
| stack5#0008 | tomato_soup_can→tuna_fish_can | tomato_soup_can | 99 | 1 | 0 | 0 |
| stack5#0008 | tomato_soup_can→tuna_fish_can | tuna_fish_can | 92 | 6 | 2 | 0 |
| stack5#0011 | foam_brick→sponge | foam_brick | 99 | 1 | 0 | 0 |
| stack5#0011 | foam_brick→sponge | sponge | 97 | 3 | 1 | 0 |
| stack5#0013 | foam_brick→sponge | foam_brick | 99 | 1 | 0 | 0 |
| stack5#0013 | foam_brick→sponge | sponge | 95 | 2 | 3 | 0 |
| stack5#0014 | sugar_box→cracker_box | sugar_box | 81 | 14 | 4 | 0 |
| stack5#0014 | sugar_box→cracker_box | cracker_box | 100 | 0 | 0 | 5 |
| stack5#0016 | tomato_soup_can→wood_block | tomato_soup_can | 97 | 3 | 0 | 0 |
| stack5#0016 | tomato_soup_can→wood_block | wood_block | 100 | 0 | 0 | 0 |
| stack5#0018 | tomato_soup_can→gelatin_box | tomato_soup_can | 98 | 2 | 0 | 0 |
| stack5#0018 | tomato_soup_can→gelatin_box | gelatin_box | 96 | 4 | 1 | 0 |
| stack5#0019 | sugar_box→cracker_box | sugar_box | 60 | 35 | 5 | 26 |
| stack5#0019 | sugar_box→cracker_box | cracker_box | 98 | 0 | 2 | 0 |
| stack5#0020 | tuna_fish_can→pudding_box | tuna_fish_can | 99 | 1 | 0 | 0 |
| stack5#0020 | tuna_fish_can→pudding_box | pudding_box | 69 | 30 | 0 | 20 |