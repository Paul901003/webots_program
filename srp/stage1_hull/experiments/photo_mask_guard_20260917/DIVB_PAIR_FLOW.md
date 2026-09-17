# Mainline Div vs Center-Photo Div: Leak Voxel Flow

This report follows only voxels that are leakage in mainline div: their GT object
is assigned to an instance dominated by a different GT object. The two grids share
world coordinates. For each such voxel, the photo result is classified as correct,
still leak, unassigned, other, or surface changed. Surface changed means the same
world coordinate is no longer the same object's GT-labelled surface after photo carving;
it is deliberately not counted as unassigned.

- On relations: 29
- Mainline leak voxels traced: 2405
- To correct: 13.9%
- Still leak: 75.1%
- To unassigned: 0.3%
- Surface changed: 10.6%
- Other assigned state: 0.0%
- Main-pair merged: mainline 4/29, photo 4/29
- Newly merged: 1; newly separated: 1

| Scene | On pair | Main leak vox | To correct | Still leak | To unassigned | Surface changed | Other | Main merged -> photo merged |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| stack3_scene0005 | foam_brick -> gelatin_box | 304 | 76.3% | 15.8% | 0.0% | 7.9% | 0.0% | yes -> no |
| stack5_scene0019 | sugar_box -> cracker_box | 141 | 25.5% | 46.1% | 5.7% | 22.7% | 0.0% | no -> no |
| stack4_scene0006 | tomato_soup_can -> gelatin_box | 126 | 24.6% | 66.7% | 0.0% | 8.7% | 0.0% | no -> no |
| stack5_scene0014 | sugar_box -> cracker_box | 63 | 38.1% | 57.1% | 0.0% | 4.8% | 0.0% | no -> no |
| stack5_scene0011 | foam_brick -> sponge | 24 | 12.5% | 62.5% | 0.0% | 25.0% | 0.0% | no -> no |
| stack3_scene0015 | foam_brick -> gelatin_box | 31 | 6.5% | 74.2% | 0.0% | 19.4% | 0.0% | no -> no |
| stack5_scene0013 | foam_brick -> sponge | 26 | 7.7% | 73.1% | 0.0% | 19.2% | 0.0% | no -> no |
| stack3_scene0011 | tomato_soup_can -> tuna_fish_can | 15 | 6.7% | 73.3% | 0.0% | 20.0% | 0.0% | no -> no |
| stack3_scene0014 | tomato_soup_can -> master_chef_can | 23 | 4.3% | 91.3% | 0.0% | 4.3% | 0.0% | no -> no |
| stack3_scene0016 | tomato_soup_can -> tuna_fish_can | 13 | 7.7% | 92.3% | 0.0% | 0.0% | 0.0% | no -> no |
| stack4_scene0009 | master_chef_can -> pudding_box | 28 | 3.6% | 89.3% | 0.0% | 7.1% | 0.0% | no -> no |
| stack5_scene0005 | tuna_fish_can -> pudding_box | 125 | 0.8% | 96.0% | 0.0% | 3.2% | 0.0% | no -> no |
| stack3_scene0006 | tomato_soup_can -> tuna_fish_can | 15 | 0.0% | 93.3% | 0.0% | 6.7% | 0.0% | no -> no |
| stack3_scene0013 | foam_brick -> sponge | 29 | 0.0% | 93.1% | 0.0% | 6.9% | 0.0% | no -> no |
| stack3_scene0017 | tomato_soup_can -> master_chef_can | 23 | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | no -> no |
| stack3_scene0019 | foam_brick -> wood_block | 33 | 0.0% | 69.7% | 0.0% | 30.3% | 0.0% | no -> no |
| stack4_scene0002 | tomato_soup_can -> tuna_fish_can | 15 | 0.0% | 86.7% | 0.0% | 13.3% | 0.0% | no -> no |
| stack4_scene0003 | master_chef_can -> pudding_box | 14 | 0.0% | 92.9% | 0.0% | 7.1% | 0.0% | no -> no |
| stack4_scene0007 | sugar_box -> sponge | 289 | 0.0% | 94.5% | 0.0% | 5.5% | 0.0% | yes -> yes |
| stack4_scene0010 | tuna_fish_can -> master_chef_can | 414 | 0.0% | 83.3% | 0.0% | 16.7% | 0.0% | yes -> yes |
| stack4_scene0011 | tomato_soup_can -> master_chef_can | 15 | 0.0% | 93.3% | 0.0% | 6.7% | 0.0% | no -> no |
| stack4_scene0012 | tomato_soup_can -> gelatin_box | 20 | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | no -> no |
| stack4_scene0014 | sugar_box -> sponge | 270 | 0.0% | 87.4% | 0.0% | 12.6% | 0.0% | no -> yes |
| stack4_scene0017 | foam_brick -> master_chef_can | 66 | 0.0% | 80.3% | 0.0% | 19.7% | 0.0% | no -> no |
| stack4_scene0018 | tomato_soup_can -> tuna_fish_can | 11 | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | no -> no |
| stack5_scene0008 | tomato_soup_can -> tuna_fish_can | 14 | 0.0% | 92.9% | 0.0% | 7.1% | 0.0% | no -> no |
| stack5_scene0016 | tomato_soup_can -> wood_block | 13 | 0.0% | 84.6% | 0.0% | 15.4% | 0.0% | no -> no |
| stack5_scene0018 | tomato_soup_can -> gelatin_box | 9 | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% | no -> no |
| stack5_scene0020 | tuna_fish_can -> pudding_box | 236 | 0.0% | 97.5% | 0.0% | 2.5% | 0.0% | yes -> yes |
