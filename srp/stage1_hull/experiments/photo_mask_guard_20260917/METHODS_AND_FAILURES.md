# Methods, Commands, and Failure Map

## Scope and Terms

This is the fixed legacy stack cohort: 60 scenes and 29 usable on relations.

- am1: center-point visual hull with one allowed missing view.
- am1_fp: footprint visual hull; more coverage and more ghost volume.
- fgpatch: NCC samples must remain in the foreground union.
- second: two source views must support one normal candidate to retain a voxel.
- reNNcSd: donut-mask CLIP clustering, center voting, connected-or-drop.
- divB_t50: raw-mask span merge with normal-diversity guard at theta 0.5.

## Aggregate Results

| Method | Output root | Correct | Leak | Unassigned | Fragment | Main merged |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Mainline before div | srp_hull_semcluster_reNNcSd_am1 | 78.47% | 4.29% | 17.24% | 15.23% | 0.0% |
| Mainline divB | srp_hull_divB_t50_reNNcSd_am1 | 74.04% | 8.71% | 17.24% | 1.78% | 13.8% |
| Center photo second before div | srp_hull_semcluster_reNNcSd_am1photo_fgpatch_second | 76.43% | 4.98% | 18.59% | 15.76% | 0.0% |
| Center photo second divB | srp_hull_divB_t50_reNNcSd_am1photo_fgpatch_second | 74.03% | 7.38% | 18.59% | 2.19% | 13.8% |
| Footprint no photo before div | srp_hull_semcluster_reNNcSd_am1fp | 79.54% | 4.82% | 15.64% | 16.54% | 0.0% |
| Footprint photo second before div | srp_hull_semcluster_reNNcSd_am1fpphoto_fgpatch_second | 77.47% | 4.91% | 17.62% | 16.34% | 3.4% |
| Footprint photo second divB | srp_hull_divB_t50_reNNcSd_am1fpphoto_fgpatch_second | 72.50% | 9.88% | 17.62% | 1.25% | 20.7% |

Stage 1 geometry on the same 60 scenes:

| Hull | Mesh missed | Ghost | IoU |
| --- | ---: | ---: | ---: |
| am1 | 8.7% | 20.8% | 0.737 |
| am1 photo second | 9.2% | 20.3% | 0.738 |
| am1_fp | 1.4% | 26.6% | 0.726 |
| am1_fp photo second | 1.8% | 25.9% | 0.731 |

## Fixed-Scene Photo Check

Stress scene: stack4_scene0007. Sugar-box visible surface loss:

| Variant from am1 | Sugar surface removed | Coverage | Ghost |
| --- | ---: | ---: | ---: |
| No photo | 0 / 1202 | 0.8979 | 0.1420 |
| Original mean | 275 / 1202 (22.9%) | 0.8745 | 0.1181 |
| fgpatch mean | 275 / 1202 (22.9%) | 0.8775 | 0.1175 |
| fgpatch max | 0 / 1202 | 0.8979 | 0.1420 |
| fgpatch second | 120 / 1202 (10.0%) | 0.8901 | 0.1378 |

For am1_fp, guarded mean removes 374 / 1304 sugar surface voxels (28.7%);
guarded second removes 176 / 1304 (13.5%).

## Failure Map

| Method | Scene or condition | Evidence | Reason supported by current data |
| --- | --- | --- | --- |
| Original mean and fgpatch mean | stack4_scene0007 sugar_box | Both remove 22.9% of visible sugar surface. | A stacked different object is still foreground, so one poor source lowers mean NCC. Foreground checking alone cannot reject it. |
| fgpatch max | stack4_scene0007 | Deletes zero voxels. | One accidental high source or normal match retains every voxel. |
| Center photo second before div | Cohort | Correct 78.47% to 76.43%; unassigned 17.24% to 18.59%. | Two-source support limits catastrophic deletion but still removes useful labelled surface. |
| divB | Contact regions | Fragment falls while leak and merged pairs rise. | Low-div fragments can bridge complete objects through raw-mask span and contact ghost geometry. |
| Footprint photo second plus divB | Cohort | Leak 9.88%; 6 / 29 main pairs merged. | Wider hull increases contact ghost volume. Reject. |
| Center photo divB | stack4_scene0014 sugar_box -> sponge | Separated becomes merged; 0% of 270 traced leak voxels become correct. | Regression. Need visual inspection to separate photo geometry from div contact merge. |
| Center photo divB | stack4_scene0007, stack4_scene0010, stack5_scene0020 | 83.3% to 97.5% of traced leak voxels remain leak; pairs stay merged. | Persistent contact and occlusion failure. |
| Center photo divB | stack3_scene0005 foam_brick -> gelatin_box | 76.3% of 304 traced leak voxels become correct; merged becomes separated. | Strong genuine improvement. |
| Center photo divB | stack5_scene0019 sugar_box -> cracker_box | 25.5% correct, 5.7% unassigned, 22.7% surface changed. | Mixed case. Inspect before accepting. |

## Direct Leak Flow

See [DIVB_PAIR_FLOW.md](DIVB_PAIR_FLOW.md) for all 29 pairs. It tracks the
2,405 voxels that leak under mainline divB at shared world coordinates.

- 13.9% become correct in center-photo divB.
- 0.3% become unassigned.
- 75.1% remain leak.
- 10.6% are no longer the same GT-labelled surface after photo carving.
- Main-pair merging stays 4 / 29, with one new merge and one new separation.

The center-photo-div leak reduction is mainly genuine recovery, not a label-0
artifact. Keep it as an investigation candidate, not a promoted mainline.

## Commands

Run the frozen 60-scene photo cohort for center and footprint:

~~~bash
./srp/stage1_hull/experiments/photo_mask_guard_20260917/run_stack_cohort.sh
~~~

Run one center scene:

~~~bash
/home/cho/.pyenv/versions/webots_visual_hull/bin/python3 \
  srp/stage1_hull/experiments/photo_mask_guard_20260917/run_fgpatch.py \
  --in-root srp_hull_mv2_v12_am1 stack4_scene0007
~~~

Run divB t50 for the fixed cohort:
Run Stage 2 center voting and donut semantic clustering for one scene:

~~~bash
env HULL_ROOT="$PWD/data/eval/srp_hull_mv2_v12_am1_photo_fgpatch_second" \
  SAM_ROOT="$PWD/data/eval/mobilesamv2_fast" \
  CAPTURES_ROOT="$PWD/data/captures_fast" \
  ARM_MASK_ROOT="$PWD/data/eval/srp_arm_masks" \
  OUT_ROOT=srp_hull_semcluster_reNNcSd_am1photo_fgpatch_second VOTE=center \
  /home/cho/.pyenv/versions/webots_visual_hull/bin/python3 \
  srp/stage2_instances/voxel_sem_cluster_reassign_soliddrop.py stack4_scene0007
~~~

Run Stage 2 center voting and donut semantic clustering for one scene:

~~~bash
env HULL_ROOT="$PWD/data/eval/srp_hull_mv2_v12_am1_photo_fgpatch_second" \
  SAM_ROOT="$PWD/data/eval/mobilesamv2_fast" \
  CAPTURES_ROOT="$PWD/data/captures_fast" \
  ARM_MASK_ROOT="$PWD/data/eval/srp_arm_masks" \
  OUT_ROOT=srp_hull_semcluster_reNNcSd_am1photo_fgpatch_second VOTE=center \
  /home/cho/.pyenv/versions/webots_visual_hull/bin/python3 \
  srp/stage2_instances/voxel_sem_cluster_reassign_soliddrop.py stack4_scene0007
~~~


~~~bash
SCENES=$(find data/eval/srp_hull_mv2_v12_am1_photo_fgpatch_second \
  -mindepth 2 -maxdepth 2 -name hull.npz -printf '%h\n' \
  | sed 's#.*/##' | sort)

# Intentional unquoted expansion: every scene name is one shell word.
env BASE_ROOT=srp_hull_semcluster_reNNcSd_am1photo_fgpatch_second \
  HULL_ROOT_NAME=srp_hull_mv2_v12_am1_photo_fgpatch_second \
  OUT_SUFFIX=_reNNcSd_am1photo_fgpatch_second THETAS=0.5 \
  /home/cho/.pyenv/versions/webots_visual_hull/bin/python3 \
  srp/stage2_instances/merge_div_guard.py $SCENES
~~~

Evaluate clean leak and generate the direct flow report:

~~~bash
/home/cho/.pyenv/versions/webots_visual_hull/bin/python3 \
  srp/stage2_instances/stack_leak_nosep.py \
  --inst-root srp_hull_divB_t50_reNNcSd_am1photo_fgpatch_second \
  --gt-root srp_hull_gtlabel_am1photo_fgpatch_second

/home/cho/.pyenv/versions/webots_visual_hull/bin/python3 \
  srp/stage1_hull/experiments/photo_mask_guard_20260917/compare_divb_pair_flow.py
~~~

View Stage 1 in gray and final divB instances in color:

~~~bash
SRP_VIZ_ARGS="stack4_scene0007 1 srp_hull_mv2_v12_am1_photo_fgpatch_second cubes grayfill" webots worlds/hull_viz.wbt
SRP_VIZ_ARGS="stack4_scene0007 1 srp_hull_divB_t50_reNNcSd_am1photo_fgpatch_second cubes" webots worlds/hull_viz.wbt
~~~
