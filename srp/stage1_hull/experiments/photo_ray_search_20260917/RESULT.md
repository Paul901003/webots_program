# RGB-Only Ray-Search Photo Experiment

## Question

Can warped-NCC choose a stronger occupied layer behind a weak visual-hull
surface, rather than only deleting the current outer layer?

## Fixed Inputs

- Cohort: 60 legacy `stack3/4/5` scenes.
- Input hull: `srp_hull_mv2_v12_am1` with 12 views and `allow_miss=1`.
- Semantic: existing donut masks, center voting, solid z-buffer, and
  connected-or-drop reassignment.
- Merge: existing divB with `theta=0.5`.

## Method

For every initial surface voxel, sample occupied voxel centers at depths 0--3
along the most front-facing camera ray. Each candidate uses foreground-patch,
plane-warped NCC with two source views. The outer ray cells are removed only
when a deeper candidate has NCC at least `0.25`, exceeds the outer NCC by
`0.10`, and the outer NCC is below `0.10`.

This is a conservative, one-way test. It moves an implicit surface inward only
by deleting outer cells. It cannot add a voxel or recover a surface with no
RGB evidence.

## Results

### Geometry Against Mesh Occupancy

| Method | Coverage | Ghost | Bloat | Mean voxels |
| --- | ---: | ---: | ---: | ---: |
| Raw am1 | 89.55% | 19.44% | 1.140 | 19,125.4 |
| fgpatch second | 89.01% | 18.90% | 1.126 | 18,908.1 |
| Ray search | 88.85% | 18.68% | 1.121 | 18,832.9 |

Ray search removes a further `0.22pp` ghost relative to `fgpatch_second`, at
the cost of `0.16pp` coverage.

### Stack Separation With Aligned GT Surface Labels

| Method | Correct | Leak | Unassigned | Fragment | Main pair merged |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ray search, no div | 75.42% | 3.90% | 20.68% | 15.15% | 0.0% |
| Ray search + divB | 72.30% | 7.02% | 20.68% | 1.64% | 13.8% |
| Existing center photo + divB | 74.03% | 7.38% | 18.59% | 2.19% | 13.8% |

The div result has the lowest aggregate leak tested so far, but loses `1.73pp`
correct assignment and creates `2.09pp` more unassigned surface than the
existing center-photo-div candidate. It is not ready to replace that method.

`DIVB_PAIR_FLOW.md` traces mainline div leak voxels into ray-search div:
15.7% become correct and only 1.0% become unassigned. The gain is therefore
not primarily a label-zero artifact. Important remaining failures are
`stack4_scene0007` (sugar box -> sponge remains merged) and
`stack4_scene0014` (new merge remains).

## Reproduce

```bash
bash srp/stage1_hull/experiments/photo_ray_search_20260917/run_stack_cohort.sh

python srp/stage1_hull/experiments/photo_ray_search_20260917/eval_stack_geometry.py

env HULL_ROOT="$PWD/data/eval/srp_hull_mv2_v12_am1_photo_raysearch" \
  SAM_ROOT="$PWD/data/eval/mobilesamv2_fast" \
  CAPTURES_ROOT="$PWD/data/captures_fast" \
  ARM_MASK_ROOT="$PWD/data/eval/srp_arm_masks" \
  OUT_ROOT=srp_hull_semcluster_reNNcSd_am1photo_raysearch VOTE=center \
  python srp/stage2_instances/voxel_sem_cluster_reassign_soliddrop.py stack3 stack4 stack5

python srp/stage2_instances/build_gtlabel.py \
  --hull srp_hull_mv2_v12_am1_photo_raysearch \
  --out srp_hull_gtlabel_am1photo_raysearch stack3 stack4 stack5

env BASE_ROOT=srp_hull_semcluster_reNNcSd_am1photo_raysearch \
  HULL_ROOT_NAME=srp_hull_mv2_v12_am1_photo_raysearch \
  OUT_SUFFIX=_reNNcSd_am1photo_raysearch THETAS=.5 \
  python srp/stage2_instances/merge_div_guard.py stack3 stack4 stack5
```
