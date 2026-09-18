# Anchor Free-Space Vote: Offline Diagnostic

## Question

Can a high-confidence photometric surface anchor safely remove occupied hull
voxels in front of it?  An anchor is accepted only when 7x7 and 11x11 NCC
select the exact same candidate voxel on its own reference ray.  This test
never writes a carved hull.

## Method

1. Score the fixed `hull_multi_ref_candidates_mv2_v12_am1` candidates with the
   CUDA NCC scorer at 7x7 and 11x11.
2. Keep a row as an anchor only when both patch sizes select the same voxel.
3. Traverse every reference-camera-to-anchor segment exactly through the voxel
   grid. The anchor cell itself is excluded.
4. Every traversed occupied voxel receives at most one vote from each camera.
   Anchors are globally protected from proposed deletion.
5. Compare each proposal with the same GT solid-mesh occupancy used by the
   existing hull geometry evaluation. GT is offline evaluation only.

`removed_is_ghost_precision` answers: among voxels proposed for removal, how
many are actually outside every GT object? `real_voxels_removed_fraction` is
the harmful counterpart: the fraction of baseline real-object voxels that
would be removed.

## Command

```bash
/home/cho/.pyenv/versions/webots_visual_hull/bin/python3 \
  srp/stage1_hull/experiments/anchor_free_space_20260919/anchor_free_space.py \
  stack3_scene0007 stack4_scene0007 stack5_scene0007 \
  --device cuda --batch-size 512 --vote-levels 1 2 3 4
```

Outputs: `data/eval/anchor_free_space_mv2_v12_am1/<scene>/`.

## Anchor quality

| Scene | Anchor rows | Unique anchor voxels | Within 12 mm of GT surface |
| --- | ---: | ---: | ---: |
| stack3_scene0007 | 13,741 | 4,653 | 86.7% |
| stack4_scene0007 | 9,356 | 3,603 | 94.7% |
| stack5_scene0007 | 10,201 | 4,339 | 92.7% |

This metric only says that an anchor is near *some* true surface. It does not
say that it is the first true surface on its reference ray.

## Result: unsafe direct carving

| Minimum cameras | stack3 ghost precision / real loss | stack4 ghost precision / real loss | stack5 ghost precision / real loss |
| --- | ---: | ---: | ---: |
| 1 | 15.5% / 52.4% | 15.4% / 47.1% | 25.8% / 48.8% |
| 2 | 19.7% / 26.8% | 21.0% / 25.9% | 34.6% / 30.0% |
| 3 | 28.6% / 11.6% | 30.3% / 13.7% | 45.7% / 16.8% |
| 4 | 37.5% / 5.3% | 43.3% / 7.1% | 57.7% / 8.8% |

Higher thresholds remain non-uniform. At 9 cameras, precision is 63.2%,
85.0%, and 90.8% for stack3/4/5 respectively, but only 43, 193, and 178 ghost
voxels are removed. At 10 cameras, stack3 removes just 10 ghost voxels and
still removes 3 real voxels. No fixed 1--12 camera threshold is safe across
all three scenes while retaining useful removal.

## Decision

Do not carve from this vote mask. Multi-scale NCC agreement identifies a
plausible surface, but it does not prove visibility order. A rear or occluded
true surface can still receive a high NCC score; declaring all occupied cells
in front of it free then cuts through the actual object.

The next experiment must test first-surface order before any free-space vote:
for an anchor projected into a source view, use that source view's independent
agreed-anchor depth map and accept support only when it selects the same 3-D
anchor at the projected pixel. An anchor without enough such source support
stays unknown, not free space. Its proposed removal mask must again be
measured against GT before any hull is written.
