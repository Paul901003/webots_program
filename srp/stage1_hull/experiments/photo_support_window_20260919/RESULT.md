# Photo Support Window: Offline Discriminability

## Question

The prior RGB carving experiments used a `patch_radius=2` (5x5 pixels) with
no evidence that this window can distinguish a true mesh surface from a deeper
visual-hull ghost. This experiment changes no hull voxel. It measures that
ranking ability before any new carving rule is considered.

## Fixed inputs

- Hull candidates: `hull_multi_ref_candidates_mv2_v12_am1`
- RGB, poses, and foreground: `captures_fast`, `mobilesamv2_fast`, and the
  same FK arm subtraction used by Stage 1.
- Candidate rays and all NCC settings are fixed. Only patch radius changes.
- GT YCB meshes are used only after scoring: a candidate is called surface-like
  when its center lies within 12 mm of a sampled GT mesh surface. The 12 mm
  tolerance is deliberately larger than one voxel because candidates are
  currently spaced by 15 mm in this diagnostic.

## Command

```bash
/home/cho/.pyenv/versions/webots_visual_hull/bin/python3 \
  srp/stage1_hull/experiments/photo_support_window_20260919/window_discriminability.py \
  stack3_scene0007 stack4_scene0007 stack5_scene0007
```

Output: `data/eval/photo_support_window_mv2_v12_am1/<scene>/`.

## Single-scale result

`surface_beats_ghost_rate` is measured only on rays containing both a
surface-like and a ghost candidate. It is the fraction where the best
surface-like NCC exceeds the best ghost NCC.

| Patch | stack3_scene0007 | stack4_scene0007 | stack5_scene0007 | Mean |
| --- | ---: | ---: | ---: | ---: |
| 3x3 | 66.5% | 76.1% | 73.1% | 71.9% |
| 5x5 | 73.6% | 81.5% | 77.6% | 77.6% |
| 7x7 | 79.2% | 85.4% | 81.8% | 82.1% |
| 11x11 | 86.7% | 90.6% | 87.1% | 88.1% |

Larger windows are much better at ranking a true surface candidate ahead of a
ghost, but are valid on fewer rays because the whole patch must remain in the
foreground mask. 5x5 is therefore not a justified default.

## Multi-scale agreement result

The following reruns use only 7x7 and 11x11. A ray is high-confidence only
when both windows select the exact same candidate voxel. This criterion is not
yet used for carving.

| Scene | Rays with both scores | Same-depth rate | Surface-like among same-depth winners |
| --- | ---: | ---: | ---: |
| stack3_scene0007 | 20,484 | 67.1% | 94.9% |
| stack4_scene0007 | 13,975 | 66.9% | 97.7% |
| stack5_scene0007 | 15,377 | 66.3% | 96.5% |

This is promising evidence, not a carve result. The next safe experiment is
to turn same-depth winners into 3-D surface anchors, compute free-space votes
in front of those anchors, and evaluate the proposed removal mask against GT
before changing any hull. Anchor voxels must be globally protected; a ray from
another view must not be allowed to delete them.

## CUDA scorer validation

The window scorer now has an explicit `--device cuda` path. It preserves the
CPU scorer's projection, 9 normal hypotheses, up-to-4 source views, complete
foreground-patch test, bilinear sampling, and NCC formula. Only the array
operations run on the GPU.

On a fixed sample of 256 `stack4_scene0007` candidates at 11x11, CPU and CUDA
had exactly the same 213 scoreable candidates. Their maximum NCC difference
was `1.19e-7` (mean `1.02e-8`).

Full-scene 21x21 validation on `stack4_scene0007` gave exactly the same
ranking statistics as the CPU result: 9,484 scoreable rows, 5,350 rows with a
surface-like and ghost candidate, and `surface_beats_ghost_rate = 95.8505%`.
The CUDA NCC section took 19.0 s with batch size 512; the previous CPU run
with batch size 64 took 102.1 s. A same-sample CPU batch-size-512 check did
not materially change CPU time, so this is a genuine GPU speedup rather than
only a batching effect.

```bash
/home/cho/.pyenv/versions/webots_visual_hull/bin/python3 \
  srp/stage1_hull/experiments/photo_support_window_20260919/window_discriminability.py \
  stack4_scene0007 --radii 10 --batch-size 512 --device cuda \
  --out-root photo_support_window_cuda_mv2_v12_am1
```

This acceleration changes neither the hull nor any carving decision.
