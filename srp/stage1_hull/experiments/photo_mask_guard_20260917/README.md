# Foreground-Patch Guard Experiment

## Hypothesis

The original warped-NCC carver accepts a photometric comparison whenever its image
samples are in bounds. A source patch can therefore cross background, a gripper, or
a foreground boundary and still contribute low-NCC deletion evidence.

This experiment only changes evidence eligibility. A source contributes to NCC when
every sample in both the reference 5x5 patch and warped source patch is in that
view's foreground union after the canonical gripper mask has been removed.

## Frozen Inputs

- Input hull: `srp_hull_mv2_v12_am1`
- Views: the existing 12 selected views
- Photo parameters: NCC 0.1, K=4, min sources=2, component=3, radius=2,
  iterations=15, normal tilt=20 degrees
- Output: `srp_hull_mv2_v12_am1_photo_fgpatch`

The baseline `srp_hull_mv2_v12_am1_photo` and original `carve_warp` behavior are
not overwritten. The guard is opt-in through `fg_patch_guard=True`.

## Run

```bash
/home/cho/.pyenv/versions/webots_visual_hull/bin/python3 \
  srp/stage1_hull/experiments/photo_mask_guard_20260917/run_fgpatch.py \
  stack4_scene0007
```

Run all legacy scenes by omitting the scene argument.

## Scope

This removes background and mask-boundary comparisons from negative photo evidence.
It cannot distinguish two different objects inside the same foreground union. That
requires a separate semantic/instance compatibility experiment.
