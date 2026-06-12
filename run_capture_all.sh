#!/bin/bash
set -e

WBT="$HOME/webots_program/worlds/ycb_supervisor_four_view_capture_multi.wbt"
WBT_BIN="webots"

update_args() {
    sed -i "s|controllerArgs \[.*\]|controllerArgs $1|" "$WBT"
}

# n1: 64 個單物體場景
for scene in $(seq 1 64); do
    echo "=== n1_scene$(printf '%04d' $scene) ==="
    update_args "[\"--1\" \"--${scene}\"]"
    $WBT_BIN --no-rendering "$WBT"
done

# n3: 61 個場景
for scene in $(seq 1 61); do
    echo "=== n3_scene$(printf '%04d' $scene) ==="
    update_args "[\"--3\" \"--${scene}\"]"
    $WBT_BIN --no-rendering "$WBT"
done

# n4: 61 個場景
for scene in $(seq 1 61); do
    echo "=== n4_scene$(printf '%04d' $scene) ==="
    update_args "[\"--4\" \"--${scene}\"]"
    $WBT_BIN --no-rendering "$WBT"
done

# n5: 61 個場景
for scene in $(seq 1 61); do
    echo "=== n5_scene$(printf '%04d' $scene) ==="
    update_args "[\"--5\" \"--${scene}\"]"
    $WBT_BIN --no-rendering "$WBT"
done

echo "全部完成。"
