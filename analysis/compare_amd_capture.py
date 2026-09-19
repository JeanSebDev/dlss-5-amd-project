#!/usr/bin/env python3
"""Measure an AMD diagnostic before/after FP16 capture without displaying it."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load(path: Path, height: int, width: int, row_pitch: int) -> np.ndarray:
    # D3D12 pads every row to rowPitch, except that GetCopyableFootprints does
    # not include trailing padding after the final row in the allocation size.
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    values = np.ndarray(
        shape=(height, width * 4),
        dtype="<f2",
        buffer=raw,
        strides=(row_pitch, np.dtype("<f2").itemsize),
    )
    return np.asarray(values, dtype=np.float32).reshape(height, width, 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--row-pitch", type=int, required=True)
    args = parser.parse_args()

    before = load(args.directory / "before_00.raw", args.height, args.width, args.row_pitch)
    after = load(args.directory / "after_00.raw", args.height, args.width, args.row_pitch)
    valid = np.isfinite(before[..., :3]).all(axis=2) & np.isfinite(after[..., :3]).all(axis=2)
    before_rgb = before[..., :3][valid]
    after_rgb = after[..., :3][valid]
    absolute = np.abs(after_rgb - before_rgb)
    pixel_max = absolute.max(axis=1)
    signal = np.maximum(np.abs(before_rgb), 1.0e-4)
    relative = absolute / signal

    print(f"pixels={before_rgb.shape[0]} valid_fraction={valid.mean():.9f}")
    print("before_mean_rgb=" + ",".join(f"{value:.9g}" for value in before_rgb.mean(axis=0)))
    print("after_mean_rgb=" + ",".join(f"{value:.9g}" for value in after_rgb.mean(axis=0)))
    print("mae_rgb=" + ",".join(f"{value:.9g}" for value in absolute.mean(axis=0)))
    print(f"mae_all={absolute.mean():.9g}")
    print(f"rmse_all={np.sqrt(np.mean((after_rgb - before_rgb) ** 2)):.9g}")
    print(f"relative_mae={relative.mean():.9g}")
    for percentile in (50, 90, 95, 99, 99.9, 100):
        print(f"pixel_max_p{percentile:g}={np.percentile(pixel_max, percentile):.9g}")
    for threshold in (1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2, 1.0e-1):
        print(f"pixels_gt_{threshold:g}={(pixel_max > threshold).mean():.9g}")


if __name__ == "__main__":
    main()
