#!/usr/bin/env python3
"""Render an FP16 AMD diagnostic capture and an exposure-matched comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from compare_amd_capture import load


def display(rgb: np.ndarray, exposure: float) -> np.ndarray:
    linear = np.clip(rgb * exposure, 0.0, 1.0)
    srgb = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    return np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def panel(array: np.ndarray, label: str, width: int = 1280) -> Image.Image:
    image = Image.fromarray(array, "RGB")
    height = round(image.height * width / image.width)
    image = image.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height + 42), "black")
    canvas.paste(image, (0, 42))
    ImageDraw.Draw(canvas).text((14, 12), label, fill="white")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--row-pitch", type=int, required=True)
    args = parser.parse_args()

    before = load(args.directory / "before_00.raw", args.height, args.width, args.row_pitch)[..., :3]
    after = load(args.directory / "after_00.raw", args.height, args.width, args.row_pitch)[..., :3]
    weights = np.array((0.2126, 0.7152, 0.0722), dtype=np.float32)
    before_luma = before @ weights
    after_luma = after @ weights
    centred_before = before_luma - before_luma.mean()
    centred_after = after_luma - after_luma.mean()
    gain = np.mean(centred_before * centred_after) / np.mean(centred_before**2)
    exposure = 0.9 / max(float(np.percentile(before_luma, 99.5)), 1.0e-4)
    matched = after / max(float(gain), 1.0e-4)
    difference = np.abs(matched - before)

    images = [
        panel(display(before, exposure), "Before (game input)"),
        panel(display(after, exposure), "Neural output, same exposure"),
        panel(display(matched, exposure), f"Neural output, exposure matched ({1.0/gain:.3f}x)"),
        panel(display(difference, exposure * 12.0), "Absolute residual after exposure match (12x)"),
    ]
    out = Image.new("RGB", (images[0].width * 2, images[0].height * 2), "black")
    out.paste(images[0], (0, 0))
    out.paste(images[1], (images[0].width, 0))
    out.paste(images[2], (0, images[0].height))
    out.paste(images[3], (images[0].width, images[0].height))
    destination = args.directory / "comparison.png"
    out.save(destination, optimize=True)
    print(destination)


if __name__ == "__main__":
    main()
