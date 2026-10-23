"""Command-line interface for the adaptive image approximation demo.

Examples (run from the project directory)::

    python main.py sunset_city.png
    python main.py sunset_city.png --output output/approximation.png --max-error 0.05
    python main.py dark_city.png -o output/dark-city.png --max-error 0.1
    python main.py sunset_city.png --detail 75 -o output/detailed.png
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

import compressor


def detail_to_max_error(detail: int) -> float:
    """Map the 0–100 detail control logarithmically to a MAE target."""
    if isinstance(detail, bool) or not isinstance(detail, int) or not 0 <= detail <= 100:
        raise ValueError("detail must be an integer between 0 and 100")
    coarse_error = 0.30
    fine_error = 0.003
    progress = detail / 100
    return coarse_error * (fine_error / coarse_error) ** progress


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode == "L":
            return np.asarray(image).copy()
        if image.mode == "RGBA":
            return np.asarray(image).copy()
        return np.asarray(image.convert("RGB")).copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Approximate an image with adaptive average-color regions."
    )
    parser.add_argument("input", type=Path, help="path to the source image")
    parser.add_argument("-o", "--output", type=Path, default=Path("approximation.png"))
    error_control = parser.add_mutually_exclusive_group()
    error_control.add_argument(
        "--max-error",
        type=float,
        help="exact target normalized mean absolute error from 0 to 1",
    )
    error_control.add_argument(
        "--detail",
        type=int,
        metavar="0-100",
        help="convenience detail level; higher values retain more detail",
    )
    args = parser.parse_args()

    if args.detail is not None:
        if not 0 <= args.detail <= 100:
            parser.error("--detail must be between 0 and 100")
        max_error = detail_to_max_error(args.detail)
    else:
        max_error = 0.05 if args.max_error is None else args.max_error
    if not math.isfinite(max_error) or not 0 <= max_error <= 1:
        parser.error("--max-error must be between 0 and 1")

    source = load_image(args.input)
    approximation = compressor.process_image(source, max_error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(approximation).save(args.output)

    error = compressor.mean_absolute_error(source, approximation)
    print(f"Saved approximation: {args.output}")
    print(f"Dimensions: {source.shape[1]} × {source.shape[0]}")
    if args.detail is not None:
        print(f"Detail: {args.detail}/100")
        print(f"Mapped target mean absolute error: {max_error:.4f}")
    else:
        print(f"Target mean absolute error: {max_error:.4f}")
    print(f"Measured mean absolute error: {error:.4f}")


if __name__ == "__main__":
    main()
