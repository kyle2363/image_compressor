"""Small reproducible timing and error benchmark for the approximation."""

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image

import compressor
from main import load_image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="path to an input image")
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 200])
    parser.add_argument("--max-errors", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    args = parser.parse_args()

    source = load_image(args.image)
    if any(size < 1 for size in args.sizes):
        parser.error("sizes must be positive")
    print("side_px,max_error,measured_error,seconds")
    for size in args.sizes:
        resized = np.asarray(Image.fromarray(source).resize((size, size)))
        for max_error in args.max_errors:
            start = time.perf_counter()
            result = compressor.process_image(resized, max_error)
            elapsed = time.perf_counter() - start
            measured = compressor.mean_absolute_error(resized, result)
            print(f"{size},{max_error:.4f},{measured:.6f},{elapsed:.4f}")


if __name__ == "__main__":
    main()
