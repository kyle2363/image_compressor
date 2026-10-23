"""Adaptive, region-based approximation for 8-bit images."""

from __future__ import annotations

import heapq
from itertools import count

import numpy as np


def mean_absolute_error(original: np.ndarray, approximation: np.ndarray) -> float:
    """Return normalized mean absolute error on a 0–1 scale."""
    original = np.asarray(original, dtype=np.float64)
    approximation = np.asarray(approximation, dtype=np.float64)
    if original.shape != approximation.shape:
        raise ValueError("Images must have matching shapes")
    if original.size == 0:
        raise ValueError("Images must not be empty")
    return float(np.mean(np.abs(original - approximation)) / 255.0)


def process_image(image: np.ndarray, max_error: float = 0.05) -> np.ndarray:
    """Approximate an image with average-color rectangular regions.

    ``max_error`` is the maximum target normalized mean absolute error (0–1).
    The algorithm repeatedly splits the region responsible for the largest
    total absolute error. The returned array has the same shape and dtype as
    the input; input values must be finite and within the 8-bit range.
    """
    source = np.asarray(image)
    if (
        isinstance(max_error, (bool, np.bool_))
        or not isinstance(max_error, (int, float, np.integer, np.floating))
        or not np.isfinite(max_error)
        or not 0 <= max_error <= 1
    ):
        raise ValueError("max_error must be a finite number between 0 and 1")
    if source.ndim not in (2, 3) or source.size == 0:
        raise ValueError("image must be a non-empty grayscale or channel image")
    if source.ndim == 3 and source.shape[2] == 0:
        raise ValueError("image must have at least one channel")
    is_real_numeric = np.issubdtype(source.dtype, np.integer) or np.issubdtype(
        source.dtype, np.floating
    )
    if not is_real_numeric or not np.all(np.isfinite(source)):
        raise ValueError("image values must be finite numbers")
    if np.any(source < 0) or np.any(source > 255):
        raise ValueError("image values must be between 0 and 255")

    work = source.astype(np.float64, copy=False)
    height, width = source.shape[:2]
    samples = source.size
    result = np.empty(source.shape, dtype=np.float64)

    def fill_region(y0: int, y1: int, x0: int, x1: int) -> float:
        pixels = work[y0:y1, x0:x1]
        color = np.mean(pixels, axis=(0, 1))
        result[y0:y1, x0:x1] = color
        return float(np.sum(np.abs(pixels - color)))

    def can_split(y0: int, y1: int, x0: int, x1: int) -> bool:
        return y1 - y0 > 1 or x1 - x0 > 1

    serial = count()
    root = (0, height, 0, width)
    total_error = fill_region(*root)
    queue: list[tuple[float, int, tuple[int, int, int, int], float]] = []
    if can_split(*root):
        heapq.heappush(queue, (-total_error, next(serial), root, total_error))

    while queue and total_error / samples / 255.0 > max_error:
        _, _, (y0, y1, x0, x1), parent_error = heapq.heappop(queue)
        y_mid = (y0 + y1) // 2
        x_mid = (x0 + x1) // 2
        y_ranges = [(y0, y_mid), (y_mid, y1)] if y1 - y0 > 1 else [(y0, y1)]
        x_ranges = [(x0, x_mid), (x_mid, x1)] if x1 - x0 > 1 else [(x0, x1)]

        child_error = 0.0
        children = []
        for child_y0, child_y1 in y_ranges:
            for child_x0, child_x1 in x_ranges:
                child = (child_y0, child_y1, child_x0, child_x1)
                error = fill_region(*child)
                child_error += error
                if can_split(*child):
                    children.append((error, child))

        total_error += child_error - parent_error
        for error, child in children:
            heapq.heappush(queue, (-error, next(serial), child, error))

    if np.issubdtype(source.dtype, np.integer):
        result = np.rint(result)
    return result.astype(source.dtype)
