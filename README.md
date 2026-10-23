# Adaptive Image Approximation

A small Python project that approximates an image with rectangular regions of
average color. It repeatedly subdivides the region contributing the most total
pixel error until the requested normalized mean absolute error (MAE) is met.

## Run it

Python 3.10 or newer is recommended.

```powershell
python -m pip install -r requirements.txt
python main.py sunset_city.png --output output/approximation.png --max-error 0.05
python main.py sunset_city.png --detail 75 --output output/detailed.png
```

`--max-error` is a normalized MAE target from `0` (pixel-exact) to `1` (the
full 8-bit range). Lower targets generally retain more detail and take longer.
The program prints the measured MAE of the saved approximation. Grayscale, RGB,
and RGBA images are supported; other Pillow modes are converted to RGB.

Use either `--max-error` or `--detail`, not both. `--max-error` sets the exact
MAE target; it defaults to `0.05` when neither option is supplied. The optional
`--detail` convenience control ranges from `0` (coarser) to `100` (finer) and
maps logarithmically to MAE targets from `0.30` down to `0.003`. This is a
convenient scale, not a guarantee of perceptually uniform changes; the program
prints the mapped target so the setting remains transparent.

## Tests

The test suite uses Python's standard library, NumPy, and Pillow:

```powershell
python -m unittest discover -v
```

## Benchmark

Run repeatable size/quality comparisons and capture CSV-like output:

```powershell
python performance.py sunset_city.png --sizes 100 200 --max-errors 0.02 0.05 0.1
```

The benchmark reports input dimensions, requested and measured MAE, and elapsed
time. Timings are single runs and should be treated as rough observations, not
formal performance claims. The method also has a visual limitation: its
axis-aligned rectangular regions can create blocky edges and do not exploit a
standard image codec's predictive or transform coding.

##Notes

> **This is mainly a image approximator not a full compressor yet**
