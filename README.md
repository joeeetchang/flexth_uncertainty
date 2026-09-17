# FLEXTH with Uncertainty

This repository runs [FLEXTH](https://code.europa.eu/floods/floods-river/flexth)
with an **uncertainty gate**, and compares the results with and without it.

FLEXTH takes a flood map and a DTM, estimates water depth, and expands the flood
into nearby low-lying areas. The uncertainty gate limits **where** that expansion
may happen, based on a per-pixel uncertainty map from a flood segmentation model.

The workflow is designed for users who need to:

1. Prepare FLEXTH input rasters.
2. Run FLEXTH with and without the uncertainty gate.
3. Compare the flood extents.
4. Export summary tables and diagnostic maps automatically.

This repository does not provide input data. You need your own rasters.

The original `FLEXTH.py` is not modified during a batch run. Instead, the batch
runner creates a copy called `FLEXTH_patched.py` inside each run folder, with
that run's settings filled in.

## Repository Contents

```text
FLEXTH.py
flexth_batch_main.py
flexth_post_analysis.py
DTM_2_floodmap.py
configs/
  example.semantic_gate.json
  example.legacy_gate.json
docs/
  UNCERTAINTY_GATE.md
  INPUTS.md
  PARAMETERS.md
test_case/
  INSTRUCTIONS.txt
```

Main files:

- `FLEXTH.py`: FLEXTH main model script, with the uncertainty gate added.
- `flexth_batch_main.py`: Batch runner for preparing inputs and running experiments.
- `flexth_post_analysis.py`: Post-processing and comparison script.
- `configs/example.semantic_gate.json`: Example config (recommended gate).
- `configs/example.legacy_gate.json`: Example config (older gate, uncertainty map only).
- `DTM_2_floodmap.py`: Original FLEXTH helper to align a raster to the flood map.

More details:

- `docs/UNCERTAINTY_GATE.md`: how the gate works and how to choose thresholds.
- `docs/INPUTS.md`: input raster formats.
- `docs/PARAMETERS.md`: FLEXTH model parameters.

## Required Inputs

Each FLEXTH run needs:

```text
flood.tif
dtm.tif
uncertainty.tif
classification.tif
water_probability.tif
```

Optional inputs:

```text
exclusion.tif
permanent_water.tif
obswater.tif
```

Input expectations:

- `flood.tif`: binary raster, where `1 = flooded` and `0 = not flooded`.
- `dtm.tif`: digital terrain model.
- `uncertainty.tif`: continuous uncertainty raster (lower = more confident).
- `classification.tif`: model classes, `0 = invalid`, `1 = land`, `2 = water`,
  `3 = cloud`, `4 = flood trace`.
- `water_probability.tif`: water probability from `0` to `1` (negative = invalid).
- All rasters should be georeferenced.
- `flood.tif` must use a projected CRS, not latitude/longitude.

If you only have `flood.tif`, `dtm.tif` and `uncertainty.tif`, use the legacy
gate (see Step 3).

The batch runner aligns all rasters to the `flood.tif` grid before each run, so
they do not need to have the same resolution or extent.

Resampling rules:

- `dtm.tif`, `uncertainty.tif` and `water_probability.tif`: bilinear resampling.
- `flood.tif`, `classification.tif` and other binary/categorical rasters:
  nearest-neighbor resampling.

The resampling method is chosen from the file name, so do not give a continuous
raster a name like `flood_dtm.tif`.

## Python Environment

Tested on Linux with Python 3.13.

With [uv](https://docs.astral.sh/uv/):

```bash
uv sync
source .venv/bin/activate
```

Or with conda / mamba:

```bash
mamba env create -f environment.yml
mamba activate flexth-uncertainty
```

Check the required dependencies:

```bash
python -c "import rasterio, cv2, astropy, scipy, imageio; print('ok')"
```

## Step 1: Create a Local Config

Copy the example config:

```bash
cp configs/example.semantic_gate.json configs/local.my_event.json
```

Local config files matching `configs/local*.json` are ignored by Git, so your
own paths are not committed.

## Step 2: Edit Input Paths

Open your local config and edit `base_inputs`.

Example:

```json
{
  "base_inputs": {
    "flood": "data/my_event/flood.tif",
    "dtm": "data/my_event/dtm.tif",
    "uncertainty": "data/my_event/uncertainty.tif",
    "classification": "data/my_event/classification.tif",
    "water_probability": "data/my_event/water_probability.tif"
  }
}
```

If optional inputs exist, add them:

```json
{
  "base_inputs": {
    "flood": "data/my_event/flood.tif",
    "dtm": "data/my_event/dtm.tif",
    "uncertainty": "data/my_event/uncertainty.tif",
    "classification": "data/my_event/classification.tif",
    "water_probability": "data/my_event/water_probability.tif",
    "exclusion": "data/my_event/exclusion.tif"
  }
}
```

Relative paths are resolved from the folder where you run the command.

## Step 3: Configure Parameters

### Gate mode

Choose the gate in `base_params`:

```json
{
  "base_params": {
    "uncertainty_gate_mode": "semantic_high_uncertainty"
  }
}
```

- `semantic_high_uncertainty` (default, recommended): flood may expand into
  **land pixels with high uncertainty**, plus invalid, cloud and flood-trace
  pixels. Needs all five input rasters.
- `legacy_low_uncertainty`: flood may expand only into pixels with **low
  uncertainty**. Needs only `flood.tif`, `dtm.tif` and `uncertainty.tif`. To use
  it, start from `configs/example.legacy_gate.json` instead.

The two modes read the threshold in opposite directions. See
[docs/UNCERTAINTY_GATE.md](docs/UNCERTAINTY_GATE.md) for details.

### Other fixed parameters

Other fixed FLEXTH parameters also go under `base_params`.

Example:

```json
{
  "base_params": {
    "uncertainty_gate_mode": "semantic_high_uncertainty",
    "param_output_map": "WL_WD",
    "param_tiling": false,
    "param_max_propagation_distance": 10,
    "param_threshold_slope": 0.1,
    "debug_uncertainty": false
  }
}
```

See [docs/PARAMETERS.md](docs/PARAMETERS.md) for all `param_*` options.

### Retention curve

`retention_curve` converts an uncertainty level into a threshold value:

```json
{
  "retention_curve": {
    "90": 0.3184,
    "80": 0.1407,
    "70": 0.0886,
    "60": 0.0653
  }
}
```

**The example values only apply to the ML4Floods EDL model.** If you use a
different model, compute your own values. See
[Computing thresholds](docs/UNCERTAINTY_GATE.md#computing-thresholds-for-your-own-uncertainty-map).

### Runs

List the runs you want under `cases`:

```json
{
  "cases": [
    {"uncertainty_on": 0, "uncertainty_level": 90},
    {"uncertainty_on": 1, "uncertainty_level": 90},
    {"uncertainty_on": 1, "uncertainty_level": 80},
    {"uncertainty_on": 1, "uncertainty_level": 70},
    {"uncertainty_on": 1, "uncertainty_level": 60}
  ]
}
```

Meaning:

- `uncertainty_on = 0`: run baseline FLEXTH without the uncertainty gate.
- `uncertainty_on = 1`: run with the uncertainty gate.
- `uncertainty_level`: selects the threshold from `retention_curve`.

Each item can also override any value from `base_params`.

The baseline ignores `uncertainty_level`, so one baseline run is enough.

## Step 4: Run the Batch

Basic command:

```bash
python flexth_batch_main.py \
  --config configs/local.my_event.json \
  --flexth-script FLEXTH.py \
  --runs-root results/my_event \
  --analyze
```

If FLEXTH should run with a different Python, add `--python-exe`:

```bash
python flexth_batch_main.py \
  --config configs/local.my_event.json \
  --flexth-script FLEXTH.py \
  --runs-root results/my_event \
  --python-exe /path/to/python \
  --analyze
```

Runs execute one after another. Large rasters can take hours and use a lot of
memory. In that case, consider setting `"param_tiling": true`.

## Step 5: Review Outputs

Each run creates a folder:

```text
results/my_event/
  run_001__uncertainty_on-0__uncertainty_level-90/
    input/
    output/
    FLEXTH_patched.py
    run_manifest.json
```

Inside each run folder:

- `input/`: input rasters aligned to the flood map grid.
- `output/WD_*.tif`: water depth (cm).
- `output/WL_*.tif`: water level (same units as the DTM).
- `output/flood_expansion.gif`: animation of the flood expansion (when expansion runs).
- `FLEXTH_patched.py`: the exact script used for this run.
- `run_manifest.json`: all parameters used for this run.

Permanent water bodies have the value `9999` in the output rasters.

Post-analysis outputs are written to:

```text
results/my_event/analysis/
```

Important files:

- `run_summary.csv`: summary for every run.
- `uncertainty_diff_summary.csv`: comparison between baseline and uncertainty runs.
- `*_confusion.png`: TP/FP/FN map against the input flood raster.
- `*_vs_no_uncertainty_diff.png`: categorical difference map.
- `*_vs_no_uncertainty_extent_pair.png`: side-by-side flood extent comparison.
- `*_vs_no_uncertainty_diff.tif`: difference classification GeoTIFF.

Note: the confusion maps compare against the **input** `flood.tif`, not ground
truth.

## Run Post-analysis Only

If the batch has already finished, post-analysis can be re-run separately:

```bash
python flexth_post_analysis.py \
  --runs-root results/my_event \
  --input-flood data/my_event/flood.tif
```

If you do not need confusion metrics:

```bash
python flexth_post_analysis.py \
  --runs-root results/my_event
```

## Run FLEXTH.py Directly (Without the Batch Runner)

1. Edit the input/output paths and parameters at the top of `FLEXTH.py`.
2. Run:

```bash
python FLEXTH.py
```

In this mode, rasters are **not** aligned automatically. All inputs must already
match the `flood.tif` grid.

To align a raster, use `DTM_2_floodmap.py`: set the paths at the top of the file
and run it once per raster. It needs GDAL:

```bash
mamba install -c conda-forge gdal
```

### Test data

[test_case/INSTRUCTIONS.txt](test_case/INSTRUCTIONS.txt) links to the original
FLEXTH test data. The test data has no uncertainty raster, so it can only check
that FLEXTH runs. Set `uncertainty_on = 0` in `FLEXTH.py` to use it.

## Known Limitations

- The batch runner edits variables in a copy of `FLEXTH.py`. If you rename or
  reformat those variables in `FLEXTH.py`, the runner stops with
  `Could not patch ...`.
- Post-analysis counts a pixel as flooded when water level `> 0`. Areas with
  water level at or below 0 m (e.g. below sea level) are not counted.
- The semantic gate treats cloud pixels as unobserved, so only terrain decides
  whether they flood. See
  [docs/UNCERTAINTY_GATE.md](docs/UNCERTAINTY_GATE.md#why-cloud-is-not-gated-by-uncertainty).

## Citation and Licence

If you use this code, please cite the original FLEXTH paper:

> Betterle, A. and Salamon, P.: Water depth estimate and flood extent enhancement
> for satellite-based inundation maps, Nat. Hazards Earth Syst. Sci., 24,
> 2817, https://doi.org/10.5194/nhess-24-2817-2024, 2024.

FLEXTH is © European Union and licensed under the [EUPL v1.2](LICENSE). This
repository is a modified version of FLEXTH v1.3.0 and uses the same licence.
Third-party notices are in [NOTICE.txt](NOTICE.txt).
