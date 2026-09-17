# Uncertainty-aware FLEXTH

FLEXTH (Betterle & Salamon, 2024) enhances satellite flood maps using terrain:
it estimates water levels and depths and extends flooding into topographically
connected low-lying areas. This repository adds an **uncertainty gate** that
restricts that extension according to a per-pixel uncertainty map, e.g. from a
deep-learning flood segmentation model. It also provides a batch runner that
compares runs with and without the gate.

- **What the gate does and how to choose a threshold:** [docs/UNCERTAINTY_GATE.md](docs/UNCERTAINTY_GATE.md)
- **Required rasters and formats:** [docs/INPUTS.md](docs/INPUTS.md)
- **FLEXTH model parameters:** [docs/PARAMETERS.md](docs/PARAMETERS.md)

This repository does not provide input data or tools to obtain it. You need
your own flood map, DTM and uncertainty rasters.

## Repository layout

```text
FLEXTH.py                 FLEXTH with the uncertainty gate (the model)
flexth_batch_main.py      aligns inputs, runs one FLEXTH job per case
flexth_post_analysis.py   compares runs with/without the gate
DTM_2_floodmap.py         (upstream) aligns one raster to the flood map grid; needs GDAL
configs/
  example.legacy_gate.json    gate from uncertainty.tif only
  example.semantic_gate.json  gate from classification + probability + uncertainty
docs/                     gate, inputs and parameter documentation
test_case/                link to the upstream FLEXTH test data
```

## Installation

Tested on Linux with Python 3.13. Pick one of the following.

**uv** (uses the locked versions in `uv.lock`):

```bash
uv sync
source .venv/bin/activate
```

**conda / mamba:**

```bash
mamba env create -f environment.yml
mamba activate flexth-uncertainty
```

Check the environment:

```bash
python -c "import rasterio, cv2, astropy, scipy, imageio; print('ok')"
```

## Quick start

**1. Prepare inputs.** Put your rasters in one folder, for example:

```text
data/my_event/
  flood.tif          1 = flooded
  dtm.tif            ground elevation
  uncertainty.tif    lower = more confident
```

For the semantic gate, also add `classification.tif` and `water_probability.tif`.
The rasters do not need to share a grid, because the batch runner aligns them
to `flood.tif`. See [docs/INPUTS.md](docs/INPUTS.md).

**2. Create a config.**

```bash
cp configs/example.legacy_gate.json configs/local.my_event.json
```

Edit `base_inputs` so it points to your files. Replace the values in
`retention_curve` with thresholds computed for **your** uncertainty map, because
the example values only apply to the ML4Floods EDL model (see
[how to compute them](docs/UNCERTAINTY_GATE.md#computing-thresholds-for-your-own-uncertainty-map)).
Local configs matching `configs/local*.json` are git-ignored.

**3. Run the cases.**

```bash
python flexth_batch_main.py \
  --config configs/local.my_event.json \
  --flexth-script FLEXTH.py \
  --runs-root results/my_event \
  --analyze
```

Relative paths in the config and on the command line are resolved against the
current working directory.

**4. Inspect the results.** See [Outputs](#outputs).

## Config reference

```jsonc
{
  "base_inputs":     { "flood": "...", "dtm": "...", "uncertainty": "...", ... },
  "retention_curve": { "90": 0.318, "80": 0.141, "70": 0.089, "60": 0.065 },
  "base_params":     { "uncertainty_gate_mode": "legacy_low_uncertainty", "param_...": ... },
  "cases":           [ { "uncertainty_on": 0 }, { "uncertainty_on": 1, "uncertainty_level": 90 } ]
}
```

| Key | Meaning |
|-----|---------|
| `base_inputs` | Paths to input rasters. `flood`, `dtm` and `uncertainty` are always required. The semantic gate also requires `classification` and `water_probability`. Optional: `exclusion`, `permanent_water`, `obswater`. |
| `retention_curve` | Maps a retention level to an uncertainty threshold. |
| `base_params` | Parameters shared by all cases. Includes `uncertainty_gate_mode` (`legacy_low_uncertainty` by default, so always set it), `water_probability_threshold` (semantic gate, default `0.5`), `debug_uncertainty`, and any `param_*` from [docs/PARAMETERS.md](docs/PARAMETERS.md). |
| `cases` | Explicit list of runs. Each entry overrides `base_params`. |
| `sweep` | Alternative to `cases`: a Cartesian product, e.g. `{"uncertainty_on": [0, 1], "param_threshold_slope": [0.1, 0.2]}`. Do not use `sweep` and `cases` together. |

Per-case gate keys:

- `uncertainty_on`: `0` for plain FLEXTH (the baseline), `1` to apply the gate. Default `1`.
- `uncertainty_level`: a key of `retention_curve`. Default `70`.
- `uncertainty_threshold`: sets the threshold directly and takes precedence over `uncertainty_level`.

With `uncertainty_on = 0`, the level and threshold are ignored. A `sweep` over
both keys therefore produces identical baseline runs, so use `cases` with a
single baseline, as the example configs do.

## Outputs

```text
results/my_event/
  run_001__uncertainty_on-0__uncertainty_level-90/
    input/                  aligned copies of the inputs
    output/
      WD_*.tif              water depth (cm)
      WL_*.tif              water level (DTM vertical units)
      flood_expansion.gif   animation of the propagation step (if propagation ran)
    FLEXTH_patched.py       the exact script that produced this run
    run_manifest.json       all resolved parameters
  run_002__uncertainty_on-1__uncertainty_level-90/
  ...
  analysis/                 written when --analyze is used
    run_summary.csv
    uncertainty_diff_summary.csv
    <run>_confusion.png
    <run>_vs_no_uncertainty_diff.tif / .png
    <run>_vs_no_uncertainty_extent_pair.png
```

In the output rasters, permanent water bodies have the value `9999`.

**Post-analysis** matches each `uncertainty_on = 1` run to the baseline run with
the same other parameters. It then reports the flood area removed or added by
the gate (`uncertainty_diff_summary.csv`). For every run, `run_summary.csv`
compares the FLEXTH extent with the **input flood map** (TP/FP/FN, IoU). This
input map is not ground truth, so evaluate against your own reference data if
you have it.

Re-run the analysis without re-running FLEXTH:

```bash
python flexth_post_analysis.py \
  --runs-root results/my_event \
  --input-flood data/my_event/flood.tif
```

## Running FLEXTH.py directly

You can also run `FLEXTH.py` on its own. Edit the input/output paths and
parameters at the top of the file, then run `python FLEXTH.py`. In this mode
there is no raster alignment, so all inputs must already share the flood map's
grid. Note that the standalone default is `uncertainty_gate_mode =
"semantic_high_uncertainty"`.

To align a raster to the flood map, use the upstream helper `DTM_2_floodmap.py`.
Set `input_raster`, `output_raster`, `input_flood_delineation` and
`continuous_input` at the top of the file, and run it once per raster. It uses
GDAL's Python bindings, which are not part of the default environment. Install
them with `mamba install -c conda-forge gdal`.

## Upstream test case

[test_case/INSTRUCTIONS.txt](test_case/INSTRUCTIONS.txt) links to the original
FLEXTH test data (Kakhovka dam 2023, Greece 2023, Odra river 2024). These cases
include a flood map and a DTM but **no uncertainty raster**. They are useful for
checking that FLEXTH runs in your environment: run `FLEXTH.py` directly with
`uncertainty_on = 0`. To try the gate, you need your own uncertainty map.

## Known limitations

- The batch runner configures each run by rewriting variables in a copy of
  `FLEXTH.py` (`FLEXTH_patched.py`). If you rename or reformat those variables,
  the runner stops with `Could not patch ...`.
- Post-analysis defines flood extent as `WL > 0`. Water levels at or below 0 m
  (e.g. below sea level) are not counted.
- Runs execute sequentially. Large rasters can take hours and need a lot of
  memory, so consider `param_tiling`.
- `flood_expansion.gif` is rendered whenever the propagation step runs (up to
  120 frames), which adds time on large rasters.

## Citation and licence

If you use this code, please cite the original FLEXTH paper:

> Betterle, A. and Salamon, P.: Water depth estimate and flood extent enhancement
> for satellite-based inundation maps, Nat. Hazards Earth Syst. Sci., 24,
> 2817, https://doi.org/10.5194/nhess-24-2817-2024, 2024.

FLEXTH is © European Union and licensed under the
[EUPL v1.2](LICENSE). This repository is a modified version of FLEXTH v1.3.0 and
is distributed under the same licence. Third-party notices are in
[NOTICE.txt](NOTICE.txt). The original FLEXTH code is available at
https://code.europa.eu/floods/floods-river/flexth.
