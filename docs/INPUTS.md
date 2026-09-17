# Input rasters

All inputs are single-band GeoTIFFs. FLEXTH works in metres, so the flood map
must be in a **projected** CRS (e.g. UTM), not latitude/longitude.

This repository does not provide data or data-download tools. Bring your own
flood map, DTM and uncertainty rasters.

## Rasters

| File | Required | Values | Resampling* |
|------|----------|--------|-------------|
| `flood.tif` | always | `1` flooded, anything else → `0` | reference grid |
| `dtm.tif` | always | ground elevation (m); `nodata` is respected | bilinear |
| `uncertainty.tif` | always in the batch runner; in FLEXTH when `uncertainty_on = 1` | float, **lower = more confident**; non-finite values never pass a threshold comparison | bilinear |
| `classification.tif` | `semantic_high_uncertainty` | class codes, see below | nearest |
| `water_probability.tif` | `semantic_high_uncertainty` | `0`–`1`; negative or non-finite = invalid | bilinear |
| `exclusion.tif` | optional | `1` = no data / excluded from mapping | nearest |
| `permanent_water.tif` | optional | `1` = permanent or seasonal water | nearest |
| `obswater.tif` | optional | `1` = all observed water (flood ∪ permanent) | nearest |

`permanent_water.tif` and `obswater.tif` are alternatives, so provide at most one.
Permanent water appears in the outputs with a dummy value of `9999`.

\* Resampling applied by `flexth_batch_main.py`, see below.

### Classification codes (`semantic_high_uncertainty`)

The semantic gate expects the ML4Floods EDL class convention:

| Code | Meaning | Role in the gate |
|------|---------|------------------|
| 0 | invalid | always a propagation candidate |
| 1 | land | a candidate only if uncertainty ≥ threshold |
| 2 | water | not a candidate (it is the flood seed) |
| 3 | cloud | always a candidate (unobserved, like invalid) |
| 4 | flood trace | always a candidate, but not a seed |

`flood.tif` is normally `classification == 2`. If your classifier uses other
codes, remap them to this convention first.

## Grid alignment

**Batch runner (`flexth_batch_main.py`).** Every input is reprojected onto the
grid of `flood.tif` (CRS, transform and size) and written into
`<run>/input/`. You do not need to pre-align rasters, but the flood map's CRS
must be projected.

The resampling method is chosen from the **source file name**. Names containing
`flood`, `classification`, `exclusion`, `permanent_water` or `obswater` use
nearest-neighbour, and everything else uses bilinear. Name your files so that a
continuous raster (DTM, uncertainty, probability) does not contain one of those
words, e.g. avoid `flood_dtm.tif`.

**Running `FLEXTH.py` directly.** No alignment is done. Every raster must share
the flood map's CRS, transform and shape exactly, otherwise FLEXTH raises
`TypeError: ... don't share the same projections and/or grid`. Use
`DTM_2_floodmap.py` (requires GDAL) to align rasters beforehand. Note that it
uses `mode` resampling for categorical rasters, while the batch runner uses
nearest-neighbour.

## Uncertainty from other models

The gate only compares `uncertainty` with a threshold, so any per-pixel
uncertainty works: entropy, predictive variance, `1 - max softmax`, and so on.
Two conditions apply:

1. **Lower must mean more confident.** If your score is a confidence, convert it
   first (e.g. `1 - confidence`).
2. **Compute thresholds on your own score's scale.** The example
   `retention_curve` values are specific to the EDL model. See
   [UNCERTAINTY_GATE.md](UNCERTAINTY_GATE.md#computing-thresholds-for-your-own-uncertainty-map).

Without an EDL-style classification and water probability, use
`legacy_low_uncertainty`.
