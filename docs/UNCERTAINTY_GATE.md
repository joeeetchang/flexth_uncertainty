# The uncertainty gate

This repository extends [FLEXTH](https://code.europa.eu/floods/floods-river/flexth)
(Betterle & Salamon, 2024) with an optional, pixel-wise **uncertainty gate**.
This page explains what the gate changes, the two gate modes, and how to pick a
threshold for your own uncertainty map.

## What the gate changes, and what it does not

FLEXTH estimates a water level for each flooded area from the terrain along its
border, then **propagates** flood water into neighbouring pixels. A neighbour
becomes flooded only if all of these conditions hold:

| # | Condition | Source |
|---|-----------|--------|
| 1 | inside the propagation domain (exclusion / observed water / permanent water) | original FLEXTH |
| 2 | ground elevation below the propagating water level | original FLEXTH |
| 3 | not already assigned a water level | original FLEXTH |
| 4 | **passes the uncertainty gate** | this extension |

The gate adds condition 4 and nothing else:

- The water level interpolation (IDW / quantile, slope filtering, etc.) is unchanged.
- The initial flood pixels in `flood.tif` are never removed by the gate.
- With `uncertainty_on = 0` the gate is all `True`, so the output matches
  unmodified FLEXTH and `uncertainty_threshold` has **no effect**.

In short, **uncertainty constrains where FLEXTH may propagate flood water. It does
not change the water depth formula.**

## Gate modes

Choose the mode with `uncertainty_gate_mode`. The two modes read the same
threshold in **opposite** directions, so always set the mode explicitly.

> **Default mismatch:** run standalone, `FLEXTH.py` defaults to
> `semantic_high_uncertainty`. `flexth_batch_main.py` defaults to
> `legacy_low_uncertainty` when a config omits the key. Both example configs set
> it explicitly.

### `legacy_low_uncertainty`

Propagation is allowed only into pixels the model is **confident** about:

```
gate = isfinite(uncertainty) AND uncertainty < threshold
```

Inputs: `uncertainty.tif`.

### `semantic_high_uncertainty`

This mode is designed for the ML4Floods EDL model's output. Propagation is
allowed into pixels where the optical classification is **unreliable or
water-like**, so terrain can decide. Where the classifier confidently says
"land", terrain is not allowed to override it.

```
high_unc = isfinite(uncertainty) AND uncertainty >= 0 AND uncertainty >= threshold
invalid  = class == 0  OR  water_probability not finite  OR  water_probability < 0

gate = invalid
    OR (class == 1 AND high_unc)                                          # uncertain land
    OR (class == 3 AND (high_unc OR water_probability >= p_threshold))    # cloud
    OR (class == 4)                                                       # flood trace
```

`p_threshold` is `water_probability_threshold` (default `0.5`). Confident land
(`class == 1` with low uncertainty) and water (`class == 2`) are never
propagation candidates. Class 2 pixels are already flood seeds.

Inputs: `uncertainty.tif`, `classification.tif`, `water_probability.tif`
(see [INPUTS.md](INPUTS.md) for the class codes).

This mode also filters pixels that FLEXTH's own preprocessing (morphological
closing and gap filling) adds to the flood map. Such additions are kept only if
they pass the gate. Original `flood.tif` pixels are always kept.

## Retention levels and thresholds

Configs select a threshold by **retention level** through `retention_curve`:

```json
"retention_curve": {"90": 0.3184, "80": 0.1407, "70": 0.0886, "60": 0.0653}
```

Retention level *X* is the uncertainty value below which *X*% of valid pixels
fall, i.e. the *X*-th percentile of the uncertainty distribution on a reference
dataset. The values shipped in the example configs were computed for the
ML4Floods EDL model's `Water_DST_Uncertainty` band on the WorldFloods test
events. **They are not meaningful for other models or uncertainty scales.**

The same level means different things in each mode:

| Level | `legacy_low_uncertainty` | `semantic_high_uncertainty` |
|-------|--------------------------|-----------------------------|
| 90 (high threshold) | ~90% of pixels pass → **most permissive** | few pixels count as high-uncertainty → **most restrictive** |
| 60 (low threshold)  | ~60% of pixels pass → **most restrictive** | many pixels count as high-uncertainty → **most permissive** |

### Computing thresholds for your own uncertainty map

Pool the valid pixels of a representative reference set (ideally the same
dataset you evaluate on, or a validation split) and take percentiles:

```python
import numpy as np
import rasterio

values = []
for path in ["event_a/uncertainty.tif", "event_b/uncertainty.tif"]:
    with rasterio.open(path) as src:
        u = src.read(1, masked=True).compressed()
    values.append(u[np.isfinite(u)])
values = np.concatenate(values)

retention_curve = {str(level): float(np.quantile(values, level / 100))
                   for level in (90, 80, 70, 60)}
print(retention_curve)
```

You can also bypass the curve and set a threshold directly per case:
`{"uncertainty_on": 1, "uncertainty_threshold": 0.12}`.

## Where the changes are in the code

Every modification in `FLEXTH.py` is tagged `UNCERTAINTY-FLEXTH CHANGE`:

| Tag | Purpose |
|-----|---------|
| U1 | configuration: `uncertainty_gate_mode`, `water_probability_threshold` |
| U2 | load and grid-check `classification.tif` and `water_probability.tif`, build the semantic candidate mask |
| U3 | preprocessing additions must pass the semantic gate |
| U4 | the gate is combined with the original propagation conditions |
| U5 | semantic inputs are tiled and passed to each tile when `param_tiling = True` |

The gate function is `build_semantic_candidate_mask()`.

## Debugging

Set `"debug_uncertainty": true` in `base_params` to print, for every propagation
step, how many neighbours pass each condition and how many the gate blocks. It
is very verbose, so use it only on small rasters.
