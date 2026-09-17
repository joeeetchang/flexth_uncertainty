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

Choose the mode with `uncertainty_gate_mode`. **`semantic_high_uncertainty` is
the recommended mode and the default** in both `FLEXTH.py` and
`flexth_batch_main.py`. The two modes read the same threshold in **opposite**
directions, so do not reuse results or thresholds across modes.

### `semantic_high_uncertainty` (recommended)

This mode is designed for the ML4Floods EDL model's output. Propagation is
allowed into pixels where the optical classification is **unreliable or
water-like**, so terrain can decide. Where the classifier confidently says
"land", terrain is not allowed to override it: **a land pixel may be flooded
only if its uncertainty is at or above the threshold.**

```
high_unc = isfinite(uncertainty) AND uncertainty >= 0 AND uncertainty >= threshold
invalid  = class == 0  OR  water_probability not finite  OR  water_probability < 0

gate = invalid                    # no observation
    OR (class == 3)               # cloud: unobserved, treated like invalid
    OR (class == 1 AND high_unc)  # uncertain land
    OR (class == 4)               # flood trace
```

Invalid and cloud pixels carry no optical information about the surface, so
terrain alone decides. Confident land (`class == 1` with low uncertainty) and
water (`class == 2`) are never propagation candidates. Class 2 pixels are
already flood seeds. The threshold therefore only affects land pixels.

Inputs: `uncertainty.tif`, `classification.tif`, `water_probability.tif`
(see [INPUTS.md](INPUTS.md) for the class codes).

This mode also filters pixels that FLEXTH's own preprocessing (morphological
closing and gap filling) adds to the flood map. Such additions are kept only if
they pass the gate. Original `flood.tif` pixels are always kept.

#### Why cloud is not gated by uncertainty

Earlier versions opened a cloud pixel only if its uncertainty was high or its
water probability was at least 0.5. Those values come from the water head,
which is not reliable under clouds:

- **Not supervised.** WorldFloods marks bright cloud pixels as invalid in the
  land/water training target, so they are excluded from the water-head loss.
  At inference, class 3 is assigned with the same brightness threshold, so
  most class 3 pixels fall in that unsupervised region.
- **Not calibrated.** The retention thresholds are computed on pixels with a
  valid land/water label, which excludes those clouds.
- **Overconfident in practice.** On nine WorldFloods test events, the median
  uncertainty under cloud (0.065) was about the same as over land (0.064),
  even though the sensor cannot see the surface.

Cloud is therefore treated as unobserved, like invalid pixels. This matches how
the original FLEXTH uses exclusion masks for areas without observation. Note
that ground truth cannot validate this choice well: on the same nine events,
90% of class 3 pixels have no valid land/water label.

### `legacy_low_uncertainty`

The earlier gate, kept for comparison and for uncertainty maps without EDL-style
classes. Propagation is allowed only into pixels the model is **confident**
about:

```
gate = isfinite(uncertainty) AND uncertainty < threshold
```

Inputs: `uncertainty.tif`.

Note that for EDL, low uncertainty means strong evidence for *either* land or
water, not "likely water". This gate therefore also opens confidently-land
pixels to terrain-based flooding.

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

| Level | `semantic_high_uncertainty` | `legacy_low_uncertainty` |
|-------|-----------------------------|--------------------------|
| 90 (high threshold) | only land pixels above the 90th-percentile uncertainty open → **most restrictive** | ~90% of pixels pass → **most permissive** |
| 60 (low threshold)  | land pixels above the 60th-percentile uncertainty open → **most permissive** | ~60% of pixels pass → **most restrictive** |

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
| U1 | configuration: `uncertainty_gate_mode` |
| U2 | load and grid-check `classification.tif` and `water_probability.tif`, build the semantic candidate mask (`build_semantic_candidate_mask()`) |
| U3 | preprocessing additions must pass the semantic gate |
| U4 | the gate is combined with the original propagation conditions |
| U5 | semantic inputs are tiled and passed to each tile when `param_tiling = True` |

## Debugging

Set `"debug_uncertainty": true` in `base_params` to print, for every propagation
step, how many neighbours pass each condition and how many the gate blocks. It
is very verbose, so use it only on small rasters.
