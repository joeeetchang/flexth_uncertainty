from __future__ import annotations

import argparse
import itertools
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
except ModuleNotFoundError as exc:
    RUNTIME_IMPORT_ERROR = exc
else:
    RUNTIME_IMPORT_ERROR = None


REQUIRED_INPUT_NAMES = {
    "flood": "flood.tif",
    "dtm": "dtm.tif",
    "uncertainty": "uncertainty.tif",
}

OPTIONAL_INPUT_NAMES = {
    "exclusion": "exclusion.tif",
    "permanent_water": "permanent_water.tif",
    "obswater": "obswater.tif",
    "classification": "classification.tif",
    "water_probability": "water_probability.tif",
}

PATCH_PATTERNS = {
    "input_dir": r"(?m)^input_dir\s*=\s*Path\(r'.*?'\s*\)",
    "output_dir": r"(?m)^output_dir\s*=\s*Path\(r'.*?'\s*\)",
    "flood_path": r'(?m)^flood_path\s*=\s*".*?"',
    "dtm_path": r'(?m)^dtm_path\s*=\s*".*?"',
    "exclusion_path": r'(?m)^exclusion_path\s*=\s*".*?"',
    "permanent_water_path": r'(?m)^permanent_water_path\s*=\s*".*?"',
    "obswater_path": r'(?m)^obswater_path\s*=\s*".*?"',
    "uncertainty_path": r'(?m)^uncertainty_path\s*=\s*".*?"',
    "classification_path": r'(?m)^classification_path\s*=\s*".*?"',
    "water_probability_path": r'(?m)^water_probability_path\s*=\s*".*?"',
    "retention_curve": r"(?m)^retention_curve\s*=\s*\{.*?\}",
    "uncertainty_threshold": r"(?m)^uncertainty_threshold\s*=\s*.*$",
    "uncertainty_on": r"(?m)^uncertainty_on\s*=\s*.*$",
    "uncertainty_gate_mode": r'(?m)^uncertainty_gate_mode\s*=\s*".*?"',
    "water_probability_threshold": r"(?m)^water_probability_threshold\s*=\s*.*$",
    "param_tiling": r"(?m)^param_tiling\s*=\s*.*$",
    "param_tile_inputs": r"(?m)^param_tile_inputs\s*=\s*.*$",
    "param_tile_size": r"(?m)^param_tile_size\s*=\s*.*$",
    "param_merge_tiles": r"(?m)^param_merge_tiles\s*=\s*.*$",
    "param_output_map": r"(?m)^param_output_map\s*=\s*.*$",
    "param_threshold_slope": r"(?m)^param_threshold_slope\s*=\s*.*$",
    "param_size_gaps_close": r"(?m)^param_size_gaps_close\s*=\s*.*$",
    "param_max_number_neighbors": r"(?m)^param_max_number_neighbors\s*=\s*.*$",
    "param_inverse_dist_exp": r"(?m)^param_inverse_dist_exp\s*=\s*.*$",
    "param_border_subsampling": r"(?m)^param_border_subsampling\s*=\s*.*$",
    "param_min_flood_border_size": r"(?m)^param_min_flood_border_size\s*=\s*.*$",
    "param_border_quantile": r"(?m)^param_border_quantile\s*=\s*.*$",
    "param_inner_quantile": r"(?m)^param_inner_quantile\s*=\s*.*$",
    "param_spread_outside_exclusion_mask": r"(?m)^param_spread_outside_exclusion_mask\s*=\s*.*$",
    "param_max_propagation_distance": r"(?m)^param_max_propagation_distance\s*=\s*.*$",
    "param_distance_range": r"(?m)^param_distance_range\s*=\s*.*$",
    "param_WD_star": r"(?m)^param_WD_star\s*=\s*.*$",
    "param_WL_estimation_method": r"(?m)^param_WL_estimation_method\s*=\s*.*$",
    "debug_uncertainty": r"(?m)^\s*debug_uncertainty\s*=\s*.*$",
}


@dataclass
class RunContext:
    name: str
    run_dir: Path
    input_dir: Path
    output_dir: Path
    flexth_script: Path


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_required_inputs(
    base_inputs: dict[str, str],
    base_params: dict[str, Any],
) -> None:
    missing = [key for key in REQUIRED_INPUT_NAMES if key not in base_inputs]
    if (
        base_params.get(
            "uncertainty_gate_mode",
            "legacy_low_uncertainty",
        )
        == "semantic_high_uncertainty"
    ):
        missing.extend(
            key
            for key in ("classification", "water_probability")
            if key not in base_inputs
        )
    if missing:
        raise ValueError(f"Missing required inputs in config: {missing}")


def iter_cases(config: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    explicit_cases = config.get("cases")
    sweep = config.get("sweep", {})
    if explicit_cases is not None and sweep:
        raise ValueError("Use either 'cases' or 'sweep', not both.")

    if explicit_cases is not None:
        if not isinstance(explicit_cases, list) or not explicit_cases:
            raise ValueError("'cases' must be a non-empty list.")
        for index, case in enumerate(explicit_cases, start=1):
            if not isinstance(case, dict):
                raise ValueError("Every item in 'cases' must be an object.")
            suffix = "__".join(
                f"{key}-{sanitize_name(value)}" for key, value in case.items()
            )
            yield f"run_{index:03d}__{suffix}", case
        return

    if not sweep:
        yield "run_001", {}
        return

    keys = list(sweep.keys())
    values = [sweep[key] for key in keys]
    for index, combo in enumerate(itertools.product(*values), start=1):
        case = dict(zip(keys, combo))
        suffix = "__".join(f"{key}-{sanitize_name(value)}" for key, value in case.items())
        yield f"run_{index:03d}__{suffix}", case


def sanitize_name(value: Any) -> str:
    return str(value).replace(" ", "_").replace("/", "-").replace("\\", "-")


def prepare_run_dirs(root: Path, run_name: str) -> RunContext:
    run_dir = root / run_name
    input_dir = run_dir / "input"
    output_dir = run_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(
        name=run_name,
        run_dir=run_dir,
        input_dir=input_dir,
        output_dir=output_dir,
        flexth_script=run_dir / "FLEXTH_patched.py",
    )


def copy_or_align_inputs(base_inputs: dict[str, str], ctx: RunContext) -> None:
    flood_src = Path(base_inputs["flood"])
    flood_dst = ctx.input_dir / REQUIRED_INPUT_NAMES["flood"]
    shutil.copy2(flood_src, flood_dst)

    with rasterio.open(flood_dst) as flood_ref:
        for key, filename in REQUIRED_INPUT_NAMES.items():
            if key == "flood":
                continue
            align_to_reference(Path(base_inputs[key]), ctx.input_dir / filename, flood_ref)

        for key, filename in OPTIONAL_INPUT_NAMES.items():
            source = base_inputs.get(key)
            if not source:
                continue
            source_path = Path(source)
            if not source_path.exists():
                print(f"[SKIP] Optional input not found: {source_path}")
                continue
            align_to_reference(source_path, ctx.input_dir / filename, flood_ref)


def align_to_reference(src_path: Path, dst_path: Path, ref: rasterio.io.DatasetReader) -> None:
    with rasterio.open(src_path) as src:
        src_data = src.read(1)
        destination = np.empty((ref.height, ref.width), dtype=src.dtypes[0])
        destination.fill(src.nodata if src.nodata is not None else 0)

        reproject(
            source=src_data,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            src_nodata=src.nodata,
            dst_nodata=src.nodata if src.nodata is not None else 0,
            resampling=choose_resampling(src_path.name),
        )

        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            width=ref.width,
            height=ref.height,
            transform=ref.transform,
            crs=ref.crs,
            count=1,
            compress="deflate",
        )

        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(destination, 1)


def choose_resampling(filename: str) -> Resampling:
    lower = filename.lower()
    if any(
        token in lower
        for token in (
            "classification",
            "flood",
            "exclusion",
            "permanent_water",
            "obswater",
        )
    ):
        return Resampling.nearest
    return Resampling.bilinear


def build_patch_values(
    ctx: RunContext,
    params: dict[str, Any],
    retention_curve: dict[str, float],
) -> dict[str, str]:
    uncertainty_key = params.get("uncertainty_level")
    if uncertainty_key is None and "uncertainty_threshold" not in params:
        uncertainty_key = 70

    if "uncertainty_threshold" in params:
        uncertainty_threshold = params["uncertainty_threshold"]
    else:
        uncertainty_threshold = retention_curve[str(uncertainty_key)] if str(uncertainty_key) in retention_curve else retention_curve[uncertainty_key]

    values = {
        "input_dir": f"input_dir  = Path(r'{ctx.input_dir.as_posix()}')",
        "output_dir": f"output_dir = Path(r'{ctx.output_dir.as_posix()}')",
        "flood_path": f'flood_path = "{(ctx.input_dir / "flood.tif").as_posix()}"',
        "dtm_path": f'dtm_path   = "{(ctx.input_dir / "dtm.tif").as_posix()}"',
        "exclusion_path": f'exclusion_path = "{(ctx.input_dir / "exclusion.tif").as_posix()}"',
        "permanent_water_path": f'permanent_water_path = "{(ctx.input_dir / "permanent_water.tif").as_posix()}"',
        "obswater_path": f'obswater_path = "{(ctx.input_dir / "obswater.tif").as_posix()}"',
        "uncertainty_path": f'uncertainty_path = "{(ctx.input_dir / "uncertainty.tif").as_posix()}"',
        "classification_path": f'classification_path = "{(ctx.input_dir / "classification.tif").as_posix()}"',
        "water_probability_path": f'water_probability_path = "{(ctx.input_dir / "water_probability.tif").as_posix()}"',
        "retention_curve": f"retention_curve = {json.dumps(retention_curve, ensure_ascii=True)}",
        "uncertainty_threshold": f"uncertainty_threshold = float({repr(float(uncertainty_threshold))})",
        "uncertainty_on": f"uncertainty_on = {int(params.get('uncertainty_on', 1))}",
        "uncertainty_gate_mode": f"uncertainty_gate_mode = {json.dumps(params.get('uncertainty_gate_mode', 'legacy_low_uncertainty'))}",
        "water_probability_threshold": f"water_probability_threshold = {float(params.get('water_probability_threshold', 0.5))}",
        "param_tiling": f"param_tiling      = {bool(params.get('param_tiling', False))}",
        "param_tile_inputs": f"param_tile_inputs = {bool(params.get('param_tile_inputs', False))}",
        "param_tile_size": f"param_tile_size   = {int(params.get('param_tile_size', 10000))}",
        "param_merge_tiles": f"param_merge_tiles = {bool(params.get('param_merge_tiles', True))}",
        "param_output_map": f"param_output_map = {repr(params.get('param_output_map', 'WL_WD'))}",
        "param_threshold_slope": f"param_threshold_slope               = {params.get('param_threshold_slope', 0.1)}",
        "param_size_gaps_close": f"param_size_gaps_close               = {params.get('param_size_gaps_close', 0.1)}",
        "param_max_number_neighbors": f"param_max_number_neighbors          = {params.get('param_max_number_neighbors', 200)}",
        "param_inverse_dist_exp": f"param_inverse_dist_exp              = {params.get('param_inverse_dist_exp', 1)}",
        "param_border_subsampling": f"param_border_subsampling            = {params.get('param_border_subsampling', 0)}",
        "param_min_flood_border_size": f"param_min_flood_border_size         = {params.get('param_min_flood_border_size', 10)}",
        "param_border_quantile": f"param_border_quantile               = {params.get('param_border_quantile', 0.5)}",
        "param_inner_quantile": f"param_inner_quantile                = {params.get('param_inner_quantile', 0.98)}",
        "param_spread_outside_exclusion_mask": f"param_spread_outside_exclusion_mask = {bool(params.get('param_spread_outside_exclusion_mask', True))}",
        "param_max_propagation_distance": f"param_max_propagation_distance      = {params.get('param_max_propagation_distance', 10)}",
        "param_distance_range": f"param_distance_range                = {params.get('param_distance_range', 10)}",
        "param_WD_star": f"param_WD_star                       = {params.get('param_WD_star', 10)}",
        "param_WL_estimation_method": f"param_WL_estimation_method = {repr(params.get('param_WL_estimation_method', 'method_A'))}",
        "debug_uncertainty": f"    debug_uncertainty = {bool(params.get('debug_uncertainty', False))}",
    }
    return values


def resolve_full_params(base_params: dict[str, Any], case_params: dict[str, Any], retention_curve: dict[str, float]) -> dict[str, Any]:
    params = {**base_params, **case_params}
    gate_mode = params.get(
        "uncertainty_gate_mode",
        "legacy_low_uncertainty",
    )
    if gate_mode not in {
        "legacy_low_uncertainty",
        "semantic_high_uncertainty",
    }:
        raise ValueError(f"Unknown uncertainty_gate_mode: {gate_mode!r}")
    params["uncertainty_gate_mode"] = gate_mode
    params.setdefault("water_probability_threshold", 0.5)
    uncertainty_key = params.get("uncertainty_level")
    if uncertainty_key is None and "uncertainty_threshold" not in params:
        uncertainty_key = 70

    if "uncertainty_threshold" not in params:
        params["uncertainty_threshold"] = (
            retention_curve[str(uncertainty_key)] if str(uncertainty_key) in retention_curve else retention_curve[uncertainty_key]
        )
    if "uncertainty_on" not in params:
        params["uncertainty_on"] = 1
    return params


def patch_flexth_script(source: Path, destination: Path, patch_values: dict[str, str]) -> None:
    text = source.read_text(encoding="utf-8")
    for key, pattern in PATCH_PATTERNS.items():
        if key not in patch_values:
            continue
        text, count = re.subn(pattern, patch_values[key], text, count=1)
        if count != 1:
            raise RuntimeError(f"Could not patch {key}. FLEXTH.py layout may have changed.")
    destination.write_text(text, encoding="utf-8")


def run_flexth(python_exe: str, script_path: Path, cwd: Path) -> None:
    command = [python_exe, str(script_path)]
    subprocess.run(command, cwd=str(cwd), check=True)


def write_manifest(ctx: RunContext, case_params: dict[str, Any]) -> None:
    manifest_path = ctx.run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(case_params, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch runner for FLEXTH combinations.")
    parser.add_argument("--config", required=True, type=Path, help="Path to batch config JSON.")
    parser.add_argument("--flexth-script", required=True, type=Path, help="Path to original FLEXTH.py.")
    parser.add_argument(
        "--runs-root",
        required=True,
        type=Path,
        help="Directory where generated input/output folders and results will be stored.",
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used to run FLEXTH.py. Default: current Python.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run flexth_post_analysis.py after all experiments finish.",
    )
    parser.add_argument(
        "--analysis-input-flood",
        type=Path,
        help="Original flood.tif for post-run TP/FP/FN metrics. Defaults to base_inputs.flood.",
    )
    args = parser.parse_args()

    if RUNTIME_IMPORT_ERROR is not None:
        raise SystemExit(
            "Missing Python dependency: "
            f"{RUNTIME_IMPORT_ERROR.name}. Run this script in the same conda/Python environment used for FLEXTH."
        )

    # FLEXTH runs with cwd=<run_dir>, so relative paths must be resolved here.
    args.runs_root = args.runs_root.resolve()
    args.flexth_script = args.flexth_script.resolve()

    config = load_config(args.config)
    base_inputs = config.get("base_inputs", {})
    base_params = config.get("base_params", {})
    retention_curve = config.get("retention_curve", {"90": 0.31842, "80": 0.14068, "70": 0.088596, "60": 0.06533})

    ensure_required_inputs(base_inputs, base_params)
    args.runs_root.mkdir(parents=True, exist_ok=True)

    for run_name, case_params in iter_cases(config):
        ctx = prepare_run_dirs(args.runs_root, run_name)
        print(f"[RUN] {ctx.name}")
        copy_or_align_inputs(base_inputs, ctx)
        full_params = resolve_full_params(base_params, case_params, retention_curve)
        patch_values = build_patch_values(ctx, full_params, retention_curve)
        patch_flexth_script(args.flexth_script, ctx.flexth_script, patch_values)
        write_manifest(ctx, full_params)
        run_flexth(args.python_exe, ctx.flexth_script, ctx.run_dir)

    if args.analyze:
        analysis_script = Path(__file__).with_name("flexth_post_analysis.py")
        input_flood = args.analysis_input_flood or Path(base_inputs["flood"])
        subprocess.run(
            [
                args.python_exe,
                str(analysis_script),
                "--runs-root",
                str(args.runs_root),
                "--input-flood",
                str(input_flood),
            ],
            check=True,
        )

    print("All runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
