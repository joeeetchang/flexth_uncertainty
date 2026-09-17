from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
except ModuleNotFoundError as exc:
    RUNTIME_IMPORT_ERROR = exc
else:
    RUNTIME_IMPORT_ERROR = None


NODATA_CLASS = 255


def load_wl(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as src:
        data = src.read(1).astype("float32")
        profile = src.profile.copy()
        nodata = src.nodata

    if nodata is not None:
        data[data == nodata] = np.nan

    data[data == 9999] = np.nan
    return data, profile


def flood_extent_from_wl(wl: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    return np.isfinite(wl) & (wl > threshold)


def pixel_area(profile: dict[str, Any]) -> float:
    transform = profile["transform"]
    return abs(transform.a) * abs(transform.e)


def find_wl(run_dir: Path) -> Path | None:
    candidates = sorted((run_dir / "output").glob("WL_*.tif"))
    if candidates:
        return candidates[0]
    merged = sorted((run_dir / "output").glob("WL_merge_*.tif"))
    return merged[0] if merged else None


def load_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def run_signature(manifest: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    ignored = {"uncertainty_on", "uncertainty_level", "uncertainty_threshold"}
    return tuple(sorted((key, repr(value)) for key, value in manifest.items() if key not in ignored))


def classify_uncertainty_difference(no_unc_wl: Path, unc_wl: Path, out_tif: Path) -> dict[str, float]:
    wl_no, profile = load_wl(no_unc_wl)
    wl_unc, unc_profile = load_wl(unc_wl)

    if wl_no.shape != wl_unc.shape or profile["transform"] != unc_profile["transform"] or profile["crs"] != unc_profile["crs"]:
        wl_unc = align_array_to_profile(wl_unc, unc_profile, profile)

    valid = np.isfinite(wl_no) | np.isfinite(wl_unc)
    flood_no = flood_extent_from_wl(wl_no) & valid
    flood_unc = flood_extent_from_wl(wl_unc) & valid

    diff = np.zeros(wl_no.shape, dtype=np.uint8)
    diff[(~flood_no) & (~flood_unc) & valid] = 0
    diff[flood_no & flood_unc & valid] = 1
    diff[flood_no & (~flood_unc) & valid] = 2
    diff[(~flood_no) & flood_unc & valid] = 3
    diff[~valid] = NODATA_CLASS

    output_profile = profile.copy()
    output_profile.update(dtype="uint8", count=1, nodata=NODATA_CLASS, compress="deflate")
    with rasterio.open(out_tif, "w", **output_profile) as dst:
        dst.write(diff, 1)

    area = pixel_area(profile)
    flooded_no = int(np.sum(flood_no))
    flooded_unc = int(np.sum(flood_unc))
    removed = int(np.sum(flood_no & (~flood_unc) & valid))
    added = int(np.sum((~flood_no) & flood_unc & valid))
    shared = int(np.sum(flood_no & flood_unc & valid))

    return {
        "area_no_uncertainty": flooded_no * area,
        "area_with_uncertainty": flooded_unc * area,
        "area_removed_by_uncertainty": removed * area,
        "area_added_with_uncertainty": added * area,
        "area_shared_flood": shared * area,
        "reduction_ratio_percent": (removed / flooded_no * 100.0) if flooded_no else float("nan"),
    }


def align_array_to_profile(data: np.ndarray, src_profile: dict[str, Any], dst_profile: dict[str, Any]) -> np.ndarray:
    aligned = np.full((dst_profile["height"], dst_profile["width"]), np.nan, dtype="float32")
    reproject(
        source=data,
        destination=aligned,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        src_nodata=np.nan,
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    return aligned


def save_uncertainty_diff_png(diff_tif: Path, png_path: Path, title: str) -> None:
    with rasterio.open(diff_tif) as src:
        diff = src.read(1)

    diff_masked = np.ma.masked_where(diff == NODATA_CLASS, diff)
    cmap = ListedColormap(["white", "lightblue", "red", "orange"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(diff_masked, cmap=cmap, norm=norm, interpolation="none")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], fraction=0.035, pad=0.04)
    cbar.ax.set_yticklabels(["Dry in both", "Flooded in both", "Removed by uncertainty", "Only with uncertainty"])
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_extent_pair_png(no_unc_wl: Path, unc_wl: Path, png_path: Path, title: str) -> None:
    wl_no, profile = load_wl(no_unc_wl)
    wl_unc, unc_profile = load_wl(unc_wl)

    if wl_no.shape != wl_unc.shape or profile["transform"] != unc_profile["transform"] or profile["crs"] != unc_profile["crs"]:
        wl_unc = align_array_to_profile(wl_unc, unc_profile, profile)

    flood_no = flood_extent_from_wl(wl_no)
    flood_unc = flood_extent_from_wl(wl_unc)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(flood_no, cmap="Blues", interpolation="none")
    axes[0].set_title("Flood extent without uncertainty restriction")
    axes[0].axis("off")

    axes[1].imshow(flood_unc, cmap="Blues", interpolation="none")
    axes[1].set_title("Flood extent with uncertainty restriction")
    axes[1].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def read_and_align_binary(binary_path: Path, ref_profile: dict[str, Any]) -> np.ndarray:
    aligned = np.zeros((ref_profile["height"], ref_profile["width"]), dtype=np.uint8)
    with rasterio.open(binary_path) as src:
        reproject(
            source=src.read(1),
            destination=aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
    return aligned == 1


def evaluate_against_input_flood(wl_path: Path, input_flood_path: Path, png_path: Path) -> dict[str, float]:
    wl, profile = load_wl(wl_path)
    flexth_flood = flood_extent_from_wl(wl)
    input_flood = read_and_align_binary(input_flood_path, profile)

    valid = np.isfinite(wl) | input_flood
    flexth_flood &= valid
    input_flood &= valid

    tp = input_flood & flexth_flood
    fp = (~input_flood) & flexth_flood & valid
    fn = input_flood & (~flexth_flood) & valid
    tn = (~input_flood) & (~flexth_flood) & valid

    tp_count = int(np.sum(tp))
    fp_count = int(np.sum(fp))
    fn_count = int(np.sum(fn))
    tn_count = int(np.sum(tn))

    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) else float("nan")
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) else float("nan")
    iou = tp_count / (tp_count + fp_count + fn_count) if (tp_count + fp_count + fn_count) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if np.isfinite(precision + recall) and (precision + recall) else float("nan")

    confusion = np.zeros(wl.shape, dtype=np.uint8)
    confusion[tp] = 1
    confusion[fp] = 2
    confusion[fn] = 3

    save_confusion_png(wl, input_flood, confusion, png_path, precision, recall, iou, f1)

    area = pixel_area(profile)
    return {
        "tp_pixels": tp_count,
        "fp_pixels": fp_count,
        "fn_pixels": fn_count,
        "tn_pixels": tn_count,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": f1,
        "flexth_area": int(np.sum(flexth_flood)) * area,
        "input_flood_area": int(np.sum(input_flood)) * area,
    }


def save_confusion_png(
    wl: np.ndarray,
    input_flood: np.ndarray,
    confusion: np.ndarray,
    png_path: Path,
    precision: float,
    recall: float,
    iou: float,
    f1: float,
) -> None:
    wl_masked = np.ma.masked_invalid(wl)
    valid_wl = wl_masked.compressed()
    vmax = np.percentile(valid_wl, 98) if valid_wl.size else 1

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    im0 = axes[0, 0].imshow(wl_masked, cmap="viridis_r", interpolation="none", vmax=vmax)
    axes[0, 0].set_title("FLEXTH WL output")
    axes[0, 0].axis("off")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.035, pad=0.03).set_label("Water Level (m)")

    axes[0, 1].imshow(input_flood, cmap="gray", interpolation="none")
    axes[0, 1].set_title("Input flood binary")
    axes[0, 1].axis("off")

    im2 = axes[1, 0].imshow(wl_masked, cmap="viridis_r", interpolation="none", vmax=vmax)
    axes[1, 0].contour(input_flood.astype(np.uint8), levels=[0.5], colors="red", linewidths=0.8)
    axes[1, 0].set_title("FLEXTH WL with input flood outline")
    axes[1, 0].axis("off")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.035, pad=0.03).set_label("Water Level (m)")

    cmap_conf = ListedColormap(["black", "green", "orange", "red"])
    norm_conf = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_conf.N)
    axes[1, 1].imshow(confusion, cmap=cmap_conf, norm=norm_conf, interpolation="none")
    axes[1, 1].set_title("Confusion map")
    axes[1, 1].axis("off")
    axes[1, 1].legend(
        handles=[
            Patch(facecolor="green", label="TP: observed & simulated"),
            Patch(facecolor="orange", label="FP: simulated only"),
            Patch(facecolor="red", label="FN: observed only"),
            Patch(facecolor="black", label="TN/background"),
        ],
        loc="lower right",
        fontsize=9,
        framealpha=0.8,
    )

    fig.suptitle(f"Precision={precision:.3f}, Recall={recall:.3f}, IoU={iou:.3f}, F1={f1:.3f}", fontsize=16)
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-process FLEXTH batch runs.")
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--input-flood", type=Path, help="Optional original flood.tif for TP/FP/FN metrics.")
    parser.add_argument("--analysis-dir", type=Path, help="Default: <runs-root>/analysis")
    args = parser.parse_args()

    if RUNTIME_IMPORT_ERROR is not None:
        raise SystemExit(
            "Missing Python dependency: "
            f"{RUNTIME_IMPORT_ERROR.name}. Run this script in the same conda/Python environment used for FLEXTH."
        )

    analysis_dir = args.analysis_dir or (args.runs_root / "analysis")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for run_dir in sorted(args.runs_root.glob("run_*")):
        wl_path = find_wl(run_dir)
        if not wl_path:
            continue
        manifest = load_manifest(run_dir)
        runs.append({"run_dir": run_dir, "wl_path": wl_path, "manifest": manifest, "signature": run_signature(manifest)})

    summary_rows: list[dict[str, Any]] = []
    for run in runs:
        wl, profile = load_wl(run["wl_path"])
        flood = flood_extent_from_wl(wl)
        row = {
            "run": run["run_dir"].name,
            "wl_path": str(run["wl_path"]),
            "flood_area": int(np.sum(flood)) * pixel_area(profile),
            **run["manifest"],
        }
        if args.input_flood:
            png_path = analysis_dir / f"{run['run_dir'].name}_confusion.png"
            row.update(evaluate_against_input_flood(run["wl_path"], args.input_flood, png_path))
            row["confusion_png"] = str(png_path)
        summary_rows.append(row)

    diff_rows: list[dict[str, Any]] = []
    baselines = {run["signature"]: run for run in runs if int(run["manifest"].get("uncertainty_on", 0)) == 0}
    for run in runs:
        if int(run["manifest"].get("uncertainty_on", 0)) != 1:
            continue
        baseline = baselines.get(run["signature"])
        if not baseline:
            continue
        level = run["manifest"].get("uncertainty_level", run["manifest"].get("uncertainty_threshold", "unc"))
        diff_tif = analysis_dir / f"{run['run_dir'].name}_vs_no_uncertainty_diff.tif"
        diff_png = analysis_dir / f"{run['run_dir'].name}_vs_no_uncertainty_diff.png"
        extent_pair_png = analysis_dir / f"{run['run_dir'].name}_vs_no_uncertainty_extent_pair.png"
        stats = classify_uncertainty_difference(baseline["wl_path"], run["wl_path"], diff_tif)
        save_uncertainty_diff_png(diff_tif, diff_png, f"Uncertainty difference ({level})")
        save_extent_pair_png(baseline["wl_path"], run["wl_path"], extent_pair_png, f"Flood extent comparison ({level})")
        diff_rows.append(
            {
                "baseline_run": baseline["run_dir"].name,
                "uncertainty_run": run["run_dir"].name,
                "diff_tif": str(diff_tif),
                "diff_png": str(diff_png),
                "extent_pair_png": str(extent_pair_png),
                **run["manifest"],
                **stats,
            }
        )

    write_csv(analysis_dir / "run_summary.csv", summary_rows)
    write_csv(analysis_dir / "uncertainty_diff_summary.csv", diff_rows)
    print(f"Analysis written to {analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
