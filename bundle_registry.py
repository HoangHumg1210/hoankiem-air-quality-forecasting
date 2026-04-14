from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_BUNDLE_FILES = (
    "config.json",
    "metrics.json",
    "feature_cols.pkl",
    "x_scaler.pkl",
    "y_scaler.pkl",
    "model.keras",
)
OPTIONAL_BUNDLE_FILES = (
    "test_timeline.csv",
    "best_info.json",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dedupe_model_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics.copy()

    ranked = metrics.copy()
    ranked["Has Timeline Rank"] = ranked["Has Timeline"].fillna(False).astype(int) * -1
    ranked = ranked.sort_values(
        by=["Model", "Source Rank", "Has Timeline Rank", "MAE"],
        ascending=[True, True, True, True],
        na_position="last",
    )
    deduped = ranked.drop_duplicates(subset=["Model"], keep="first")
    deduped = deduped.sort_values(by=["MAE", "Model"], ascending=[True, True], na_position="last").reset_index(drop=True)
    return deduped.drop(columns=["Source Rank", "Has Timeline Rank"], errors="ignore")


def iter_bundle_dirs(
    *,
    app_dir: str | Path,
    registry_dir: str | Path | None = None,
    best_bundle_dir: str | Path | None = None,
) -> list[tuple[str, Path]]:
    app_path = Path(app_dir).resolve()
    registry_path = Path(registry_dir).resolve() if registry_dir else app_path / "model_registry"
    best_bundle_path = Path(best_bundle_dir).resolve() if best_bundle_dir else app_path / "best_model_bundle"

    bundle_dirs: list[tuple[str, Path]] = []
    if best_bundle_path.exists():
        bundle_dirs.append(("best_model_bundle", best_bundle_path))

    if registry_path.exists():
        for candidate in sorted(registry_path.iterdir()):
            if candidate.is_dir() and (candidate / "config.json").exists():
                bundle_dirs.append(("model_registry", candidate))

    return bundle_dirs


def load_registry_metrics(
    app_dir: str | Path,
    *,
    registry_dir: str | Path | None = None,
    best_bundle_dir: str | Path | None = None,
) -> pd.DataFrame:
    app_path = Path(app_dir).resolve()
    rows: list[dict[str, Any]] = []

    for source_name, bundle_dir in iter_bundle_dirs(
        app_dir=app_path,
        registry_dir=registry_dir,
        best_bundle_dir=best_bundle_dir,
    ):
        config_path = bundle_dir / "config.json"
        metrics_path = bundle_dir / "metrics.json"
        if not config_path.exists() or not metrics_path.exists():
            continue

        config = read_json(config_path)
        metrics = read_json(metrics_path)
        best_info_path = bundle_dir / "best_info.json"
        best_info = read_json(best_info_path) if best_info_path.exists() else {}
        model_name = (
            config.get("model_name")
            or best_info.get("winner_model")
            or config.get("bundle_key")
            or bundle_dir.name
        )
        data_start = best_info.get("train_data_start") or config.get("train_data_start")
        data_end = best_info.get("train_data_end") or config.get("train_data_end")

        rows.append(
            {
                "Model": model_name,
                "Bundle Key": str(config.get("bundle_key", bundle_dir.name)),
                "Bundle Dir": str(bundle_dir.relative_to(app_path)),
                "MAE": metrics.get("mae"),
                "MSE": metrics.get("mse"),
                "RMSE": metrics.get("rmse"),
                "MAPE": metrics.get("mape"),
                "Peak MAE": metrics.get("peak_mae"),
                "Data Start": data_start,
                "Data End": data_end,
                "Has Timeline": (bundle_dir / "test_timeline.csv").exists(),
                "Source": source_name,
                "Source Rank": 1 if source_name == "model_registry" else 2,
            }
        )

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        return metrics_df
    return dedupe_model_metrics(metrics_df)


def select_best_bundle(metrics_df: pd.DataFrame) -> pd.Series:
    if metrics_df.empty:
        raise ValueError("Không có bundle nào để chọn.")
    return metrics_df.sort_values(by=["MAE", "Model"], ascending=[True, True], na_position="last").iloc[0]


def build_best_info(source_bundle_dir: Path, best_row: pd.Series) -> dict[str, Any]:
    metrics_path = source_bundle_dir / "metrics.json"
    source_metrics = read_json(metrics_path) if metrics_path.exists() else {}

    return {
        "winner_model": str(best_row["Model"]),
        "source_bundle": source_bundle_dir.name,
        "source_dir": str(source_bundle_dir),
        "selected_by": "lowest_mae",
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_snapshot": source_metrics,
    }


def promote_best_bundle(
    app_dir: str | Path,
    *,
    registry_dir: str | Path | None = None,
    best_bundle_dir: str | Path | None = None,
    bundle_key: str | None = None,
) -> dict[str, Any]:
    app_path = Path(app_dir).resolve()
    registry_path = Path(registry_dir).resolve() if registry_dir else app_path / "model_registry"
    target_dir = Path(best_bundle_dir).resolve() if best_bundle_dir else app_path / "best_model_bundle"

    metrics_df = load_registry_metrics(
        app_path,
        registry_dir=registry_path,
        best_bundle_dir=target_dir,
    )
    if metrics_df.empty:
        raise ValueError("Không có bundle nào để chọn.")

    if bundle_key:
        matching = metrics_df.loc[metrics_df["Bundle Key"] == bundle_key]
        if matching.empty:
            raise ValueError(f"Không tìm thấy bundle với key: {bundle_key}")
        best_row = matching.iloc[0]
    else:
        best_row = select_best_bundle(metrics_df)

    source_bundle_dir = app_path / str(best_row["Bundle Dir"])
    if not source_bundle_dir.exists():
        raise FileNotFoundError(f"Kông tìm thấy thư mục bundle nguồn: {source_bundle_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_BUNDLE_FILES + OPTIONAL_BUNDLE_FILES:
        source_path = source_bundle_dir / filename
        target_path = target_dir / filename
        if source_path.exists():
            shutil.copy2(source_path, target_path)
        elif target_path.exists() and filename != "best_info.json":
            target_path.unlink()

    best_info = build_best_info(source_bundle_dir, best_row)
    (target_dir / "best_info.json").write_text(
        json.dumps(best_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "source_bundle_dir": str(source_bundle_dir),
        "target_bundle_dir": str(target_dir),
        "model_name": str(best_row["Model"]),
        "bundle_key": str(best_row["Bundle Key"]),
        "mae": float(best_row["MAE"]),
    }
