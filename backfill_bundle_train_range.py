from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from inference import DEFAULT_TIMESTAMP_COL, build_history_feature_frame, prepare_raw_frame


def compute_train_range(raw_df: pd.DataFrame, config: dict) -> tuple[str, str]:
    step_hours = int(config.get("step_hours", 3))
    feature_cols = list(config.get("feature_cols") or [])

    prepared = prepare_raw_frame(raw_df, step_hours=step_hours)
    featured = build_history_feature_frame(prepared)
    required = feature_cols + ["PM25"]
    feature_frame = featured.dropna(subset=required).copy()
    if feature_frame.empty:
        raise ValueError("Feature frame is empty after dropna; cannot infer training range.")

    return str(feature_frame.index.min()), str(feature_frame.index.max())


def iter_bundle_dirs(app_dir: Path) -> list[Path]:
    bundle_dirs: list[Path] = []

    best_bundle_dir = app_dir / "best_model_bundle"
    if (best_bundle_dir / "config.json").exists():
        bundle_dirs.append(best_bundle_dir)

    registry_dir = app_dir / "model_registry"
    if registry_dir.exists():
        for candidate in sorted(registry_dir.iterdir()):
            if candidate.is_dir() and (candidate / "config.json").exists():
                bundle_dirs.append(candidate)

    return bundle_dirs


def backfill_bundle(bundle_dir: Path, raw_df: pd.DataFrame, *, dry_run: bool = False) -> tuple[str, str]:
    config_path = bundle_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    train_data_start, train_data_end = compute_train_range(raw_df, config)

    config["train_data_start"] = train_data_start
    config["train_data_end"] = train_data_end

    if not dry_run:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    return train_data_start, train_data_end


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill train_data_start/train_data_end into bundle config.json files.")
    parser.add_argument("--app-dir", default=".", help="Project root containing model_registry/ and best_model_bundle/")
    parser.add_argument(
        "--raw-data",
        default="data/processed/data2225_done.csv",
        help="Path to the processed raw data CSV, relative to --app-dir unless absolute.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute and print results without writing files.")
    args = parser.parse_args()

    app_dir = Path(args.app_dir).resolve()
    raw_data_path = Path(args.raw_data)
    if not raw_data_path.is_absolute():
        raw_data_path = app_dir / raw_data_path

    raw_df = pd.read_csv(raw_data_path)
    raw_df[DEFAULT_TIMESTAMP_COL] = pd.to_datetime(raw_df[DEFAULT_TIMESTAMP_COL])

    bundle_dirs = iter_bundle_dirs(app_dir)
    if not bundle_dirs:
        raise FileNotFoundError("No bundle directories with config.json were found.")

    for bundle_dir in bundle_dirs:
        start, end = backfill_bundle(bundle_dir, raw_df, dry_run=args.dry_run)
        print(f"{bundle_dir.relative_to(app_dir)}: {start} -> {end}")


if __name__ == "__main__":
    main()
