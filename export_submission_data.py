from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd

from inference import (
    DEFAULT_STEP_HOURS,
    DEFAULT_TIMESTAMP_COL,
    build_history_feature_frame,
    prepare_raw_frame,
)


APP_DIR = Path(__file__).resolve().parent
DEFAULT_CLEANED_INPUT = APP_DIR / "data" / "processed" / "data2225_done.csv"
DEFAULT_BUNDLE_DIR = APP_DIR / "best_model_bundle"
DEFAULT_OUTPUT_DIR = APP_DIR / "data" / "submission"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export cleaned and model-ready CSV files for submission."
    )
    parser.add_argument(
        "--cleaned-input",
        type=Path,
        default=DEFAULT_CLEANED_INPUT,
        help="Path to the cleaned dataset CSV.",
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=DEFAULT_BUNDLE_DIR,
        help="Model bundle directory containing feature_cols.pkl and config.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write submission CSV files into.",
    )
    parser.add_argument(
        "--step-hours",
        type=int,
        default=DEFAULT_STEP_HOURS,
        help="Resample frequency used by the model pipeline.",
    )
    return parser.parse_args()


def load_feature_cols(bundle_dir: Path) -> list[str]:
    feature_cols_path = bundle_dir / "feature_cols.pkl"
    if not feature_cols_path.exists():
        raise FileNotFoundError(f"Missing feature_cols.pkl: {feature_cols_path}")
    with feature_cols_path.open("rb") as file_obj:
        return list(pickle.load(file_obj))


def export_submission_files(
    cleaned_input: Path,
    bundle_dir: Path,
    output_dir: Path,
    *,
    step_hours: int,
) -> tuple[Path, Path]:
    if not cleaned_input.exists():
        raise FileNotFoundError(f"Cleaned input does not exist: {cleaned_input}")

    feature_cols = load_feature_cols(bundle_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_df = pd.read_csv(cleaned_input)
    cleaned_output = output_dir / f"{cleaned_input.stem}_cleaned.csv"
    cleaned_df.to_csv(cleaned_output, index=False)

    prepared = prepare_raw_frame(cleaned_df, step_hours=step_hours)
    featured = build_history_feature_frame(prepared)
    required_cols = feature_cols + ["PM25"]
    model_ready = featured.dropna(subset=required_cols).reset_index()
    model_ready = model_ready[[DEFAULT_TIMESTAMP_COL] + required_cols]

    model_ready_output = output_dir / f"{cleaned_input.stem}_model_ready.csv"
    model_ready.to_csv(model_ready_output, index=False)

    return cleaned_output, model_ready_output


def main() -> None:
    args = parse_args()
    cleaned_output, model_ready_output = export_submission_files(
        cleaned_input=args.cleaned_input.resolve(),
        bundle_dir=args.bundle_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        step_hours=args.step_hours,
    )

    print(f"Cleaned CSV: {cleaned_output}")
    print(f"Model-ready CSV: {model_ready_output}")


if __name__ == "__main__":
    main()
