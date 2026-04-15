from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_BUNDLE_DIR = Path(__file__).resolve().parent / "best_model_bundle"
DEFAULT_TIMESTAMP_COL = "Local Time"
DEFAULT_STEP_HOURS = 3
PRODUCTION_LAGS = (1, 8, 16, 24, 32, 40, 48, 56)
DERIVED_FEATURE_PREFIXES = ("PM25_", "hour_", "dow_", "month_")


@dataclass(slots=True)
class LoadedBundle:
    bundle_dir: Path
    model: Any
    x_scaler: Any
    y_scaler: Any
    feature_cols: list[str]
    config: dict[str, Any]
    metrics: dict[str, Any]
    best_info: dict[str, Any] | None = None

    @property
    def model_name(self) -> str:
        if self.config.get("model_name"):
            return str(self.config["model_name"])
        if self.best_info and self.best_info.get("winner_model"):
            return str(self.best_info["winner_model"])
        return str(self.config.get("bundle_key", "model"))

    @property
    def bundle_key(self) -> str:
        return str(self.config.get("bundle_key", self.bundle_dir.name))

    @property
    def lookback(self) -> int:
        return int(self.config.get("lookback", 72))

    @property
    def chunk_horizon(self) -> int:
        return int(self.config.get("chunk_horizon", 1))

    @property
    def rollout_horizon(self) -> int:
        return int(self.config.get("rollout_horizon", self.chunk_horizon))

    @property
    def target_mode(self) -> str:
        return str(self.config.get("target_transform_mode", "log1p"))

    @property
    def step_hours(self) -> int:
        return int(self.config.get("step_hours", DEFAULT_STEP_HOURS))

    @property
    def decoder_future_cols(self) -> list[str]:
        cols = self.config.get("decoder_future_cols", [])
        if cols is None:
            return []
        return list(cols)

    @property
    def decoder_input_dim(self) -> int:
        if not getattr(self.model, "inputs", None) or len(self.model.inputs) < 2:
            return 1

        decoder_shape = getattr(self.model.inputs[1], "shape", None)
        if decoder_shape is None or decoder_shape[-1] is None:
            return 1
        return int(decoder_shape[-1])

    @property
    def required_raw_columns(self) -> list[str]:
        required = {"PM25"}
        for col in self.feature_cols:
            if col.startswith(DERIVED_FEATURE_PREFIXES):
                continue
            required.add(col)
        return sorted(required)


def _load_keras_model(model_path: Path) -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to load bundle models. Install it before running inference."
        ) from exc

    return tf.keras.models.load_model(model_path, compile=False)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_pickle(path: Path) -> Any:
    with path.open("rb") as file_obj:
        return pickle.load(file_obj)


def load_model_bundle(bundle_dir: str | Path = DEFAULT_BUNDLE_DIR) -> LoadedBundle:
    bundle_path = Path(bundle_dir).resolve()
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle directory does not exist: {bundle_path}")

    config = _read_json(bundle_path / "config.json")
    metrics = _read_json(bundle_path / "metrics.json")
    best_info_path = bundle_path / "best_info.json"
    best_info = _read_json(best_info_path) if best_info_path.exists() else None

    feature_cols = list(_read_pickle(bundle_path / "feature_cols.pkl"))
    x_scaler = _read_pickle(bundle_path / "x_scaler.pkl")
    y_scaler = _read_pickle(bundle_path / "y_scaler.pkl")
    model = _load_keras_model(bundle_path / "model.keras")

    return LoadedBundle(
        bundle_dir=bundle_path,
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_cols=feature_cols,
        config=config,
        metrics=metrics,
        best_info=best_info,
    )


def prepare_raw_frame(
    df: pd.DataFrame,
    *,
    timestamp_col: str = DEFAULT_TIMESTAMP_COL,
    step_hours: int = DEFAULT_STEP_HOURS,
) -> pd.DataFrame:
    frame = df.copy()
    if timestamp_col in frame.columns:
        frame[timestamp_col] = pd.to_datetime(frame[timestamp_col])
        frame = frame.set_index(timestamp_col)
    else:
        frame.index = pd.to_datetime(frame.index)

    frame = frame.sort_index()

    if "IsHoliday" in frame.columns:
        frame["IsHoliday"] = frame["IsHoliday"].astype(float)

    if step_hours > 0:
        rule = f"{step_hours}h"
        agg_map = {col: "mean" for col in frame.columns}
        if "IsHoliday" in agg_map:
            agg_map["IsHoliday"] = "max"
        frame = frame.resample(rule).agg(agg_map).dropna(how="all")

    return frame


def build_history_feature_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    frame = raw_df.copy().sort_index()

    frame["hour"] = frame.index.hour
    frame["dayofweek"] = frame.index.dayofweek
    frame["month"] = frame.index.month
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * frame["dayofweek"] / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["dayofweek"] / 7)
    frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12)

    for lag in PRODUCTION_LAGS:
        frame[f"PM25_lag_{lag}"] = frame["PM25"].shift(lag)

    shifted = frame["PM25"].shift(1)
    frame["PM25_roll_mean_8"] = shifted.rolling(window=8).mean()
    frame["PM25_roll_std_8"] = shifted.rolling(window=8).std()
    frame["PM25_roll_max_8"] = shifted.rolling(window=8).max()
    frame["PM25_roll_min_8"] = shifted.rolling(window=8).min()
    frame["PM25_roll_mean_24"] = shifted.rolling(window=24).mean()
    frame["PM25_roll_std_24"] = shifted.rolling(window=24).std()
    frame["PM25_roll_max_24"] = shifted.rolling(window=24).max()
    frame["PM25_roll_min_24"] = shifted.rolling(window=24).min()
    frame["PM25_diff_1"] = shifted.diff(1)
    frame["PM25_diff_8"] = shifted.diff(8)

    same_hour_lags_3d = ["PM25_lag_8", "PM25_lag_16", "PM25_lag_24"]
    same_hour_lags_7d = [
        "PM25_lag_8",
        "PM25_lag_16",
        "PM25_lag_24",
        "PM25_lag_32",
        "PM25_lag_40",
        "PM25_lag_48",
        "PM25_lag_56",
    ]
    frame["PM25_same_hour_mean_3d"] = frame[same_hour_lags_3d].mean(axis=1)
    frame["PM25_same_hour_mean_7d"] = frame[same_hour_lags_7d].mean(axis=1)
    frame["PM25_same_hour_std_7d"] = frame[same_hour_lags_7d].std(axis=1)
    frame["PM25_same_hour_max_7d"] = frame[same_hour_lags_7d].max(axis=1)

    return frame


def build_decoder_future_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    frame = raw_df.copy().sort_index()

    frame["hour"] = frame.index.hour
    frame["dayofweek"] = frame.index.dayofweek
    frame["month"] = frame.index.month
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * frame["dayofweek"] / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["dayofweek"] / 7)
    frame["month_sin"] = np.sin(2 * np.pi * frame["month"] / 12)
    frame["month_cos"] = np.cos(2 * np.pi * frame["month"] / 12)

    if "IsHoliday" in frame.columns:
        frame["IsHoliday"] = frame["IsHoliday"].astype(float)
    else:
        frame["IsHoliday"] = 0.0

    return frame


def transform_target(
    y_raw: np.ndarray | list[float],
    *,
    scaler: Any | None = None,
    fit: bool = False,
    mode: str = "log1p",
) -> tuple[np.ndarray, Any]:
    y_raw = np.asarray(y_raw, dtype=np.float64).reshape(-1)
    y_clip = np.clip(y_raw, 0.0, None)

    if mode == "log1p":
        y_t = np.log1p(y_clip).reshape(-1, 1)
    elif mode == "sqrt":
        y_t = np.sqrt(y_clip).reshape(-1, 1)
    elif mode == "raw":
        y_t = y_clip.reshape(-1, 1)
    else:
        raise ValueError(f"Invalid target transform mode: {mode}")

    if fit:
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(y_t)
    elif scaler is None:
        raise ValueError("Scaler is required when fit=False.")

    return scaler.transform(y_t).reshape(-1), scaler


def inverse_target(y_scaled: np.ndarray | list[float], scaler: Any, mode: str = "log1p") -> np.ndarray:
    y_scaled = np.asarray(y_scaled, dtype=np.float64)
    original_shape = y_scaled.shape
    y_unscaled = scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)

    if mode == "log1p":
        y_raw = np.expm1(y_unscaled)
    elif mode == "sqrt":
        y_raw = np.square(np.clip(y_unscaled, 0.0, None))
    elif mode == "raw":
        y_raw = y_unscaled
    else:
        raise ValueError(f"Invalid target transform mode: {mode}")

    return np.clip(y_raw, 0.0, None).reshape(original_shape)


def build_inference_inputs(
    history_raw_df: pd.DataFrame,
    *,
    feature_cols: list[str],
    x_scaler: Any,
    y_scaler: Any,
    lookback: int = 72,
    target_mode: str = "log1p",
) -> tuple[np.ndarray, float]:
    feature_frame = build_history_feature_frame(history_raw_df)
    feature_frame = feature_frame.dropna(subset=feature_cols + ["PM25"]).copy()
    if len(feature_frame) < lookback:
        raise ValueError("Not enough history after feature engineering to build inference inputs.")

    x_window = feature_frame[feature_cols].tail(lookback).to_numpy(dtype=np.float32)
    x_scaled = x_scaler.transform(x_window)
    last_target_scaled, _ = transform_target(
        np.array([feature_frame["PM25"].iloc[-1]], dtype=np.float32),
        scaler=y_scaler,
        fit=False,
        mode=target_mode,
    )
    return x_scaled[np.newaxis, ...], float(last_target_scaled[0])


def _validate_future_columns(bundle: LoadedBundle, future_df: pd.DataFrame) -> None:
    required = [col for col in bundle.required_raw_columns if col != "PM25"]
    missing = [col for col in required if col not in future_df.columns]
    if missing:
        raise ValueError(
            "Future covariates are missing required raw columns for inference: "
            + ", ".join(missing)
        )


def scale_decoder_future_features(bundle: LoadedBundle, future_raw_df: pd.DataFrame) -> np.ndarray:
    decoder_future_cols = list(bundle.decoder_future_cols)
    if not decoder_future_cols:
        return np.empty((len(future_raw_df), 0), dtype=np.float32)

    frame = build_decoder_future_frame(future_raw_df)
    missing_cols = [col for col in decoder_future_cols if col not in frame.columns]
    if missing_cols:
        raise ValueError(
            "Future decoder covariates are missing columns after feature engineering: "
            + ", ".join(missing_cols)
        )

    missing_from_feature_cols = [col for col in decoder_future_cols if col not in bundle.feature_cols]
    if missing_from_feature_cols:
        raise ValueError(
            "decoder_future_cols are not present in feature_cols: "
            + ", ".join(missing_from_feature_cols)
        )

    idx = [bundle.feature_cols.index(col) for col in decoder_future_cols]
    values = frame[decoder_future_cols].to_numpy(dtype=np.float32)

    mean = np.asarray(bundle.x_scaler.mean_, dtype=np.float32)[idx]
    scale = np.asarray(bundle.x_scaler.scale_, dtype=np.float32)[idx]
    scale = np.where(np.isclose(scale, 0.0), 1.0, scale)
    return ((values - mean) / scale).astype(np.float32)


def make_decoder_input(
    bundle: LoadedBundle,
    chunk_future: pd.DataFrame,
    *,
    last_target_scaled: float,
) -> np.ndarray:
    decoder_input = np.zeros((1, bundle.chunk_horizon, bundle.decoder_input_dim), dtype=np.float32)
    effective_horizon = min(len(chunk_future), bundle.chunk_horizon)

    if bundle.decoder_future_cols:
        expected_dim = 1 + len(bundle.decoder_future_cols)
        if bundle.decoder_input_dim != expected_dim:
            raise ValueError(
                "Bundle decoder_input_dim does not match decoder_future_cols: "
                f"decoder_input_dim={bundle.decoder_input_dim}, expected={expected_dim}"
            )

        decoder_input[0, :, 0] = float(last_target_scaled)
        if effective_horizon > 0:
            decoder_future = scale_decoder_future_features(bundle, chunk_future)
            decoder_input[0, :effective_horizon, 1:] = decoder_future[:effective_horizon]
        return decoder_input

    if bundle.decoder_input_dim != 1:
        raise ValueError(
            "Bundle expects multi-dimensional decoder input but config.json does not define "
            "`decoder_future_cols`."
        )

    decoder_input[0, 0, 0] = float(last_target_scaled)
    return decoder_input


def forecast_recursive(
    bundle: LoadedBundle,
    history_df: pd.DataFrame,
    future_covariates_df: pd.DataFrame,
    *,
    horizon: int | None = None,
    timestamp_col: str = DEFAULT_TIMESTAMP_COL,
) -> pd.DataFrame:
    history_raw_df = prepare_raw_frame(
        history_df,
        timestamp_col=timestamp_col,
        step_hours=bundle.step_hours,
    )
    future_raw_df = prepare_raw_frame(
        future_covariates_df,
        timestamp_col=timestamp_col,
        step_hours=bundle.step_hours,
    )

    _validate_future_columns(bundle, future_raw_df)

    if "PM25" not in future_raw_df.columns:
        future_raw_df["PM25"] = np.nan

    future_raw_df = future_raw_df[future_raw_df.index > history_raw_df.index.max()].copy()
    rollout_horizon = int(horizon or bundle.rollout_horizon)
    future_raw_df = future_raw_df.iloc[:rollout_horizon]

    if future_raw_df.empty:
        raise ValueError("Future covariates do not contain rows after the history window.")

    rows: list[dict[str, Any]] = []
    recursive_history = history_raw_df.copy()

    for chunk_start in range(0, len(future_raw_df), bundle.chunk_horizon):
        chunk_future = future_raw_df.iloc[chunk_start : chunk_start + bundle.chunk_horizon].copy()
        effective_chunk_horizon = len(chunk_future)
        if effective_chunk_horizon == 0:
            break

        x_input, last_target_scaled = build_inference_inputs(
            recursive_history,
            feature_cols=bundle.feature_cols,
            x_scaler=bundle.x_scaler,
            y_scaler=bundle.y_scaler,
            lookback=bundle.lookback,
            target_mode=bundle.target_mode,
        )

        decoder_input = make_decoder_input(
            bundle,
            chunk_future,
            last_target_scaled=last_target_scaled,
        )

        if bundle.decoder_future_cols:
            y_pred_scaled = bundle.model.predict([x_input, decoder_input], verbose=0)[0][
                :effective_chunk_horizon
            ].astype(np.float32)
        else:
            y_pred_scaled = np.zeros((bundle.chunk_horizon,), dtype=np.float32)
            for step_idx in range(effective_chunk_horizon):
                decoder_forecast = bundle.model.predict([x_input, decoder_input], verbose=0)[0]
                y_pred_scaled[step_idx] = decoder_forecast[step_idx]
                if step_idx + 1 < bundle.chunk_horizon:
                    decoder_input[0, step_idx + 1, 0] = y_pred_scaled[step_idx]

        y_pred = inverse_target(
            y_pred_scaled[:effective_chunk_horizon],
            bundle.y_scaler,
            mode=bundle.target_mode,
        )

        for step_idx, timestamp in enumerate(chunk_future.index, start=1):
            row = {
                timestamp_col: timestamp,
                "step": chunk_start + step_idx,
                "y_pred": float(y_pred[step_idx - 1]),
                "model_name": bundle.model_name,
                "bundle_key": bundle.bundle_key,
            }

            y_true_value = chunk_future["PM25"].iloc[step_idx - 1]
            if pd.notna(y_true_value):
                row["y_true"] = float(y_true_value)

            rows.append(row)

        recursive_chunk = chunk_future.copy()
        recursive_chunk.loc[:, "PM25"] = np.asarray(y_pred, dtype=np.float64)
        recursive_history = pd.concat([recursive_history, recursive_chunk], axis=0)

    forecast_df = pd.DataFrame(rows).sort_values(timestamp_col).reset_index(drop=True)
    if forecast_df.empty:
        raise ValueError("Inference completed without producing any forecast rows.")
    return forecast_df


def load_best_bundle(bundle_dir: str | Path = DEFAULT_BUNDLE_DIR) -> LoadedBundle:
    return load_model_bundle(bundle_dir)
