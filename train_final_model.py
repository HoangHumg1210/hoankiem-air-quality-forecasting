"""train_final_model.py
======================
Deployment preparation script cho ứng dụng Streamlit dự báo PM2.5 Hà Nội.

Pipeline gồm 3 giai đoạn:
  1. Promote   – chọn bundle tốt nhất trong model_registry/ (theo MAE thấp nhất),
                 copy sang best_model_bundle/
  2. Retrain   – đọc toàn bộ data 2022-2025, build feature giống pipeline notebook,
                 tạo sequence, fit lại model trên full data
  3. Persist   – lưu model.keras, x_scaler.pkl, y_scaler.pkl, feature_cols.pkl,
                 cập nhật config.json và best_info.json

Chạy:
    python train_final_model.py [--app-dir PATH] [--bundle-key KEY]
                                [--warmup-epochs N] [--final-epochs N]
                                [--batch-size N] [--inner-val-steps N]
                                [--learning-rate F] [--loss mse|mae|huber|weighted_huber]
                                [--seed N] [--dry-run] [--skip-promote]
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import random
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hằng số – đồng bộ với notebook 03_model.ipynb
# ---------------------------------------------------------------------------

TIMESTAMP_COL = "Local Time"
STEP_HOURS = 3          # resample 3H
LOOKBACK = 112
CHUNK_HORIZON = 1
TARGET_MODE = "log1p"

# Lag list từ PRODUCTION_LAGS trong notebook
PRODUCTION_LAGS = [1, 8, 16, 24, 32, 40, 48, 56]

# Feature set "all" (37 features) – candidate_feature_sets["all"] trong notebook
ALL_FEATURE_COLS: list[str] = [
    "PM25_lag_1", "PM25_lag_8", "PM25_lag_24", "PM25_lag_56",
    "PM25_roll_mean_8", "PM25_roll_std_8", "PM25_roll_max_8", "PM25_roll_min_8",
    "PM25_roll_mean_24", "PM25_roll_std_24", "PM25_roll_max_24", "PM25_roll_min_24",
    "PM25_diff_1", "PM25_diff_8",
    "PM25_same_hour_mean_3d", "PM25_same_hour_mean_7d",
    "PM25_same_hour_std_7d", "PM25_same_hour_max_7d",
    "Temperature", "Pressure", "Wind Speed",
    "Clouds", "Precipitation", "Relative Humidity",
    "Accumulated Hours of Rain",
    "PM10", "CO", "NO2", "O3", "SO2",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "IsHoliday",
]

# Training defaults
DEFAULT_WARMUP_EPOCHS = 70
DEFAULT_INNER_VAL_STEPS = 56
DEFAULT_FINAL_EPOCH_FLOOR = 10
DEFAULT_FINAL_EPOCH_MULTIPLIER = 1.35
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_SEED = 62
DEPLOY_BUNDLE_NAME = "best_model_bundle"
CANDIDATE_BUNDLE_NAME = "candidate_model_bundle"

# Callback params từ notebook
EARLY_STOPPING_PATIENCE = 24
LR_REDUCE_PATIENCE = 8

# Peak loss params từ notebook
PEAK_QUANTILE = 0.90
PEAK_WEIGHT = 2.5
HUBER_DELTA = 1.0


# ===========================================================================
# PHẦN 1 – FEATURE ENGINEERING
# Tái tạo chính xác pipeline trong notebook: resample → build_history_feature_frame
# ===========================================================================

def resample_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Resample về 3H, IsHoliday dùng max, các cột còn lại dùng mean."""
    df = df.copy()
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.set_index(TIMESTAMP_COL).sort_index()

    agg_map = {col: "mean" for col in df.columns}
    if "IsHoliday" in agg_map:
        agg_map["IsHoliday"] = "max"

    return df.resample(f"{STEP_HOURS}h").agg(agg_map).dropna().copy()


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tái tạo build_history_feature_frame() từ notebook.
    Input: DataFrame đã resample, index là DatetimeIndex.
    Output: DataFrame bổ sung đầy đủ các cột feature.
    """
    df = df.copy().sort_index()

    # Cyclic calendar encoding
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]  = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]  = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # PM25 lag features
    for lag in PRODUCTION_LAGS:
        df[f"PM25_lag_{lag}"] = df["PM25"].shift(lag)

    # Rolling stats trên PM25.shift(1) – tránh data leakage
    shifted = df["PM25"].shift(1)
    for w in [8, 24]:
        df[f"PM25_roll_mean_{w}"] = shifted.rolling(w).mean()
        df[f"PM25_roll_std_{w}"]  = shifted.rolling(w).std()
        df[f"PM25_roll_max_{w}"]  = shifted.rolling(w).max()
        df[f"PM25_roll_min_{w}"]  = shifted.rolling(w).min()

    # Diff features
    df["PM25_diff_1"] = shifted.diff(1)
    df["PM25_diff_8"] = shifted.diff(8)

    # Same-hour aggregates (proxy cho chu kỳ ngày)
    same_3d = ["PM25_lag_8", "PM25_lag_16", "PM25_lag_24"]
    same_7d = ["PM25_lag_8",  "PM25_lag_16", "PM25_lag_24",
               "PM25_lag_32", "PM25_lag_40", "PM25_lag_48", "PM25_lag_56"]
    df["PM25_same_hour_mean_3d"] = df[same_3d].mean(axis=1)
    df["PM25_same_hour_mean_7d"] = df[same_7d].mean(axis=1)
    df["PM25_same_hour_std_7d"]  = df[same_7d].std(axis=1)
    df["PM25_same_hour_max_7d"]  = df[same_7d].max(axis=1)

    return df


def prepare_feature_frame(
    raw_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Resample → build feature → validate → dropna."""
    resampled = resample_raw(raw_df)
    featured  = build_feature_frame(resampled)

    required = feature_cols + ["PM25"]
    missing = [c for c in required if c not in featured.columns]
    if missing:
        raise ValueError(
            f"Feature frame thiếu cột (kiểm tra raw data): {missing}"
        )

    featured = featured.dropna(subset=required).copy()
    if featured.empty:
        raise ValueError(
            "Feature frame rỗng sau dropna. "
            "Kiểm tra phạm vi thời gian raw data và PRODUCTION_LAGS."
        )
    return featured


# ===========================================================================
# PHẦN 2 – SEQUENCE BUILDER
# Tái tạo make_sequences() trong notebook (hỗ trợ decoder_future)
# ===========================================================================

def make_sequences(
    x_scaled: np.ndarray,
    y_scaled: np.ndarray,
    lookback: int,
    horizon: int,
    decoder_future: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tạo (encoder_input, decoder_input, target) arrays.

    decoder_input[:, 0, 0] = PM25 scaled của bước cuối trước horizon.
    Nếu decoder_future được cung cấp (future covariates đã scaled),
    chúng được nối vào sau dim PM25 → decoder_width = 1 + n_extra_cols.
    """
    x_scaled = np.asarray(x_scaled, dtype=np.float32)
    y_scaled  = np.asarray(y_scaled, dtype=np.float32).reshape(-1)

    if len(x_scaled) != len(y_scaled):
        raise ValueError(
            f"x_scaled và y_scaled phải cùng độ dài: "
            f"len(x)={len(x_scaled)}, len(y)={len(y_scaled)}"
        )

    if decoder_future is not None:
        decoder_future = np.asarray(decoder_future, dtype=np.float32)
        if len(decoder_future) != len(x_scaled):
            raise ValueError(
                f"decoder_future phải cùng độ dài với x_scaled: "
                f"{len(decoder_future)} vs {len(x_scaled)}"
            )
        decoder_width = 1 + decoder_future.shape[1]
    else:
        decoder_width = 1

    max_start = len(x_scaled) - lookback - horizon + 1
    if max_start <= 0:
        return (
            np.empty((0, lookback, x_scaled.shape[1]), dtype=np.float32),
            np.empty((0, horizon, decoder_width), dtype=np.float32),
            np.empty((0, horizon), dtype=np.float32),
        )

    x_list, dec_list, y_list = [], [], []
    for i in range(max_start):
        x_list.append(x_scaled[i : i + lookback])

        last_y = y_scaled[i + lookback - 1]
        seed = np.full((horizon, 1), last_y, dtype=np.float32)

        if decoder_future is not None:
            future_cov = decoder_future[i + lookback : i + lookback + horizon]
            dec_input = np.concatenate([seed, future_cov], axis=1)
        else:
            dec_input = seed

        dec_list.append(dec_input)
        y_list.append(y_scaled[i + lookback : i + lookback + horizon])

    return (
        np.asarray(x_list, dtype=np.float32),
        np.asarray(dec_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32),
    )


# ===========================================================================
# PHẦN 3 – TARGET TRANSFORM
# Tái tạo transform_target() / inverse_target() trong notebook
# ===========================================================================

def transform_target(
    y_raw: np.ndarray,
    *,
    scaler: StandardScaler | None = None,
    fit: bool = False,
    mode: str = "log1p",
) -> tuple[np.ndarray, StandardScaler]:
    """log1p/sqrt/raw → StandardScaler."""
    y_raw = np.asarray(y_raw, dtype=np.float64).reshape(-1)
    y_clip = np.clip(y_raw, 0.0, None)

    if mode == "log1p":
        y_t = np.log1p(y_clip).reshape(-1, 1)
    elif mode == "sqrt":
        y_t = np.sqrt(y_clip).reshape(-1, 1)
    elif mode == "raw":
        y_t = y_clip.reshape(-1, 1)
    else:
        raise ValueError(f"mode không hợp lệ: {mode!r}")

    if fit:
        scaler = StandardScaler()
        scaler.fit(y_t)
    elif scaler is None:
        raise ValueError("Phải cung cấp scaler khi fit=False.")

    return scaler.transform(y_t).reshape(-1).astype(np.float32), scaler


def inverse_target(
    y_scaled: np.ndarray,
    scaler: StandardScaler,
    mode: str = "log1p",
) -> np.ndarray:
    """Inverse về PM25 gốc (µg/m³)."""
    y_scaled = np.asarray(y_scaled, dtype=np.float64)
    shape = y_scaled.shape
    y_us = scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)

    if mode == "log1p":
        y_raw = np.expm1(y_us)
    elif mode == "sqrt":
        y_raw = np.square(np.clip(y_us, 0.0, None))
    elif mode == "raw":
        y_raw = y_us
    else:
        raise ValueError(f"mode không hợp lệ: {mode!r}")

    return np.clip(y_raw, 0.0, None).reshape(shape)


# ===========================================================================
# PHẦN 4 – LOSS FUNCTION
# Tái tạo make_weighted_huber_loss() trong notebook
# ===========================================================================

def make_weighted_huber_loss(
    peak_threshold: float,
    peak_weight: float = PEAK_WEIGHT,
    delta: float = HUBER_DELTA,
    horizon: int = CHUNK_HORIZON,
):
    """
    Peak-weighted + step-weighted Huber loss.
    Điểm PM25 >= peak_threshold bị phạt nặng hơn peak_weight lần.
    Bước xa hơn trong horizon bị phạt nặng hơn (linspace 1.0 → 1.8).
    """
    import tensorflow as tf

    thr  = tf.constant(float(peak_threshold), dtype=tf.float32)
    pw   = tf.constant(float(peak_weight), dtype=tf.float32)
    dlt  = tf.constant(float(delta), dtype=tf.float32)
    sw   = tf.reshape(tf.linspace(1.0, 1.8, horizon), (1, horizon))
    sw   = sw / tf.reduce_mean(sw)

    def loss(y_true, y_pred):
        err = y_true - y_pred
        abs_err = tf.abs(err)
        huber = tf.where(
            abs_err <= dlt,
            0.5 * tf.square(err),
            dlt * (abs_err - 0.5 * dlt),
        )
        peak_mask = tf.cast(y_true >= thr, tf.float32)
        w = (1.0 + pw * peak_mask) * sw
        return tf.reduce_mean(huber * w)

    return loss


# ===========================================================================
# PHẦN 5 – PROMOTE
# Chọn bundle tốt nhất trong model_registry và copy sang best_model_bundle       
# ===========================================================================

def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("%s l?i JSON: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _read_bundle_info(bundle_dir: Path) -> dict[str, Any]:
    info: dict[str, Any] = {}
    for fname in ("metrics.json", "best_info.json", "config.json"):
        info.update(_read_json_file(bundle_dir / fname))
    return info


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _metric_from_info(info: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = _safe_float(info.get(key))
        if value is not None:
            return value
    return None


def _format_metric(value: float | None, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def _replace_bundle_dir(source_dir: Path, dest_dir: Path) -> None:
    """Copy source → dest theo cach atomic: copy vao .tmp truoc, roi rename.
    Neu crash giua chung, dest_dir goc van con nguyen (khong mat data).
    """
    tmp_dir = dest_dir.with_suffix(".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    shutil.copytree(source_dir, tmp_dir)
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    tmp_dir.rename(dest_dir)


def select_source_bundle_dir(
    app_dir: Path,
    bundle_key: str | None = None,
) -> Path:
    registry_dir = app_dir / "model_registry"
    if not registry_dir.exists():
        raise FileNotFoundError(
            f"Khong tim thay model_registry tai {registry_dir}. "
            "Chay notebook huan luyen truoc."
        )

    bundles = [p for p in registry_dir.iterdir() if p.is_dir()]
    if not bundles:
        raise ValueError(f"model_registry tai {registry_dir} khong co bundle nao.")

    if bundle_key is not None:
        chosen = registry_dir / bundle_key
        if not chosen.exists():
            raise ValueError(
                f"Bundle key '{bundle_key}' khong ton tai trong {registry_dir}. "
                f"Cac bundle hien co: {[b.name for b in bundles]}"
            )
        logger.info("Bundle duoc chi dinh thu cong: %s", bundle_key)
        return chosen

    def _get_mae(bundle_dir: Path) -> float:
        info = _read_bundle_info(bundle_dir)
        for key in ("val_mae", "mae", "test_mae"):
            if key in info:
                try:
                    return float(info[key])
                except (TypeError, ValueError):
                    pass
        return float("inf")

    chosen = min(bundles, key=_get_mae)
    mae_val = _get_mae(chosen)
    logger.info(
        "Bundle tot nhat trong registry: %s (MAE=%s)",
        chosen.name,
        _format_metric(None if mae_val == float("inf") else mae_val),
    )
    return chosen


def summarize_bundle(bundle_dir: Path) -> dict[str, Any]:
    info = _read_bundle_info(bundle_dir)
    feature_cols = info.get("feature_cols") or []
    decoder_future_cols = info.get("decoder_future_cols") or []

    if not isinstance(feature_cols, list):
        feature_cols = []
    if not isinstance(decoder_future_cols, list):
        decoder_future_cols = []

    return {
        "bundle_key": bundle_dir.name,
        "bundle_dir": str(bundle_dir),
        "model_name": info.get("model_name", bundle_dir.name),
        "mae": _metric_from_info(info, "val_mae", "mae", "test_mae"),
        "rmse": _metric_from_info(info, "rmse"),
        "mse": _metric_from_info(info, "mse"),
        "mape": _metric_from_info(info, "mape"),
        "peak_mae": _metric_from_info(info, "peak_mae"),
        "peak_threshold": _metric_from_info(info, "peak_threshold"),
        "lookback": info.get("lookback"),
        "chunk_horizon": info.get("chunk_horizon"),
        "rollout_horizon": info.get("rollout_horizon"),
        "best_epoch": info.get("best_epoch"),
        "final_epochs": info.get("final_epochs"),
        "target_mode": info.get("target_mode") or info.get("target_transform_mode"),
        "feature_count": len(feature_cols),
        "decoder_future_count": len(decoder_future_cols),
    }


def collect_bundle_summaries(registry_dir: Path) -> list[dict[str, Any]]:
    summaries = [summarize_bundle(p) for p in registry_dir.iterdir() if p.is_dir()]
    return sorted(
        summaries,
        key=lambda s: (
            s["mae"] if s["mae"] is not None else float("inf"),
            s["bundle_key"],
        ),
    )


def log_bundle_summary(title: str, summary: dict[str, Any]) -> None:
    logger.info("%s", title)
    logger.info("  Bundle key        : %s", summary["bundle_key"])
    logger.info("  Model             : %s", summary["model_name"])
    logger.info("  Bundle dir        : %s", summary["bundle_dir"])
    logger.info(
        "  Metrics           : MAE=%s | RMSE=%s | MSE=%s | MAPE=%s | Peak_MAE=%s",
        _format_metric(summary["mae"]),
        _format_metric(summary["rmse"]),
        _format_metric(summary["mse"]),
        _format_metric(summary["mape"]),
        _format_metric(summary["peak_mae"]),
    )
    logger.info(
        "  Shape             : lookback=%s | chunk_horizon=%s | rollout_horizon=%s",
        summary["lookback"],
        summary["chunk_horizon"],
        summary["rollout_horizon"],
    )
    logger.info(
        "  Features          : %s input features | %s decoder future features",
        summary["feature_count"],
        summary["decoder_future_count"],
    )
    logger.info(
        "  Training meta     : best_epoch=%s | final_epochs=%s | target_mode=%s | peak_threshold=%s",
        summary["best_epoch"],
        summary["final_epochs"],
        summary["target_mode"],
        _format_metric(summary["peak_threshold"]),
    )


def log_registry_ranking(app_dir: Path) -> None:
    registry_dir = app_dir / "model_registry"
    if not registry_dir.exists():
        logger.warning("Khong tim thay model_registry tai %s", registry_dir)
        return

    summaries = collect_bundle_summaries(registry_dir)
    if not summaries:
        logger.warning("model_registry tai %s khong co bundle nao.", registry_dir)
        return

    logger.info("XEP HANG BUNDLE TRONG model_registry (MAE tang dan)")
    for idx, summary in enumerate(summaries, start=1):
        logger.info(
            "  %d. %s | model=%s | MAE=%s | RMSE=%s | Peak_MAE=%s | best_epoch=%s | final_epochs=%s",
            idx,
            summary["bundle_key"],
            summary["model_name"],
            _format_metric(summary["mae"]),
            _format_metric(summary["rmse"]),
            _format_metric(summary["peak_mae"]),
            summary["best_epoch"],
            summary["final_epochs"],
        )





# ===========================================================================
# PHẦN 6 – LOAD BUNDLE
# ===========================================================================

def _pickle_load(path: Path) -> Any:
    with path.open("rb") as fh:
        return pickle.load(fh)


def _pickle_dump(path: Path, obj: Any) -> None:
    with path.open("wb") as fh:
        pickle.dump(obj, fh)


def load_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Đọc tất cả artefact từ bundle_dir."""
    import tensorflow as tf

    required = ["model.keras", "x_scaler.pkl", "feature_cols.pkl"]
    missing  = [f for f in required if not (bundle_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Bundle tại {bundle_dir} thiếu file: {missing}"
        )

    # compile=False để tránh conflict với loss custom
    model = tf.keras.models.load_model(bundle_dir / "model.keras", compile=False)

    config: dict[str, Any] = {}
    cfg_path = bundle_dir / "config.json"
    if cfg_path.exists():
        try:
            config = json.loads(cfg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("config.json lỗi JSON: %s – dùng config rỗng.", exc)

    feature_cols: list[str] = _pickle_load(bundle_dir / "feature_cols.pkl")
    x_scaler: StandardScaler = _pickle_load(bundle_dir / "x_scaler.pkl")

    y_scaler: StandardScaler | None = None
    ysp = bundle_dir / "y_scaler.pkl"
    if ysp.exists():
        y_scaler = _pickle_load(ysp)

    decoder_input_dim = int(model.inputs[1].shape[-1])

    bundle = {
        "model":             model,
        "config":            config,
        "feature_cols":      feature_cols,
        "x_scaler":          x_scaler,
        "y_scaler":          y_scaler,
        "lookback":          int(config.get("lookback", LOOKBACK)),
        "chunk_horizon":     int(config.get("chunk_horizon", CHUNK_HORIZON)),
        "step_hours":        int(config.get("step_hours", STEP_HOURS)),
        "target_mode":       str(config.get("target_mode", TARGET_MODE)),
        "model_name":        str(config.get("model_name", model.name)),
        "decoder_input_dim": decoder_input_dim,
    }

    logger.info(
        "Loaded: %s | lookback=%d | chunk_horizon=%d | "
        "n_features=%d | decoder_dim=%d | target_mode=%s",
        bundle["model_name"], bundle["lookback"], bundle["chunk_horizon"],
        len(feature_cols), decoder_input_dim, bundle["target_mode"],
    )
    return bundle


# ===========================================================================
# PHẦN 7 – MODEL UTILITIES
# ===========================================================================

def _compile_model(model, *, lr: float, loss_fn: Any):
    import tensorflow as tf
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=loss_fn,
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def _fresh_clone(source_model, *, lr: float, loss_fn: Any):
    """Clone kiến trúc + weights từ source, compile lại."""
    import tensorflow as tf
    cloned = tf.keras.models.clone_model(source_model)
    cloned.set_weights(source_model.get_weights())
    return _compile_model(cloned, lr=lr, loss_fn=loss_fn)


def _build_decoder_future_from_scaled(
    x_scaled: np.ndarray,
    feature_cols: list[str],
    decoder_input_dim: int,
) -> np.ndarray | None:
    if decoder_input_dim <= 1:
        return None

    decoder_future_cols = [
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "IsHoliday",
    ]
    decoder_future_idx = [feature_cols.index(col) for col in decoder_future_cols if col in feature_cols]
    expected_extra = decoder_input_dim - 1
    if len(decoder_future_idx) != expected_extra:
        raise ValueError(
            f"Decoder input dim khong khop: model can {expected_extra} future cols, "
            f"nhung chi tao duoc {len(decoder_future_idx)} tu feature_cols."
        )
    return x_scaled[:, decoder_future_idx]


def _compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    mse = float(np.mean(np.square(errors)))
    peak_threshold = float(np.quantile(y_true, PEAK_QUANTILE)) if y_true.size else 0.0
    peak_mask = y_true >= peak_threshold
    return {
        "mae": float(np.mean(abs_errors)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "peak_mae": float(np.mean(abs_errors[peak_mask])) if np.any(peak_mask) else float(np.mean(abs_errors)),
    }


def evaluate_bundle_on_holdout(
    *,
    model,
    feature_frame: pd.DataFrame,
    feature_cols: list[str],
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    target_mode: str,
    lookback: int,
    chunk_horizon: int,
    decoder_input_dim: int,
    eval_count: int,
) -> dict[str, Any]:
    x_values = feature_frame[feature_cols].to_numpy(dtype=np.float32)
    y_values = feature_frame["PM25"].to_numpy(dtype=np.float32)
    x_scaled = x_scaler.transform(x_values).astype(np.float32)
    y_scaled, _ = transform_target(y_values, scaler=y_scaler, fit=False, mode=target_mode)
    decoder_future = _build_decoder_future_from_scaled(x_scaled, feature_cols, decoder_input_dim)

    x_seq, dec_seq, y_seq = make_sequences(
        x_scaled,
        y_scaled,
        lookback=lookback,
        horizon=chunk_horizon,
        decoder_future=decoder_future,
    )
    if len(x_seq) == 0:
        raise ValueError("Khong tao duoc evaluation sequence cho holdout.")

    eval_count = min(eval_count, len(x_seq))
    y_pred_scaled = model.predict([x_seq[-eval_count:], dec_seq[-eval_count:]], verbose=0)
    y_true_raw = inverse_target(y_seq[-eval_count:], y_scaler, mode=target_mode)
    y_pred_raw = inverse_target(y_pred_scaled, y_scaler, mode=target_mode)

    metrics = _compute_regression_metrics(y_true_raw, y_pred_raw)
    metrics["eval_sequences"] = int(eval_count)
    metrics["eval_points"] = int(np.asarray(y_true_raw).size)
    return metrics


def resolve_final_epochs(
    best_epoch: int,
    *,
    config_final_epochs: int | None = None,
    floor: int = DEFAULT_FINAL_EPOCH_FLOOR,
    multiplier: float = DEFAULT_FINAL_EPOCH_MULTIPLIER,
    ceiling: int = DEFAULT_WARMUP_EPOCHS,
) -> int:
    """
    Tái tạo resolve_final_epochs() trong notebook:
    final = min(max(ceil(best_epoch * multiplier), floor), ceiling)
    """
    scaled = int(np.ceil(float(best_epoch) * multiplier))
    result = min(max(scaled, floor), ceiling)
    if config_final_epochs is not None:
        result = max(result, int(config_final_epochs))
    return result


# ===========================================================================
# PHẦN 8 – SAVE BUNDLE
# ===========================================================================

def save_bundle(
    bundle_dir: Path,
    *,
    model,
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    feature_cols: list[str],
    config: dict[str, Any],
    best_info: dict[str, Any],
) -> None:
    """Ghi toàn bộ artefact vào bundle_dir."""
    model.save(bundle_dir / "model.keras")
    _pickle_dump(bundle_dir / "x_scaler.pkl", x_scaler)
    _pickle_dump(bundle_dir / "y_scaler.pkl", y_scaler)
    _pickle_dump(bundle_dir / "feature_cols.pkl", feature_cols)
    (bundle_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (bundle_dir / "best_info.json").write_text(
        json.dumps(best_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Đã lưu bundle → %s", bundle_dir)


# ===========================================================================
# PHẦN 9 – CORE RETRAIN PIPELINE
# ===========================================================================

def run_final_training(
    *,
    bundle_dir: Path,
    raw_data_path: Path,
    comparison_bundle_dir: Path | None,
    warmup_epochs: int,
    final_epochs: int | None,
    batch_size: int,
    inner_val_steps: int,
    learning_rate: float,
    loss_type: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Pipeline retrain đầy đủ.

    Bước 1 – Load bundle đã promoted.
    Bước 2 – Load và preprocess raw data (resample + feature engineering).
    Bước 3 – Fit x_scaler, y_scaler mới trên toàn bộ data.
    Bước 4 – Tạo sequences (encoder_input, decoder_input, target).
    Bước 5 – Warmup fit với val split để tìm best_epoch.
    Bước 6 – Final fit trên 100% sequences.
    Bước 7 – Lưu tất cả artefact.
    """
    import tensorflow as tf

    # ------------------------------------------------------------------
    # Bước 1: Load bundle
    # ------------------------------------------------------------------
    logger.info("─" * 55)
    logger.info("Bước 1/6 – Load bundle từ %s", bundle_dir)
    bundle = load_bundle(bundle_dir)

    feature_cols      = bundle["feature_cols"] or list(ALL_FEATURE_COLS)
    lookback          = bundle["lookback"]
    chunk_horizon     = bundle["chunk_horizon"]
    target_mode       = bundle["target_mode"]
    decoder_input_dim = bundle["decoder_input_dim"]

    # ------------------------------------------------------------------
    # Bước 2: Load và preprocess data
    # ------------------------------------------------------------------
    logger.info("Bước 2/6 – Load raw data từ %s", raw_data_path)
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy raw data: {raw_data_path}")

    raw_df = pd.read_csv(raw_data_path)
    logger.info("Raw CSV: %d dòng, %d cột", len(raw_df), len(raw_df.columns))

    feature_frame = prepare_feature_frame(raw_df, feature_cols)
    logger.info(
        "Feature frame: %d dòng | %s → %s",
        len(feature_frame),
        feature_frame.index.min(),
        feature_frame.index.max(),
    )

    # ------------------------------------------------------------------
    # Bước 3: Fit scalers mới trên full data
    # ------------------------------------------------------------------
    logger.info("Bước 3/6 – Fit x_scaler và y_scaler trên full data")
    x_values = feature_frame[feature_cols].to_numpy(dtype=np.float32)
    y_values = feature_frame["PM25"].to_numpy(dtype=np.float32)

    x_scaler = StandardScaler()
    x_scaled = x_scaler.fit_transform(x_values).astype(np.float32)

    y_scaled, y_scaler = transform_target(y_values, fit=True, mode=target_mode)

    # ------------------------------------------------------------------
    # Bước 4: Tạo sequences
    # ------------------------------------------------------------------
    logger.info(
        "Bước 4/6 – Tạo sequences (lookback=%d, horizon=%d, decoder_dim=%d)",
        lookback, chunk_horizon, decoder_input_dim,
    )

    # Nếu bundle dùng decoder đa chiều (có DECODER_FUTURE_COLS),
    # các chiều phụ được pad bằng 0 vì không có future covariates khi retrain.
    decoder_future = _build_decoder_future_from_scaled(x_scaled, feature_cols, decoder_input_dim)
    if decoder_future is not None:
        logger.info(
            "decoder_input_dim=%d > 1 -> dùng %d decoder future covariates",
            decoder_input_dim, decoder_future.shape[1],
        )

    x_seq, dec_seq, y_seq = make_sequences(
        x_scaled, y_scaled,
        lookback=lookback,
        horizon=chunk_horizon,
        decoder_future=decoder_future,
    )

    if len(x_seq) == 0:
        raise ValueError(
            f"Không tạo được sequence. "
            f"Data length={len(x_scaled)} quá ngắn với lookback={lookback}, horizon={chunk_horizon}."
        )
    logger.info("Tạo được %d sequences (shape encoder: %s)", len(x_seq), x_seq.shape)

    # ------------------------------------------------------------------
    # Build loss function
    # ------------------------------------------------------------------
    if loss_type == "weighted_huber":
        peak_thr = float(np.quantile(y_seq.reshape(-1), PEAK_QUANTILE))
        loss_fn = make_weighted_huber_loss(
            peak_threshold=peak_thr,
            peak_weight=PEAK_WEIGHT,
            delta=HUBER_DELTA,
            horizon=chunk_horizon,
        )
        logger.info(
            "Loss: weighted_huber (peak_thr=%.4f, peak_weight=%.1f, delta=%.2f)",
            peak_thr, PEAK_WEIGHT, HUBER_DELTA,
        )
    elif loss_type == "huber":
        loss_fn = tf.keras.losses.Huber()
        logger.info("Loss: huber")
    else:
        loss_fn = loss_type  # "mse" hoặc "mae"
        logger.info("Loss: %s", loss_type)

    # ------------------------------------------------------------------
    # Bước 5: Warmup fit
    # ------------------------------------------------------------------
    val_count  = min(inner_val_steps, max(len(x_seq) // 10, 1))
    best_epoch = warmup_epochs  # fallback
    warmup_model = None
    gate_decision = {
        "accepted": True,
        "reason": "khong_can_so_sanh",
        "candidate_metrics": None,
        "current_metrics": None,
    }

    logger.info(
        "Bước 5/6 – Warmup fit (max=%d epochs, val_count=%d)",
        warmup_epochs, val_count,
    )

    if dry_run:
        logger.info("[DRY RUN] Bỏ qua warmup fit.")
    elif len(x_seq) <= val_count + 1:
        logger.warning(
            "Không đủ sequence để tách val (%d total, cần > %d). "
            "Dùng warmup_epochs=%d làm best_epoch.",
            len(x_seq), val_count + 1, warmup_epochs,
        )
    else:
        warmup_model = _fresh_clone(bundle["model"], lr=learning_rate, loss_fn=loss_fn)

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = str(Path(tmp) / "warmup_best.keras")
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=EARLY_STOPPING_PATIENCE,
                    restore_best_weights=True,
                    min_delta=1e-4,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=LR_REDUCE_PATIENCE,
                    min_lr=1e-6,
                    verbose=1,
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=ckpt,
                    monitor="val_loss",
                    save_best_only=True,
                    verbose=0,
                ),
            ]
            history = warmup_model.fit(
                [x_seq[:-val_count], dec_seq[:-val_count]],
                y_seq[:-val_count],
                validation_data=(
                    [x_seq[-val_count:], dec_seq[-val_count:]],
                    y_seq[-val_count:],
                ),
                epochs=warmup_epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1,
            )

        val_losses = history.history.get("val_loss") or history.history.get("loss") or []
        if val_losses:
            best_epoch = int(np.argmin(val_losses) + 1)
            logger.info(
                "Warmup xong – best_epoch=%d (val_loss=%.5f)",
                best_epoch, min(val_losses),
            )
        # Lưu ý: KHÔNG del warmup_model ở đây – biến này còn được dùng ở gate
        # evaluation (so sánh với current_best) và khởi tạo final_model.
        # GC sẽ tự giải phóng khi warmup_model được gán lại = None sau đó.

    # ------------------------------------------------------------------
    # Bước 6: Final fit trên 100% data
    # ------------------------------------------------------------------
    if not dry_run and warmup_model is not None and comparison_bundle_dir is not None and comparison_bundle_dir.exists():
        logger.info("Đánh giá candidate với best hiện tại trên holdout có định (%d sequences)", val_count)
        candidate_metrics = evaluate_bundle_on_holdout(
            model=warmup_model,
            feature_frame=feature_frame,
            feature_cols=feature_cols,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            target_mode=target_mode,
            lookback=lookback,
            chunk_horizon=chunk_horizon,
            decoder_input_dim=decoder_input_dim,
            eval_count=val_count,
        )

        current_bundle = load_bundle(comparison_bundle_dir)
        current_feature_cols = current_bundle["feature_cols"] or list(ALL_FEATURE_COLS)
        current_feature_frame = prepare_feature_frame(raw_df, current_feature_cols)
        if current_bundle["y_scaler"] is None:
            raise ValueError(f"Bundle hien tai tai {comparison_bundle_dir} thieu y_scaler.pkl.")

        current_metrics = evaluate_bundle_on_holdout(
            model=current_bundle["model"],
            feature_frame=current_feature_frame,
            feature_cols=current_feature_cols,
            x_scaler=current_bundle["x_scaler"],
            y_scaler=current_bundle["y_scaler"],
            target_mode=current_bundle["target_mode"],
            lookback=current_bundle["lookback"],
            chunk_horizon=current_bundle["chunk_horizon"],
            decoder_input_dim=current_bundle["decoder_input_dim"],
            eval_count=val_count,
        )

        accepted = float(candidate_metrics["mae"]) < float(current_metrics["mae"])
        gate_decision = {
            "accepted": accepted,
            "reason": "candidate_tot_hon" if accepted else "candidate_khong_tot_hon",
            "candidate_metrics": candidate_metrics,
            "current_metrics": current_metrics,
        }
        logger.info(
            "Promotion gate | current MAE=%s | candidate MAE=%s | accepted=%s",
            _format_metric(current_metrics["mae"]),
            _format_metric(candidate_metrics["mae"]),
            accepted,
        )

        if not accepted:
            logger.info("Không promote candidate vì MAE holdout không cải thiện.")
            return {
                "bundle_dir":          str(bundle_dir),
                "model_name":          bundle["model_name"],
                "lookback":            lookback,
                "chunk_horizon":       chunk_horizon,
                "step_hours":          bundle["step_hours"],
                "n_features":          len(feature_cols),
                "training_rows":       int(len(feature_frame)),
                "training_sequences":  int(len(x_seq)),
                "best_epoch":          int(best_epoch),
                "final_epochs":        0,
                "data_start":          str(feature_frame.index.min()),
                "data_end":            str(feature_frame.index.max()),
                "accepted":            False,
                "gate":                gate_decision,
            }

    if not dry_run and comparison_bundle_dir is not None and comparison_bundle_dir.exists() and warmup_model is None:
        gate_decision = {
            "accepted": False,
            "reason": "không_đủ_holdout_để_so_sánh",
            "candidate_metrics": None,
            "current_metrics": None,
        }
        logger.info("Không promote candidate vì không đủ holdout để so sánh với best hiện tại.")
        return {
            "bundle_dir":          str(bundle_dir),
            "model_name":          bundle["model_name"],
            "lookback":            lookback,
            "chunk_horizon":       chunk_horizon,
            "step_hours":          bundle["step_hours"],
            "n_features":          len(feature_cols),
            "training_rows":       int(len(feature_frame)),
            "training_sequences":  int(len(x_seq)),
            "best_epoch":          int(best_epoch),
            "final_epochs":        0,
            "data_start":          str(feature_frame.index.min()),
            "data_end":            str(feature_frame.index.max()),
            "accepted":            False,
            "gate":                gate_decision,
        }

    n_final = final_epochs if final_epochs is not None else resolve_final_epochs(
        best_epoch,
        config_final_epochs=int(bundle["config"].get("final_epochs", 0)) or None,
    )
    logger.info("Bước 6/6 – Final fit (%d epochs, 100%% data)", n_final)

    # Nếu warmup_model đã fit, dùng làm khởi điểm cho final fit (warm start).
    # Nếu không có (dry_run / không đủ seq), clone từ bundle gốc.
    # NOTE: Gate evaluation dùng warmup_model (fit trên n-val sequences),
    # còn final_model sẽ tiếp tục train trên 100% data. Model deploy
    # chưa từng được eval trực tiếp – đây là trade-off chấp nhận được vì
    # warmup là estimator thận trọng (ít data hơn, val split).
    final_model = warmup_model if warmup_model is not None else _fresh_clone(bundle["model"], lr=learning_rate, loss_fn=loss_fn)
    warmup_model = None  # giải phóng GPU memory

    if not dry_run:
        final_model.fit(
            [x_seq, dec_seq],
            y_seq,
            epochs=n_final,
            batch_size=batch_size,
            verbose=1,
        )
    else:
        logger.info("[DRY RUN] Bỏ qua final fit.")

    # ------------------------------------------------------------------
    # Lưu artefact
    # ------------------------------------------------------------------
    retrained_at = datetime.now(timezone.utc).isoformat()

    updated_config = dict(bundle["config"])
    updated_config.update({
        "model_name":               bundle["model_name"],
        "lookback":                 lookback,
        "chunk_horizon":            chunk_horizon,
        "step_hours":               bundle["step_hours"],
        "target_mode":              target_mode,
        "feature_cols":             feature_cols,
        "best_epoch":               int(best_epoch),
        "final_epochs":             int(n_final),
        "retrained_on_full_data":   True,
        "retrained_at_utc":         retrained_at,
        "n_full_training_rows":     int(len(feature_frame)),
        "n_full_training_sequences": int(len(x_seq)),
        "train_data_start":         str(feature_frame.index.min()),
        "train_data_end":           str(feature_frame.index.max()),
        "dry_run":                  dry_run,
    })

    # Giữ lại thông tin MAE từ notebook (không ghi đè)
    best_info: dict[str, Any] = {}
    bip = bundle_dir / "best_info.json"
    if bip.exists():
        try:
            best_info = json.loads(bip.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("best_info.json lỗi: %s – ghi đè.", exc)

    best_info.update({
        "retrained_on_full_data":    True,
        "retrained_at_utc":          retrained_at,
        "full_training_rows":        int(len(feature_frame)),
        "full_training_sequences":   int(len(x_seq)),
        "best_epoch":                int(best_epoch),
        "final_epochs":              int(n_final),
        "dry_run":                   dry_run,
        "promotion_gate":            gate_decision,
    })

    if not dry_run:
        save_bundle(
            bundle_dir,
            model=final_model,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            feature_cols=feature_cols,
            config=updated_config,
            best_info=best_info,
        )
    else:
        logger.info("[DRY RUN] Bỏ qua save_bundle.")

    return {
        "bundle_dir":          str(bundle_dir),
        "model_name":          bundle["model_name"],
        "lookback":            lookback,
        "chunk_horizon":       chunk_horizon,
        "step_hours":          bundle["step_hours"],
        "n_features":          len(feature_cols),
        "training_rows":       int(len(feature_frame)),
        "training_sequences":  int(len(x_seq)),
        "best_epoch":          int(best_epoch),
        "final_epochs":        int(n_final),
        "data_start":          str(feature_frame.index.min()),
        "data_end":            str(feature_frame.index.max()),
        "accepted":            True,
        "gate":                gate_decision,
    }


# ===========================================================================
# CLI & MAIN
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Deployment preparation: "
            "promote best model → retrain on full 2022-2025 data → save for Streamlit."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--app-dir",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Project root chứa data/, model_registry/, best_model_bundle/.",
    )
    p.add_argument(
        "--bundle-key",
        default=None,
        help="Promote bundle cụ thể; nếu bỏ trống → tự chọn bundle MAE thấp nhất.",
    )
    p.add_argument(
        "--warmup-epochs",
        type=int, default=DEFAULT_WARMUP_EPOCHS,
        help="Số epoch tối đa cho warmup fit (ước tính best_epoch).",
    )
    p.add_argument(
        "--final-epochs",
        type=int, default=None,
        help="Ghi đè số epoch final. Mặc định tự tính từ best_epoch.",
    )
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument(
        "--inner-val-steps",
        type=int, default=DEFAULT_INNER_VAL_STEPS,
        help="Số sequence cuối dùng làm validation trong warmup.",
    )
    p.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    p.add_argument(
        "--loss",
        default="weighted_huber",
        choices=["mse", "mae", "huber", "weighted_huber"],
        help=(
            "'weighted_huber' = loss từ notebook "
            "(peak-weighted + step-weighted Huber, khuyên dùng)."
        ),
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Chạy full pipeline nhưng không fit hay lưu file. Dùng để kiểm tra data.",
    )
    p.add_argument(
        "--skip-promote",
        action="store_true",
        help="Bỏ qua bước promote, dùng best_model_bundle hiện tại.",
    )
    return p.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Không bật được op determinism: %s", exc)


def main() -> None:
    args = parse_args()
    app_dir = args.app_dir.resolve()
    raw_data_path = app_dir / "data2225_done.csv"
    bundle_dir = app_dir / DEPLOY_BUNDLE_NAME
    candidate_dir = app_dir / CANDIDATE_BUNDLE_NAME

    logger.info("=" * 55)
    logger.info("PM2.5 FORECAST - DEPLOYMENT PREPARATION")
    logger.info("  app_dir    : %s", app_dir)
    logger.info("  raw data   : %s", raw_data_path)
    logger.info("  bundle dir : %s", bundle_dir)
    logger.info("  candidate  : %s", candidate_dir)
    logger.info("  seed       : %d", args.seed)
    logger.info("  loss       : %s", args.loss)
    if args.dry_run:
        logger.info("  [!] DRY RUN - không lưu file nào")
    logger.info("=" * 55)

    set_global_seed(args.seed)
    current_best_dir = bundle_dir if bundle_dir.exists() else None

    if args.skip_promote:
        if not bundle_dir.exists():
            raise FileNotFoundError(
                f"--skip-promote nhưng {bundle_dir} không tồn tại. "
                "Chạy trước không có --skip-promote."
            )
        source_bundle_dir = bundle_dir
        logger.info("--skip-promote: dùng bundle hiện tại tại %s", source_bundle_dir)
    else:
        source_bundle_dir = select_source_bundle_dir(app_dir, bundle_key=args.bundle_key)

    logger.info("Sao chép source bundle sang candidate: %s -> %s", source_bundle_dir, candidate_dir)
    _replace_bundle_dir(source_bundle_dir, candidate_dir)

    result = run_final_training(
        bundle_dir=candidate_dir,
        raw_data_path=raw_data_path,
        comparison_bundle_dir=current_best_dir,
        warmup_epochs=args.warmup_epochs,
        final_epochs=args.final_epochs,
        batch_size=args.batch_size,
        inner_val_steps=args.inner_val_steps,
        learning_rate=args.learning_rate,
        loss_type=args.loss,
        dry_run=args.dry_run,
    )

    promoted = False
    if not args.dry_run and result["accepted"]:
        logger.info("Candidate đạt điều kiện, promote vào %s", bundle_dir)
        _replace_bundle_dir(candidate_dir, bundle_dir)
        promoted = True
        # Dọn candidate_dir sau promote thành công để tránh chiếm dung lượng
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)
            logger.info("Đã xóa candidate_dir sau promote thành công: %s", candidate_dir)
    elif not args.dry_run:
        logger.info("Giữ nguyên %s. Candidate được lưu tại %s để xem lại.", bundle_dir, candidate_dir)

    logger.info("=" * 55)
    logger.info("HOAN TAT")
    logger.info("  Source bundle     : %s", source_bundle_dir.name)
    logger.info("  Candidate bundle  : %s", candidate_dir)
    logger.info("  Promoted          : %s", promoted)
    logger.info("  Model             : %s", result["model_name"])
    logger.info("  Bundle dir        : %s", result["bundle_dir"])
    logger.info("  Data range        : %s -> %s", result["data_start"], result["data_end"])
    logger.info("  Training rows     : %d", result["training_rows"])
    logger.info("  Training seqs     : %d", result["training_sequences"])
    logger.info("  n_features        : %d", result["n_features"])
    logger.info("  Lookback / Horizon: %d / %d", result["lookback"], result["chunk_horizon"])
    logger.info("  Step granularity  : %dh", result["step_hours"])
    logger.info("  best_epoch        : %d", result["best_epoch"])
    logger.info("  final_epochs      : %d", result["final_epochs"])
    gate = result.get("gate") or {}
    if gate.get("current_metrics") and gate.get("candidate_metrics"):
        logger.info(
            "  Holdout gate      : current MAE=%s | candidate MAE=%s | accepted=%s",
            _format_metric(gate["current_metrics"]["mae"]),
            _format_metric(gate["candidate_metrics"]["mae"]),
            result["accepted"],
        )
    if args.dry_run:
        logger.info("  [DRY RUN] Không có file nào được lưu")
    logger.info("=" * 55)


if __name__ == "__main__":
    main()