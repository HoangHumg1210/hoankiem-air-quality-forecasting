import random
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================
# CONFIG
# =========================
@dataclass
class DataConfig:
    data_path: str = "data/processed/data2225_done.csv"
    target: str = "PM25"
    time_col: str = "Local Time"

    lookback: int = 336
    horizon: int = 72

    target_transform: str = "log"   # "log", "sqrt", "none"

    train_end: str = "2023-12-31 23:00:00"
    val_start: str = "2024-01-01 00:00:00"
    val_end: str = "2024-12-31 23:00:00"
    test_start: str = "2025-01-01 00:00:00"

    freq: str = "1h"
    seed: int = 42


CFG = DataConfig()


# =========================
# REPRODUCIBILITY
# =========================
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


# =========================
# TARGET TRANSFORM
# =========================
class TargetTransformer:
    def __init__(self, mode: str = "log"):
        mode = mode.lower()
        if mode not in {"log", "sqrt", "none"}:
            raise ValueError(f"Unknown target transform mode: {mode}")
        self.mode = mode
        self.scaler = StandardScaler()

    def _validate_non_negative(self, y: np.ndarray) -> None:
        if np.any(y < 0):
            raise ValueError(
                f"Target transform '{self.mode}' does not support negative values."
            )

    def _forward(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)

        if self.mode == "log":
            self._validate_non_negative(y)
            return np.log1p(y)
        if self.mode == "sqrt":
            self._validate_non_negative(y)
            return np.sqrt(y)
        return y

    def _inverse(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=np.float64)

        if self.mode == "log":
            return np.expm1(y)
        if self.mode == "sqrt":
            return np.square(np.clip(y, 0, None))
        return y

    def fit(self, y: np.ndarray) -> "TargetTransformer":
        y_t = self._forward(y)
        self.scaler.fit(y_t)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        y_t = self._forward(y)
        return self.scaler.transform(y_t)

    def fit_transform(self, y: np.ndarray) -> np.ndarray:
        y_t = self._forward(y)
        return self.scaler.fit_transform(y_t)

    def inverse_transform(self, y_scaled: np.ndarray) -> np.ndarray:
        y_scaled = np.asarray(y_scaled, dtype=np.float64)
        original_shape = y_scaled.shape
        y_unscaled = self.scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(
            original_shape
        )
        return self._inverse(y_unscaled)


# =========================
# FEATURE PREPROCESSOR
# =========================
class FeaturePreprocessor:
    def __init__(self):
        self.num_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.transformer: Optional[ColumnTransformer] = None

    def fit(self, df: pd.DataFrame, target: str) -> "FeaturePreprocessor":
        X_df = df.drop(columns=[target]).copy()

        self.num_cols = X_df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        self.cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()

        self.transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), self.num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.cat_cols),
            ],
            remainder="drop",
        )
        self.transformer.fit(X_df)
        return self

    def transform(self, df: pd.DataFrame, target: str) -> np.ndarray:
        if self.transformer is None:
            raise ValueError("Preprocessor chưa được fit.")

        X_df = df.drop(columns=[target]).copy()
        X = self.transformer.transform(X_df)
        return np.asarray(X, dtype=np.float32)

    def fit_transform(self, df: pd.DataFrame, target: str) -> np.ndarray:
        self.fit(df, target)
        return self.transform(df, target)


# =========================
# DATA LOADING & CLEANING
# =========================
def load_and_clean_data(path: str, time_col: str, target: str, freq: str = "1h") -> pd.DataFrame:
    df = pd.read_csv(path)

    if time_col not in df.columns:
        raise KeyError(f"Không tìm thấy cột thời gian '{time_col}'.")
    if target not in df.columns:
        raise KeyError(f"Không tìm thấy cột target '{target}'.")

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    df = df.set_index(time_col).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = df.asfreq(freq)
    return df


def _clean_split_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        df[num_cols] = df[num_cols].interpolate(method="time").ffill().bfill()

    if "IsHoliday" in df.columns:
        holiday = pd.to_numeric(df["IsHoliday"], errors="coerce").ffill().bfill().fillna(0)
        df["IsHoliday"] = holiday.astype(int)

    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in obj_cols:
        df[col] = df[col].ffill().bfill()

    return df


def _build_context_frame(history_df: pd.DataFrame, current_df: pd.DataFrame, history_len: int) -> pd.DataFrame:
    if history_df.empty:
        return current_df.copy()

    context_df = history_df.tail(history_len)
    return pd.concat([context_df, current_df], axis=0)


def _process_frame(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = _clean_split_frame(df)
    df = add_time_features(df)
    return add_target_features(df, target=target)


def _slice_processed_split(
    df: pd.DataFrame,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    if start is None and end is None:
        return df.copy()

    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None

    mask = pd.Series(True, index=df.index)
    if start_ts is not None:
        mask &= df.index >= start_ts
    if end_ts is not None:
        mask &= df.index <= end_ts
    return df.loc[mask].copy()


def _filter_sequences_by_time(
    X_seq: np.ndarray,
    y_seq: np.ndarray,
    forecast_times: np.ndarray,
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    if len(forecast_times) == 0:
        return X_seq, y_seq, forecast_times

    forecast_index = pd.DatetimeIndex(forecast_times)
    mask = np.ones(len(forecast_index), dtype=bool)

    if start is not None:
        mask &= forecast_index >= pd.Timestamp(start)
    if end is not None:
        mask &= forecast_index <= pd.Timestamp(end)

    return X_seq[mask], y_seq[mask], np.asarray(forecast_index[mask])


# =========================
# FEATURE ENGINEERING
# =========================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    idx = df.index

    df["day_of_week"] = idx.dayofweek
    df["month"] = idx.month
    df["hour"] = idx.hour
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)

    df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)

    df["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * idx.month / 12)

    return df


def add_target_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = df.copy()

    for lag in [1, 3, 6, 12, 24, 48]:
        df[f"{target}_lag{lag}"] = df[target].shift(lag)

    # Dùng shift(1) để tránh leakage.
    df[f"{target}_roll_mean_24"] = df[target].shift(1).rolling(24).mean()
    df[f"{target}_roll_std_24"] = df[target].shift(1).rolling(24).std()
    df[f"{target}_roll_mean_72"] = df[target].shift(1).rolling(72).mean()

    return df.dropna()


# =========================
# SPLIT
# =========================
def split_data(df: pd.DataFrame, cfg: DataConfig):
    train_df = df.loc[: cfg.train_end].copy()
    val_df = df.loc[cfg.val_start: cfg.val_end].copy()
    test_df = df.loc[cfg.test_start:].copy()

    if train_df.empty:
        raise ValueError("Train set rỗng.")
    if val_df.empty:
        raise ValueError("Validation set rỗng.")
    if test_df.empty:
        print("Cảnh báo: Test set rỗng. Kiểm tra dữ liệu có giai đoạn test hay không.")

    return train_df, val_df, test_df


# =========================
# SEQUENCE BUILDING
# =========================
def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    time_index: Optional[pd.Index] = None,
    lookback: int = 336,
    horizon: int = 72,
):
    """
    X_seq shape: (N, lookback, n_features)
    y_seq shape: (N, horizon)
    forecast_start_times shape: (N,) nếu time_index != None
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    y_1d = y.reshape(-1)

    X_seq, y_seq = [], []
    forecast_start_times = []

    max_start = len(X) - lookback - horizon + 1
    if max_start <= 0:
        empty_X = np.empty((0, lookback, X.shape[1]), dtype=np.float32)
        empty_y = np.empty((0, horizon), dtype=np.float32)

        if time_index is not None:
            empty_t = np.array([], dtype="datetime64[ns]")
            return empty_X, empty_y, empty_t
        return empty_X, empty_y

    for i in range(max_start):
        X_seq.append(X[i:i + lookback])
        y_seq.append(y_1d[i + lookback:i + lookback + horizon])

        if time_index is not None:
            forecast_start_times.append(time_index[i + lookback])

    X_seq = np.asarray(X_seq, dtype=np.float32)
    y_seq = np.asarray(y_seq, dtype=np.float32)

    if time_index is not None:
        return X_seq, y_seq, np.asarray(forecast_start_times)

    return X_seq, y_seq


# =========================
# EVALUATION
# =========================
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_regression(y_true, y_pred, name="Set") -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = float(mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1)))
    mse = float(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1)))
    rmse_val = float(np.sqrt(mse))

    print(f"{name} MAE : {mae:.4f}")
    print(f"{name} MSE : {mse:.4f}")
    print(f"{name} RMSE: {rmse_val:.4f}")

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse_val,
    }


def evaluate_by_horizon(y_true, y_pred) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("y_true và y_pred phải có shape (N, horizon).")

    rows = []
    for h in range(y_true.shape[1]):
        mae_h = mean_absolute_error(y_true[:, h], y_pred[:, h])
        rmse_h = np.sqrt(mean_squared_error(y_true[:, h], y_pred[:, h]))
        rows.append({
            "horizon_step": h + 1,
            "MAE": mae_h,
            "RMSE": rmse_h,
        })

    return pd.DataFrame(rows)


# =========================
# FULL PREP PIPELINE
# =========================
def _prepare_dataset_from_raw_splits(
    raw_train_df: pd.DataFrame,
    raw_val_df: pd.DataFrame,
    raw_test_df: pd.DataFrame,
    cfg: DataConfig,
    verbose: bool = True,
):
    target_feature_history = 72
    history_len = cfg.lookback + target_feature_history

    train_processed = _process_frame(raw_train_df, target=cfg.target)
    val_context_raw = _build_context_frame(raw_train_df, raw_val_df, history_len=history_len)
    val_processed_full = _process_frame(val_context_raw, target=cfg.target)
    test_history_raw = pd.concat([raw_train_df, raw_val_df], axis=0)
    test_context_raw = _build_context_frame(test_history_raw, raw_test_df, history_len=history_len)
    test_processed_full = _process_frame(test_context_raw, target=cfg.target)

    train_df = train_processed.copy()
    val_df = _slice_processed_split(val_processed_full, start=cfg.val_start, end=cfg.val_end)
    test_df = _slice_processed_split(test_processed_full, start=cfg.test_start)
    df = pd.concat([train_df, val_df, test_df], axis=0).sort_index()

    if verbose:
        print("After feature engineering:", df.shape)
        print("Train:", train_df.shape)
        print("Val  :", val_df.shape)
        print("Test :", test_df.shape)

    target_transformer = TargetTransformer(mode=cfg.target_transform)

    y_train = target_transformer.fit_transform(train_df[[cfg.target]].values)
    y_val = target_transformer.transform(val_df[[cfg.target]].values)
    y_test = (
        target_transformer.transform(test_df[[cfg.target]].values)
        if not test_df.empty
        else np.empty((0, 1))
    )

    preprocessor = FeaturePreprocessor()
    X_train = preprocessor.fit_transform(train_df, target=cfg.target)
    X_val = preprocessor.transform(val_df, target=cfg.target)
    X_test = (
        preprocessor.transform(test_df, target=cfg.target)
        if not test_df.empty
        else np.empty((0, X_train.shape[1]), dtype=np.float32)
    )

    if verbose:
        print("X_train:", X_train.shape, "y_train:", y_train.shape)
        print("X_val  :", X_val.shape, "y_val  :", y_val.shape)
        print("X_test :", X_test.shape, "y_test :", y_test.shape)

    X_train_seq, y_train_seq, train_times = create_sequences(
        X_train,
        y_train,
        time_index=train_df.index,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
    )

    X_val_full = preprocessor.transform(val_processed_full, target=cfg.target)
    y_val_full = target_transformer.transform(val_processed_full[[cfg.target]].values)
    X_val_seq, y_val_seq, val_times = create_sequences(
        X_val_full,
        y_val_full,
        time_index=val_processed_full.index,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
    )
    X_val_seq, y_val_seq, val_times = _filter_sequences_by_time(
        X_val_seq,
        y_val_seq,
        val_times,
        start=cfg.val_start,
        end=cfg.val_end,
    )

    X_test_full = (
        preprocessor.transform(test_processed_full, target=cfg.target)
        if not test_processed_full.empty
        else np.empty((0, X_train.shape[1]), dtype=np.float32)
    )
    y_test_full = (
        target_transformer.transform(test_processed_full[[cfg.target]].values)
        if not test_processed_full.empty
        else np.empty((0, 1))
    )
    X_test_seq, y_test_seq, test_times = create_sequences(
        X_test_full,
        y_test_full,
        time_index=test_processed_full.index if not test_processed_full.empty else pd.Index([]),
        lookback=cfg.lookback,
        horizon=cfg.horizon,
    )
    X_test_seq, y_test_seq, test_times = _filter_sequences_by_time(
        X_test_seq,
        y_test_seq,
        test_times,
        start=cfg.test_start,
    )

    if verbose:
        print("Sequence shapes:")
        print("Train:", X_train_seq.shape, y_train_seq.shape)
        print("Val  :", X_val_seq.shape, y_val_seq.shape)
        print("Test :", X_test_seq.shape, y_test_seq.shape)

    if len(X_train_seq) == 0:
        raise ValueError("Không tạo được sequence cho train. Kiểm tra lookback/horizon hoặc dữ liệu.")
    if len(X_val_seq) == 0:
        raise ValueError("Không tạo được sequence cho val. Kiểm tra lookback/horizon hoặc dữ liệu.")

    n_features = X_train_seq.shape[2]
    if verbose:
        print("n_features =", n_features)

    return {
        "cfg": cfg,
        "df": df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_seq": X_train_seq,
        "y_train_seq": y_train_seq,
        "X_val_seq": X_val_seq,
        "y_val_seq": y_val_seq,
        "X_test_seq": X_test_seq,
        "y_test_seq": y_test_seq,
        "train_times": train_times,
        "val_times": val_times,
        "test_times": test_times,
        "target_transformer": target_transformer,
        "preprocessor": preprocessor,
        "n_features": n_features,
    }


def prepare_dataset(cfg: DataConfig):
    df = load_and_clean_data(
        path=cfg.data_path,
        time_col=cfg.time_col,
        target=cfg.target,
        freq=cfg.freq,
    )
    print("Loaded df:", df.shape)

    raw_train_df, raw_val_df, raw_test_df = split_data(df, cfg)
    return _prepare_dataset_from_raw_splits(
        raw_train_df=raw_train_df,
        raw_val_df=raw_val_df,
        raw_test_df=raw_test_df,
        cfg=cfg,
        verbose=True,
    )


def prepare_walk_forward_datasets(
    cfg: DataConfig,
    eval_size: Optional[int] = None,
    step_size: Optional[int] = None,
    max_folds: Optional[int] = None,
    expanding: bool = True,
    train_size: Optional[int] = None,
):
    df = load_and_clean_data(
        path=cfg.data_path,
        time_col=cfg.time_col,
        target=cfg.target,
        freq=cfg.freq,
    )
    print("Loaded df:", df.shape)

    raw_train_df, raw_val_df, raw_test_df = split_data(df, cfg)
    eval_pool = pd.concat([raw_val_df, raw_test_df], axis=0)
    if eval_pool.empty:
        raise ValueError("Không có dữ liệu tương lai để chạy walk-forward.")

    if eval_size is None:
        eval_size = len(raw_val_df)
    if step_size is None:
        step_size = eval_size

    if eval_size <= 0:
        raise ValueError("eval_size phải > 0.")
    if step_size <= 0:
        raise ValueError("step_size phải > 0.")
    if max_folds is not None and max_folds <= 0:
        raise ValueError("max_folds phải > 0 nếu được truyền.")

    base_train_size = len(raw_train_df) if train_size is None else train_size
    min_train_rows = 72 + cfg.lookback + cfg.horizon
    if not expanding and base_train_size <= 0:
        raise ValueError("train_size phải > 0 khi expanding=False.")
    if not expanding and base_train_size < min_train_rows:
        raise ValueError(
            f"train_size phải >= {min_train_rows} để đủ lag/lookback/horizon cho walk-forward."
        )

    freq_offset = pd.tseries.frequencies.to_offset(cfg.freq)
    empty_future_df = raw_train_df.iloc[0:0].copy()
    folds = []

    fold_idx = 0
    start_idx = 0
    while start_idx < len(eval_pool):
        if max_folds is not None and fold_idx >= max_folds:
            break

        end_idx = min(start_idx + eval_size, len(eval_pool))
        current_val_raw = eval_pool.iloc[start_idx:end_idx].copy()
        if len(current_val_raw) < cfg.horizon:
            break

        history_before_fold = pd.concat([raw_train_df, eval_pool.iloc[:start_idx]], axis=0)
        if expanding:
            current_train_raw = history_before_fold
        else:
            current_train_raw = history_before_fold.tail(base_train_size)
        if len(current_train_raw) < min_train_rows:
            raise ValueError(
                f"Fold {fold_idx + 1} không đủ lịch sử train. Cần ít nhất {min_train_rows} dòng raw."
            )

        fold_val_start = current_val_raw.index.min()
        fold_val_end = current_val_raw.index.max()
        fold_test_start = fold_val_end + freq_offset
        fold_cfg = replace(
            cfg,
            train_end=str(current_train_raw.index.max()),
            val_start=str(fold_val_start),
            val_end=str(fold_val_end),
            test_start=str(fold_test_start),
        )

        artifacts = _prepare_dataset_from_raw_splits(
            raw_train_df=current_train_raw,
            raw_val_df=current_val_raw,
            raw_test_df=empty_future_df,
            cfg=fold_cfg,
            verbose=False,
        )
        artifacts["fold"] = fold_idx + 1
        artifacts["walk_forward"] = {
            "fold": fold_idx + 1,
            "expanding": expanding,
            "eval_size": len(current_val_raw),
            "step_size": step_size,
            "train_size": len(current_train_raw),
            "train_end": current_train_raw.index.max(),
            "val_start": fold_val_start,
            "val_end": fold_val_end,
        }
        folds.append(artifacts)

        print(
            f"Walk-forward fold {fold_idx + 1}: "
            f"train_end={current_train_raw.index.max()} "
            f"val=({fold_val_start} -> {fold_val_end}) "
            f"seq={artifacts['X_val_seq'].shape[0]}"
        )

        fold_idx += 1
        start_idx += step_size

    if not folds:
        raise ValueError("Không tạo được fold walk-forward nào.")

    return folds


# =========================
# OPTIONAL: HELPER FOR MODEL PREDICTION
# =========================
def predict_and_inverse(model, X_seq: np.ndarray, target_transformer: TargetTransformer) -> np.ndarray:
    y_pred_scaled = model.predict(X_seq, verbose=0)
    return target_transformer.inverse_transform(y_pred_scaled)


def inverse_y(y_scaled: np.ndarray, target_transformer: TargetTransformer) -> np.ndarray:
    return target_transformer.inverse_transform(y_scaled)


# =========================
# MAIN
# =========================
def main():
    set_seed(CFG.seed)
    artifacts = prepare_dataset(CFG)
    return artifacts


if __name__ == "__main__":
    artifacts = main()
