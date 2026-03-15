import pandas as pd
import numpy as np
import tensorflow as tf
import random

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error



TARGET = "PM25"
LOOKBACK = 336
HORIZON = 72

# Mặc định trước đây dùng log1p cho target.
# Giữ lại cờ USE_LOG_TARGET để tương thích ngược,
# nhưng ưu tiên dùng tham số transform trong hàm transform_target.
USE_LOG_TARGET = True

# Các lựa chọn transform cho target:
# - "log"  : log1p(x)
# - "sqrt" : sqrt(x)
# - "none" : giữ nguyên
TARGET_TRANSFORM = "log"

DATA_PATH = "data/processed/data2225_done.csv"



def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)



def load_and_clean_data(path):
    df = pd.read_csv(path)

    if "Local Time" not in df.columns:
        raise KeyError("Không tìm thấy cột 'Local Time' trong dữ liệu.")
    if TARGET not in df.columns:
        raise KeyError(f"Không tìm thấy cột target '{TARGET}' trong dữ liệu.")

    df["Local Time"] = pd.to_datetime(df["Local Time"])
    df = df.set_index("Local Time").sort_index()

    df = df[~df.index.duplicated(keep="last")]
    df = df.asfreq("1h")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 0:
        df[num_cols] = df[num_cols].interpolate(method="time").ffill().bfill()

    if "IsHoliday" in df.columns:
        df["IsHoliday"] = df["IsHoliday"].ffill().bfill().astype(int)

    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in obj_cols:
        df[col] = df[col].ffill().bfill()

    return df


# TIME FEATURES
def add_time_features(df):
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




def add_target_features(df, target=TARGET):
    for lag in [1, 3, 6, 12, 24, 48]:
        df[f"{target}_lag{lag}"] = df[target].shift(lag)

    # shift(1) để tránh leakage
    df[f"{target}_roll_mean_24"] = df[target].shift(1).rolling(24).mean()
    df[f"{target}_roll_std_24"] = df[target].shift(1).rolling(24).std()
    df[f"{target}_roll_mean_72"] = df[target].shift(1).rolling(72).mean()

    return df.dropna()



def split_data(df):
    train_df = df[: "2023-12-31 23:00:00"].copy()
    val_df = df["2024-01-01 00:00:00":"2024-12-31 23:00:00"].copy()
    test_df = df["2025-01-01 00:00:00":].copy()

    if len(train_df) == 0:
        raise ValueError("Train set rỗng.")
    if len(val_df) == 0:
        raise ValueError("Validation set rỗng.")
    if len(test_df) == 0:
        print("Cảnh báo: Test set rỗng. Kiểm tra dữ liệu có năm 2025 hay không.")

    return train_df, val_df, test_df



def transform_target(
    train_df,
    val_df,
    test_df,
    target=TARGET,
    transform: str | None = None,
    use_log: bool | None = None,
):
    """
    Biến đổi target theo nhiều lựa chọn khác nhau (log / sqrt / none)
    mà vẫn đảm bảo không rò rỉ dữ liệu (scaler fit trên train).

    Thứ tự ưu tiên:
    1) Nếu tham số `transform` được truyền (\"log\" / \"sqrt\" / \"none\"),
       sẽ dùng trực tiếp.
    2) Nếu `transform` = None nhưng `use_log` không None -> dùng use_log
       (True -> \"log\", False -> \"none\") để tương thích với code cũ.
    3) Nếu cả hai đều None -> dùng TARGET_TRANSFORM / USE_LOG_TARGET.
    """

    # Xác định chế độ transform
    if transform is not None:
        mode = transform.lower()
    elif use_log is not None:
        mode = "log" if use_log else "none"
    else:
        # Backward compatible: ưu tiên TARGET_TRANSFORM nếu hợp lệ,
        # nếu không thì quay về USE_LOG_TARGET.
        if str(TARGET_TRANSFORM).lower() in {"log", "sqrt", "none"}:
            mode = str(TARGET_TRANSFORM).lower()
        else:
            mode = "log" if USE_LOG_TARGET else "none"

    if mode not in {"log", "sqrt", "none"}:
        raise ValueError(f"Unknown target transform mode: {mode}")

    def forward(y):
        y = np.asarray(y, dtype=np.float64)
        y = np.clip(y, 0, None)
        if mode == "log":
            return np.log1p(y)
        if mode == "sqrt":
            return np.sqrt(y)
        return y

    def inverse(y):
        y = np.asarray(y, dtype=np.float64)
        if mode == "log":
            return np.expm1(y)
        if mode == "sqrt":
            # Đảm bảo không sinh giá trị âm do nhiễu số
            return np.square(np.clip(y, 0, None))
        return y

    y_train_raw = train_df[[target]].values
    y_val_raw = val_df[[target]].values
    y_test_raw = test_df[[target]].values if len(test_df) > 0 else np.empty((0, 1))

    y_train_t = forward(y_train_raw)
    y_val_t = forward(y_val_raw)
    y_test_t = forward(y_test_raw) if len(y_test_raw) > 0 else y_test_raw

    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train_t)
    y_val = scaler_y.transform(y_val_t)
    y_test = scaler_y.transform(y_test_t) if len(y_test_t) > 0 else y_test_t

    return y_train, y_val, y_test, scaler_y, inverse




def preprocess_features(train_df, val_df, test_df, target=TARGET):
    X_train_df = train_df.drop(columns=[target]).copy()
    X_val_df = val_df.drop(columns=[target]).copy()
    X_test_df = test_df.drop(columns=[target]).copy()

    num_cols = X_train_df.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = X_train_df.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        remainder="drop"
    )

    X_train = preprocess.fit_transform(X_train_df)
    X_val = preprocess.transform(X_val_df)
    X_test = preprocess.transform(X_test_df) if len(X_test_df) > 0 else np.empty((0, X_train.shape[1]))

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    return X_train, X_val, X_test, preprocess



def inverse_transform_target(y, scaler_y, inverse_func):
    y = np.asarray(y, dtype=np.float64)
    y_inv_scaled = scaler_y.inverse_transform(y)
    return inverse_func(y_inv_scaled)


def create_sequences_1(X, y, lookback=LOOKBACK, horizon=HORIZON):
    X = np.asarray(X)
    y = np.asarray(y)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # ép về 1 chiều để dễ cắt cửa sổ target
    y_1d = y.reshape(-1)

    X_seq, y_seq = [], []

    for i in range(len(X) - lookback - horizon + 1):
        X_seq.append(X[i : i + lookback])
        y_seq.append(y_1d[i + lookback : i + lookback + horizon])

    if len(X_seq) == 0:
        return (
            np.empty((0, lookback, X.shape[1]), dtype=np.float32),
            np.empty((0, horizon), dtype=np.float32),
        )

    return np.asarray(X_seq, dtype=np.float32), np.asarray(y_seq, dtype=np.float32)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_regression(y_true, y_pred, name="Set"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse_val = np.sqrt(mse)

    print(f"{name} MAE : {mae:.4f}")
    print(f"{name} MSE : {mse:.4f}")
    print(f"{name} RMSE: {rmse_val:.4f}")


def main():
    set_seed(42)

    # 1) Load + clean
    df = load_and_clean_data(DATA_PATH)
    print("Loaded df:", df.shape)

    # 2) Feature engineering
    df = add_time_features(df)
    df = add_target_features(df, target=TARGET)
    print("After feature engineering:", df.shape)

    # 3) Split
    train_df, val_df, test_df = split_data(df)
    print("Train:", train_df.shape)
    print("Val  :", val_df.shape)
    print("Test :", test_df.shape)

    # 4) Transform target
    y_train, y_val, y_test, scaler_y, inverse_target_func = transform_target(
        train_df, val_df, test_df, target=TARGET, use_log=USE_LOG_TARGET
    )

    # 5) Preprocess features
    X_train, X_val, X_test, preprocess = preprocess_features(
        train_df, val_df, test_df, target=TARGET
    )

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_val  :", X_val.shape, "y_val  :", y_val.shape)
    print("X_test :", X_test.shape, "y_test :", y_test.shape)

    # 6) Create sequences
    X_train_seq, y_train_seq = create_sequences_1(X_train, y_train, LOOKBACK, HORIZON)
    X_val_seq, y_val_seq = create_sequences_1(X_val, y_val, LOOKBACK, HORIZON)
    X_test_seq, y_test_seq = create_sequences_1(X_test, y_test, LOOKBACK, HORIZON)

    print("Sequence shapes:")
    print("Train:", X_train_seq.shape, y_train_seq.shape)
    print("Val  :", X_val_seq.shape, y_val_seq.shape)
    print("Test :", X_test_seq.shape, y_test_seq.shape)

    if len(X_train_seq) == 0:
        raise ValueError("Không tạo được sequence cho train. Kiểm tra LOOKBACK/HORIZON hoặc dữ liệu.")
    if len(X_val_seq) == 0:
        raise ValueError("Không tạo được sequence cho val. Kiểm tra LOOKBACK/HORIZON hoặc dữ liệu.")

    n_features = X_train_seq.shape[2]
    print("n_features =", n_features)

    return {
        "df": df,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "X_train_seq": X_train_seq,
        "y_train_seq": y_train_seq,
        "X_val_seq": X_val_seq,
        "y_val_seq": y_val_seq,
        "X_test_seq": X_test_seq,
        "y_test_seq": y_test_seq,
        "scaler_y": scaler_y,
        "inverse_target_func": inverse_target_func,
        "preprocess": preprocess,
        "n_features": n_features,
    }


if __name__ == "__main__":
    artifacts = main()