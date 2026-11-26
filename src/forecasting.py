"""Shared code for notebooks 07 (GRU), 08 (Transformer) and 09 (test comparison).

Both models use the same data, splits, scaling, evaluation samples, tuning budget
and training loop. Only the network architecture differs, so any gap between them
comes from the architecture and not from the pipeline.
"""
from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats

HORIZONS = [3, 6, 12, 24]
TARGET = 'PM25'
# Every model is scored on the same issue times: a sample is kept only if the last
# MAX_WINDOW hours are complete, whatever window the model actually uses.
MAX_WINDOW = 168

# Reanalysis inputs are published days late. In operation they are replaced by (see
# src/fetch_operational_met.py): ('forecast', column) = Open-Meteo forecast calibrated to the
# training scale on Validation; ('lag', hours) = the latest value that would already be published.
OPERATIONAL = {
    'Boundary Layer Height': ('forecast', 'blh_forecast'),
    'Wind Speed 10m': ('forecast', 'wind10_forecast'),
    'Soil Moisture': ('lag', 120),
}
OPERATIONAL_FILE = Path('data') / 'external' / 'operational' / 'openmeteo_forecast_hourly.csv'


# ----------------------------------------------------------------------------- setup

def find_root() -> Path:
    cwd = Path.cwd()
    return cwd.parent if cwd.name == 'notebooks' else cwd


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def enable_determinism() -> None:
    """Same seed -> same result, also on GPU (slightly slower)."""
    tf.config.experimental.enable_op_determinism()


def load_feature_config(root: Path) -> dict:
    path = root / 'results' / 'feature_selection' / 'feature_config.json'
    if not path.exists():
        raise FileNotFoundError(f'{path} không tồn tại. Hãy chạy 06_feature_selection.ipynb trước.')
    return json.loads(path.read_text(encoding='utf-8'))


def load_splits(root: Path) -> dict:
    sp = pd.read_csv(root / 'data' / 'processed' / 'split_dates.csv', parse_dates=['Start', 'End'])
    return {r.Split: (r.Start, r.End) for r in sp.itertuples()}


# ------------------------------------------------------------------------------ data

class WindowData:
    """Hourly feature matrix + the issue-time positions of each split.

    A sample at position p uses rows p-window+1 .. p as input and PM25 at p+h as target.
    It belongs to a split when its issue time and all its target times fall in that split.
    Inputs are standardised with Train statistics only.

    operational=True: every input hour used by a Test sample gets the real-time substitute of the
    reanalysis columns (OPERATIONAL) instead of the reanalysis value, i.e. what the model would
    really see online. Samples and scaling stay identical to operational=False.
    """

    def __init__(self, root: Path, features: list[str], horizons=HORIZONS, max_window=MAX_WINDOW,
                 operational=False):
        df = pd.read_csv(root / 'data' / 'processed' / 'feature_data.csv', parse_dates=['Local Time'])
        df = df.set_index('Local Time').sort_index()
        steps = df.index.to_series().diff().dropna()
        assert steps.eq(pd.Timedelta(hours=1)).all(), 'feature_data.csv phải là lưới giờ liên tục'
        missing = [f for f in features if f not in df.columns]
        assert not missing, f'Thiếu cột: {missing}'
        assert TARGET in features, 'PM25 phải nằm trong features (dùng cho persistence và target delta)'

        self.index = df.index
        self.features = list(features)
        self.H = np.asarray(horizons)
        self.splits = load_splits(root)
        self.raw = df[self.features].to_numpy(dtype='float32')
        self.pm = df[TARGET].to_numpy(dtype='float32')

        train_rows = self.index <= self.splits['Train'][1]
        self.x_mean = np.nanmean(self.raw[train_rows], axis=0)
        self.x_std = np.nanstd(self.raw[train_rows], axis=0)
        self.x_std[self.x_std < 1e-6] = 1.0
        scaled = (self.raw - self.x_mean) / self.x_std
        self.scaled = np.nan_to_num(scaled, nan=0.0).astype('float32')  # NaN rows are never sampled

        row_ok = np.isfinite(self.raw).all(axis=1) & np.isfinite(self.pm)
        window_ok = pd.Series(row_ok).rolling(max_window).sum().eq(max_window).to_numpy()
        hmax = int(self.H.max())
        pos = np.arange(max_window - 1, len(self.index) - hmax)
        pos = pos[window_ok[pos] & np.isfinite(self.pm[pos[:, None] + self.H]).all(axis=1)]

        self.positions = {}
        for name, (start, end) in self.splits.items():
            issue, last = self.index[pos], self.index[pos + hmax]
            self.positions[name] = pos[(issue >= start) & (last <= end)]

        self.high_threshold = float(np.nanquantile(self.pm[train_rows], 0.9))
        self.substitution = None
        if operational:
            self._apply_operational(root, max_window)

    def _apply_operational(self, root: Path, max_window):
        op = pd.read_csv(root / OPERATIONAL_FILE, parse_dates=['time_local'])
        op = op.set_index('time_local').reindex(self.index)
        swap = self.index >= self.splits['Test'][0] - pd.Timedelta(hours=max_window - 1)
        fit = (self.index >= self.splits['Validation'][0]) & ~swap   # calibrate on Validation only
        rows = []
        for col, (kind, arg) in OPERATIONAL.items():
            if col not in self.features:
                continue
            j = self.features.index(col)
            orig = self.raw[:, j].copy()
            if kind == 'forecast':
                src = op[arg].to_numpy(dtype='float64')
                ok = fit & np.isfinite(src) & np.isfinite(orig)
                b, a = np.polyfit(src[ok], orig[ok], 1)
                new = pd.Series(np.clip(a + b * src, 0, None)).ffill(limit=6).to_numpy()
                how = f'Open-Meteo forecast `{arg}`, hiệu chỉnh {a:.3g} + {b:.3g}·x (học trên {ok.sum()} giờ Validation)'
            else:
                new = pd.Series(orig).shift(arg).to_numpy()
                how = f'giá trị đã công bố gần nhất (trễ {arg} giờ)'
            assert np.isfinite(new[swap]).all(), f'Nguồn thay thế của {col} còn thiếu trong Test'
            rows.append({'Feature': col, 'Nguồn khi chạy thật': how,
                         'Tương quan với gốc (Test)': float(np.corrcoef(orig[swap], new[swap])[0, 1]),
                         'MAE so với gốc (Test)': float(np.mean(np.abs(orig[swap] - new[swap])))})
            self.raw[swap, j] = new[swap].astype('float32')
        self.substitution = pd.DataFrame(rows)
        scaled = (self.raw - self.x_mean) / self.x_std
        self.scaled = np.nan_to_num(scaled, nan=0.0).astype('float32')

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame([{'Split': k, 'N_samples': len(p),
                              'First_issue': self.index[p[0]], 'Last_issue': self.index[p[-1]]}
                             for k, p in self.positions.items()])

    def targets(self, pos):
        return self.pm[pos[:, None] + self.H]

    def base(self, pos, mode):
        """'delta': the model predicts the change from the current PM2.5."""
        if mode == 'delta':
            return np.repeat(self.pm[pos][:, None], len(self.H), axis=1)
        return np.zeros((len(pos), len(self.H)), dtype='float32')

    def target_stats(self, mode):
        pos = self.positions['Train']
        d = self.targets(pos) - self.base(pos, mode)
        return d.mean(axis=0).astype('float32'), d.std(axis=0).astype('float32')

    def scaled_targets(self, split, mode, mu, sd):
        pos = self.positions[split]
        return ((self.targets(pos) - self.base(pos, mode) - mu) / sd).astype('float32')

    def tf_dataset(self, pos, window, y=None, batch=256, shuffle=False, seed=0):
        values = tf.constant(self.scaled)
        offsets = tf.range(-window + 1, 1, dtype=tf.int32)
        if y is None:
            y = np.zeros((len(pos), len(self.H)), dtype='float32')
        ds = tf.data.Dataset.from_tensor_slices((pos.astype('int32'), y))
        if shuffle:
            ds = ds.shuffle(len(pos), seed=seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch).map(lambda p, t: (tf.gather(values, p[:, None] + offsets[None, :]), t),
                                 num_parallel_calls=tf.data.AUTOTUNE)
        return ds.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------- models

def _compile(model, lr):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr, clipnorm=1.0), loss='mae')
    return model


def build_gru(window, n_features, units=64, layers=1, dropout=0.1, dense=64, lr=1e-3):
    inp = tf.keras.Input(shape=(window, n_features))
    x = inp
    for i in range(layers):
        # No recurrent_dropout so the fast cuDNN kernel is used on GPU
        x = tf.keras.layers.GRU(units, return_sequences=i < layers - 1)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Dense(dense, activation='relu')(x)
    out = tf.keras.layers.Dense(len(HORIZONS))(x)
    return _compile(tf.keras.Model(inp, out, name='GRU'), lr)


@tf.keras.utils.register_keras_serializable(package='pm25')
class PositionEmbedding(tf.keras.layers.Layer):
    """Learned position vector added to each time step."""

    def build(self, input_shape):
        self.pos = self.add_weight(name='pos', shape=(input_shape[1], input_shape[2]),
                                   initializer=tf.keras.initializers.RandomNormal(stddev=0.02))

    def call(self, x):
        return x + self.pos[None, :, :]


def build_transformer(window, n_features, d_model=64, heads=4, layers=2, ff_mult=2,
                      dropout=0.1, dense=64, lr=1e-3):
    L = tf.keras.layers
    inp = tf.keras.Input(shape=(window, n_features))
    x = PositionEmbedding()(L.Dense(d_model)(inp))
    x = L.Dropout(dropout)(x)
    for _ in range(layers):  # pre-LayerNorm encoder block
        h = L.LayerNormalization(epsilon=1e-6)(x)
        h = L.MultiHeadAttention(num_heads=heads, key_dim=d_model // heads, dropout=dropout)(h, h)
        x = L.Add()([x, L.Dropout(dropout)(h)])
        h = L.LayerNormalization(epsilon=1e-6)(x)
        h = L.Dense(d_model * ff_mult, activation='gelu')(h)
        h = L.Dropout(dropout)(h)
        h = L.Dense(d_model)(h)
        x = L.Add()([x, L.Dropout(dropout)(h)])
    x = L.LayerNormalization(epsilon=1e-6)(x)
    last = L.Flatten()(L.Cropping1D((window - 1, 0))(x))  # most recent hour
    pooled = L.GlobalAveragePooling1D()(x)                 # whole window
    x = L.Dense(dense, activation='relu')(L.Concatenate()([last, pooled]))
    x = L.Dropout(dropout)(x)
    out = L.Dense(len(HORIZONS))(x)
    return _compile(tf.keras.Model(inp, out, name='Transformer'), lr)


BUILDERS = {'gru': build_gru, 'transformer': build_transformer}


# -------------------------------------------------------------------------- training

class KeepBest(tf.keras.callbacks.Callback):
    """Early stopping on val_loss that always restores the best epoch's weights."""

    def __init__(self, patience):
        super().__init__()
        self.patience = patience

    def on_train_begin(self, logs=None):
        self.best, self.best_epoch, self.wait, self.weights = np.inf, 0, 0, None

    def on_epoch_end(self, epoch, logs=None):
        v = logs['val_loss']
        if v < self.best:
            self.best, self.best_epoch, self.wait = v, epoch + 1, 0
            self.weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.weights is not None:
            self.model.set_weights(self.weights)


def split_config(cfg: dict):
    """A config holds the model params plus 'window' and 'target_mode'."""
    params = {k: v for k, v in cfg.items() if k not in ('window', 'target_mode')}
    return int(cfg['window']), cfg['target_mode'], params


def fit(kind, cfg, data: WindowData, seed, max_epochs=100, patience=10, batch=256, verbose=0):
    window, mode, params = split_config(cfg)
    tf.keras.backend.clear_session()
    set_seed(seed)
    mu, sd = data.target_stats(mode)
    train = data.tf_dataset(data.positions['Train'], window,
                            data.scaled_targets('Train', mode, mu, sd), batch, shuffle=True, seed=seed)
    val = data.tf_dataset(data.positions['Validation'], window,
                          data.scaled_targets('Validation', mode, mu, sd), batch)
    model = BUILDERS[kind](window, len(data.features), **params)
    keep = KeepBest(patience)
    lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                 patience=max(2, patience // 3), min_lr=1e-5)
    t0 = time.time()
    model.fit(train, validation_data=val, epochs=max_epochs, callbacks=[keep, lr_cb], verbose=verbose)
    info = {'best_epoch': keep.best_epoch, 'train_seconds': round(time.time() - t0, 1),
            'n_params': int(model.count_params()), 'target_mean': mu.tolist(), 'target_std': sd.tolist()}
    return model, info


def predict(model, cfg, info, data: WindowData, split, batch=1024):
    window, mode, _ = split_config(cfg)
    pos = data.positions[split]
    out = model.predict(data.tf_dataset(pos, window, batch=batch), verbose=0)
    out = out * np.asarray(info['target_std']) + np.asarray(info['target_mean']) + data.base(pos, mode)
    return np.clip(out, 0, None)


def free(model=None):
    del model
    tf.keras.backend.clear_session()
    gc.collect()


# ------------------------------------------------------------------------ evaluation

def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def metrics_table(y, p, data: WindowData, horizons=HORIZONS):
    rows = []
    for j, h in enumerate(list(horizons) + ['All']):
        yt = y if h == 'All' else y[:, j]
        pt = p if h == 'All' else p[:, j]
        err = yt - pt
        high = yt >= data.high_threshold
        rows.append({'Horizon': h if h == 'All' else f'{h}h',
                     'MAE': float(np.mean(np.abs(err))),
                     'RMSE': float(np.sqrt(np.mean(err ** 2))),
                     'R2': float(1 - np.sum(err ** 2) / np.sum((yt - yt.mean()) ** 2)),
                     'Bias': float(np.mean(pt - yt)),
                     'High_PM_MAE': float(np.mean(np.abs(err[high]))) if high.any() else np.nan})
    return pd.DataFrame(rows)


def baselines(data: WindowData, split):
    """Persistence (value now) and seasonal naive (same hour yesterday); both use only the past."""
    pos = data.positions[split]
    persistence = data.base(pos, 'delta')
    seasonal = data.pm[pos[:, None] + data.H - 24]
    return {'Persistence': persistence, 'Seasonal naive': seasonal}


def to_long(data: WindowData, split, pred, model_name):
    pos = data.positions[split]
    y = data.targets(pos)
    frames = []
    for j, h in enumerate(data.H):
        frames.append(pd.DataFrame({'issue_time': data.index[pos], 'horizon': int(h),
                                    'target_time': data.index[pos + h],
                                    'y_true': y[:, j], 'y_pred': pred[:, j], 'model': model_name}))
    return pd.concat(frames, ignore_index=True)


def diebold_mariano(err_a, err_b, h):
    """DM test on absolute errors with Newey-West variance (lag h-1).
    Negative statistic -> model A has lower error. Returns (stat, p_value)."""
    d = np.abs(err_a) - np.abs(err_b)
    n = len(d)
    dc = d - d.mean()
    var = np.sum(dc ** 2) / n
    for k in range(1, h):
        var += 2 * (1 - k / h) * np.sum(dc[k:] * dc[:-k]) / n
    if var <= 0:  # identical forecasts
        return 0.0, 1.0
    stat = d.mean() / np.sqrt(var / n)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


# ---------------------------------------------------------------------------- tuning

def sample_configs(space: dict, n: int, seed: int, always: list[dict] | None = None):
    """Random search: n distinct configs drawn from the grid, starting with `always`."""
    rng = np.random.default_rng(seed)
    keys = list(space)
    seen, configs = set(), []
    for cfg in (always or []):
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    total = int(np.prod([len(v) for v in space.values()]))
    while len(configs) < min(n, total):
        cfg = {k: space[k][rng.integers(len(space[k]))] for k in keys}
        cfg = {k: (v.item() if hasattr(v, 'item') else v) for k, v in cfg.items()}
        key = json.dumps(cfg, sort_keys=True)
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    return configs[:n]


def tune(kind, configs, data: WindowData, out_csv: Path, seed=42, **fit_kw):
    """Train each config once and score it on Validation. Results are appended to
    out_csv after every trial, so a disconnected Colab session can resume."""
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    done_keys = set(done['config']) if len(done) else set()
    rows = done.to_dict('records')
    y_val = data.targets(data.positions['Validation'])
    for i, cfg in enumerate(configs, 1):
        key = json.dumps(cfg, sort_keys=True)
        if key in done_keys:
            continue
        model, info = fit(kind, cfg, data, seed, **fit_kw)
        pred = predict(model, cfg, info, data, 'Validation')
        free(model)
        row = {'trial': i, 'config': key, **cfg, 'val_MAE': mae(y_val, pred),
               **{f'val_MAE_{h}h': mae(y_val[:, j], pred[:, j]) for j, h in enumerate(data.H)},
               'best_epoch': info['best_epoch'], 'n_params': info['n_params'],
               'train_seconds': info['train_seconds']}
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[{i}/{len(configs)}] val MAE {row['val_MAE']:.3f} | epoch {info['best_epoch']} "
              f"| {info['train_seconds']:.0f}s | {cfg}")
    return pd.DataFrame(rows).sort_values('val_MAE').reset_index(drop=True)


def needs_operational(features) -> bool:
    return any(f in OPERATIONAL for f in features)


def train_final(kind, cfg, data: WindowData, seeds, out_dir: Path, model_name, data_op=None, **fit_kw):
    """Train the chosen config with several seeds; the reported model is the seed average.
    data_op: WindowData(operational=True); adds Test predictions with real-time inputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = {'Validation': [], 'Test': []}
    op_preds = []
    seed_rows = []
    for seed in seeds:
        model, info = fit(kind, cfg, data, seed, **fit_kw)
        model.save(out_dir / f'model_seed{seed}.keras')
        (out_dir / f'model_seed{seed}_info.json').write_text(json.dumps(info, indent=2))
        for split in preds:
            p = predict(model, cfg, info, data, split)
            preds[split].append(p)
            seed_rows.append({'Seed': seed, 'Split': split, 'Best_epoch': info['best_epoch'],
                              'MAE': mae(data.targets(data.positions[split]), p)})
        if data_op is not None:
            p = predict(model, cfg, info, data_op, 'Test')
            op_preds.append(p)
            seed_rows.append({'Seed': seed, 'Split': 'Test (vận hành)', 'Best_epoch': info['best_epoch'],
                              'MAE': mae(data.targets(data.positions['Test']), p)})
        free(model)
        print(f'seed {seed}: xong')
    ensemble = {split: np.mean(v, axis=0) for split, v in preds.items()}
    for split, p in ensemble.items():
        to_long(data, split, p, model_name).to_csv(out_dir / f'predictions_{split.lower()}.csv', index=False)
    if data_op is not None:
        assert np.array_equal(data_op.positions['Test'], data.positions['Test'])
        ensemble['Test_operational'] = np.mean(op_preds, axis=0)
        to_long(data, 'Test', ensemble['Test_operational'], model_name).to_csv(
            out_dir / 'predictions_test_operational.csv', index=False)
        data_op.substitution.to_csv(out_dir / 'operational_substitution.csv', index=False)
    (out_dir / 'final_config.json').write_text(json.dumps(
        {'model': model_name, 'config': cfg, 'seeds': list(seeds), 'features': data.features,
         'horizons': data.H.tolist()}, indent=2, ensure_ascii=False), encoding='utf-8')
    return ensemble, pd.DataFrame(seed_rows)
