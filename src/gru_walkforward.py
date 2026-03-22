





def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred) * 100.0),
    }


def train_single_fold(
    fold_artifacts: dict[str, Any],
    seed: int = 42,
    epochs: int = 40,
    batch_size: int = 64,
    verbose: int = 0,
) -> dict[str, Any]:
    set_seed(seed)

    cfg: DataConfig = fold_artifacts["cfg"]
    n_features = int(fold_artifacts["n_features"])

    model = build_gru_model(
        lookback=cfg.lookback,
        n_features=n_features,
        horizon=cfg.horizon,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            min_delta=1e-4,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=0,
        ),
    ]

    history = model.fit(
        fold_artifacts["X_train_seq"],
        fold_artifacts["y_train_seq"],
        validation_data=(fold_artifacts["X_val_seq"], fold_artifacts["y_val_seq"]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )

    y_val_pred_scaled = model.predict(fold_artifacts["X_val_seq"], verbose=0)
    y_val_true = inverse_y(
        fold_artifacts["y_val_seq"],
        fold_artifacts["target_transformer"],
    )
    y_val_pred = inverse_y(
        y_val_pred_scaled,
        fold_artifacts["target_transformer"],
    )

    val_metrics = _metrics(y_val_true, y_val_pred)

    result = {
        "model": model,
        "history": history.history,
        "fold": fold_artifacts.get("fold"),
        "walk_forward": fold_artifacts.get("walk_forward"),
        "val_metrics": val_metrics,
        "y_val_true": y_val_true,
        "y_val_pred": y_val_pred,
        "val_times": fold_artifacts.get("val_times"),
    }
    return result


def run_walk_forward_training(
    cfg: DataConfig,
    eval_size: int = 72,
    step_size: int = 24,
    max_folds: int = 4,
    expanding: bool = True,
    train_size: int | None = None,
    seed: int = 42,
    epochs: int = 40,
    batch_size: int = 64,
    verbose: int = 0,
) -> dict[str, Any]:
    folds = prepare_walk_forward_datasets(
        cfg,
        eval_size=eval_size,
        step_size=step_size,
        max_folds=max_folds,
        expanding=expanding,
        train_size=train_size,
    )

    fold_results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for fold_artifacts in folds:
        fold_res = train_single_fold(
            fold_artifacts=fold_artifacts,
            seed=seed,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        fold_results.append(fold_res)

        info = fold_artifacts.get("walk_forward", {})
        rows.append(
            {
                "fold": int(info.get("fold", len(rows) + 1)),
                "train_end": info.get("train_end"),
                "val_start": info.get("val_start"),
                "val_end": info.get("val_end"),
                "val_mae": fold_res["val_metrics"]["mae"],
                "val_rmse": fold_res["val_metrics"]["rmse"],
                "val_mape": fold_res["val_metrics"]["mape"],
            }
        )

    fold_metrics_df = pd.DataFrame(rows)
    summary = {
        "mae_mean": float(fold_metrics_df["val_mae"].mean()),
        "mae_std": float(fold_metrics_df["val_mae"].std(ddof=0)),
        "rmse_mean": float(fold_metrics_df["val_rmse"].mean()),
        "rmse_std": float(fold_metrics_df["val_rmse"].std(ddof=0)),
        "mape_mean": float(fold_metrics_df["val_mape"].mean()),
        "mape_std": float(fold_metrics_df["val_mape"].std(ddof=0)),
    }

    return {
        "cfg": asdict(cfg),
        "fold_metrics": fold_metrics_df,
        "summary": summary,
        "fold_results": fold_results,
    }


def train_final_model(
    cfg: DataConfig,
    seed: int = 42,
    epochs: int = 40,
    batch_size: int = 64,
    verbose: int = 0,
) -> dict[str, Any]:
    artifacts = prepare_dataset(cfg)

    result = train_single_fold(
        fold_artifacts=artifacts,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    return {
        "artifacts": artifacts,
        "model": result["model"],
        "history": result["history"],
        "val_metrics": result["val_metrics"],
    }


def forecast_next_hours(
    model: tf.keras.Model,
    artifacts: dict[str, Any],
    hours: int = 72,
) -> pd.DataFrame:
    cfg: DataConfig = artifacts["cfg"]

    if hours <= 0:
        raise ValueError("hours phải > 0.")

    if hours > cfg.horizon:
        raise ValueError(
            f"Model đang output horizon={cfg.horizon}. Hãy forecast <= {cfg.horizon} giờ."
        )

    df_all = artifacts["df"].sort_index()
    preprocessor = artifacts["preprocessor"]
    target_transformer = artifacts["target_transformer"]

    X_all = preprocessor.transform(df_all, target=cfg.target)
    if len(X_all) < cfg.lookback:
        raise ValueError("Không đủ dữ liệu để tạo cửa sổ lookback cho dự báo.")

    last_window = X_all[-cfg.lookback :]
    y_pred_scaled = model.predict(last_window[np.newaxis, ...], verbose=0)[0]
    y_pred = target_transformer.inverse_transform(y_pred_scaled[:hours]).reshape(-1)
    y_pred = np.clip(y_pred, 0.0, None)

    start_time = df_all.index.max() + pd.tseries.frequencies.to_offset(cfg.freq)
    future_index = pd.date_range(start=start_time, periods=hours, freq=cfg.freq)

    return pd.DataFrame(
        {
            "timestamp": future_index,
            "pm25_pred": y_pred,
        }
    )


def summarize_daily_forecast(hourly_forecast: pd.DataFrame) -> pd.DataFrame:
    daily = (
        hourly_forecast.assign(date=hourly_forecast["timestamp"].dt.date)
        .groupby("date", as_index=False)
        .agg(
            pm25_mean=("pm25_pred", "mean"),
            pm25_max=("pm25_pred", "max"),
            pm25_min=("pm25_pred", "min"),
        )
    )
    return daily


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Train GRU with walk-forward and forecast next 72 hours.")
    parser.add_argument("--eval-size", type=int, default=72)
    parser.add_argument("--step-size", type=int, default=24)
    parser.add_argument("--max-folds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--expanding", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, default="data/processed")
    args = parser.parse_args()

    cfg = DataConfig()

    walk_result = run_walk_forward_training(
        cfg=cfg,
        eval_size=args.eval_size,
        step_size=args.step_size,
        max_folds=args.max_folds,
        expanding=args.expanding,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
    )

    print("Walk-forward fold metrics:")
    print(walk_result["fold_metrics"].to_string(index=False))
    print("\nWalk-forward summary:")
    for k, v in walk_result["summary"].items():
        print(f"  {k}: {v:.4f}")

    final_fit = train_final_model(
        cfg=cfg,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=0,
    )
    hourly = forecast_next_hours(final_fit["model"], final_fit["artifacts"], hours=72)
    daily = summarize_daily_forecast(hourly)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hourly_path = output_dir / "gru_forecast_72h.csv"
    daily_path = output_dir / "gru_forecast_daily.csv"

    hourly.to_csv(hourly_path, index=False)
    daily.to_csv(daily_path, index=False)

    print(f"\nSaved hourly forecast: {hourly_path}")
    print(f"Saved daily forecast: {daily_path}")


if __name__ == "__main__":
    run_cli()
