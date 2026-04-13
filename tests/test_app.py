import pandas as pd

from app import dedupe_model_metrics, load_registry_metrics


def test_dedupe_model_metrics_keeps_best_row_per_model():
    metrics = pd.DataFrame(
        [
            {
                "Model": "Seq2Seq LSTM + Attention",
                "Bundle Key": "best_model_bundle",
                "Bundle Dir": "best_model_bundle",
                "MAE": 5.70,
                "RMSE": 9.90,
                "MAPE": 24.1,
                "Peak MAE": 17.4,
                "R2": None,
                "Has Timeline": True,
                "Source": "best_model_bundle",
                "Source Rank": 2,
            },
            {
                "Model": "Seq2Seq LSTM + Attention",
                "Bundle Key": "seq2seq_lstm_attention",
                "Bundle Dir": "model_registry/seq2seq_lstm_attention",
                "MAE": 5.70,
                "RMSE": 9.90,
                "MAPE": 24.1,
                "Peak MAE": 17.4,
                "R2": None,
                "Has Timeline": True,
                "Source": "model_registry",
                "Source Rank": 1,
            },
            {
                "Model": "Seq2Seq GRU + Attention",
                "Bundle Key": "seq2seq_gru_attention",
                "Bundle Dir": "model_registry/seq2seq_gru_attention",
                "MAE": 6.28,
                "RMSE": 10.41,
                "MAPE": 26.5,
                "Peak MAE": 18.0,
                "R2": None,
                "Has Timeline": True,
                "Source": "model_registry",
                "Source Rank": 1,
            },
        ]
    )

    deduped = dedupe_model_metrics(metrics)

    assert deduped["Model"].tolist() == ["Seq2Seq LSTM + Attention", "Seq2Seq GRU + Attention"]
    assert deduped.iloc[0]["Source"] == "model_registry"
    assert "Source Rank" not in deduped.columns


def test_load_registry_metrics_returns_unique_model_names():
    metrics = load_registry_metrics()

    assert metrics["Model"].is_unique
    assert "Seq2Seq LSTM + Attention" in set(metrics["Model"])
