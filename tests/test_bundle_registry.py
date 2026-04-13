import json
from pathlib import Path

from bundle_registry import load_registry_metrics, promote_best_bundle, select_best_bundle


def _write_bundle(bundle_dir: Path, *, model_name: str, bundle_key: str, mae: float) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "config.json").write_text(
        json.dumps(
            {
                "model_name": model_name,
                "bundle_key": bundle_key,
                "lookback": 72,
                "chunk_horizon": 1,
                "rollout_horizon": 24,
            }
        ),
        encoding="utf-8",
    )
    (bundle_dir / "metrics.json").write_text(
        json.dumps(
            {
                "mae": mae,
                "mse": mae * mae,
                "rmse": mae,
                "mape": mae * 10,
                "peak_mae": mae * 2,
            }
        ),
        encoding="utf-8",
    )
    for filename in ("feature_cols.pkl", "x_scaler.pkl", "y_scaler.pkl", "model.keras", "test_timeline.csv"):
        (bundle_dir / filename).write_bytes(f"{bundle_key}:{filename}".encode("utf-8"))


def test_load_registry_metrics_and_select_best_bundle(tmp_path):
    registry_dir = tmp_path / "model_registry"
    best_bundle_dir = tmp_path / "best_model_bundle"

    _write_bundle(registry_dir / "model_a", model_name="Model A", bundle_key="model_a", mae=6.2)
    _write_bundle(registry_dir / "model_b", model_name="Model B", bundle_key="model_b", mae=4.8)
    _write_bundle(best_bundle_dir, model_name="Model B", bundle_key="model_b", mae=4.8)

    metrics = load_registry_metrics(tmp_path, registry_dir=registry_dir, best_bundle_dir=best_bundle_dir)
    best_row = select_best_bundle(metrics)

    assert metrics["Model"].is_unique
    assert best_row["Bundle Key"] == "model_b"
    assert best_row["Source"] == "model_registry"


def test_promote_best_bundle_copies_best_registry_bundle(tmp_path):
    registry_dir = tmp_path / "model_registry"
    best_bundle_dir = tmp_path / "best_model_bundle"

    _write_bundle(registry_dir / "winner_bundle", model_name="Winner", bundle_key="winner_bundle", mae=3.5)
    _write_bundle(registry_dir / "runner_up", model_name="Runner Up", bundle_key="runner_up", mae=5.1)

    result = promote_best_bundle(tmp_path, registry_dir=registry_dir, best_bundle_dir=best_bundle_dir)

    assert result["bundle_key"] == "winner_bundle"
    assert (best_bundle_dir / "model.keras").read_bytes() == b"winner_bundle:model.keras"

    best_info = json.loads((best_bundle_dir / "best_info.json").read_text(encoding="utf-8"))
    assert best_info["winner_model"] == "Winner"
    assert best_info["source_bundle"] == "winner_bundle"
