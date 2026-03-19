import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_utils import CFG, DataConfig, TargetTransformer, prepare_dataset


def build_synthetic_dataset() -> pd.DataFrame:
    index = pd.date_range("2024-01-01 00:00:00", periods=100, freq="1h")
    df = pd.DataFrame(
        {
            "Local Time": index,
            "PM25": np.linspace(10.0, 109.0, num=100),
            "CO": np.linspace(100.0, 199.0, num=100),
            "HolidayName": ["Weekday"] * 100,
            "IsHoliday": [False] * 100,
        }
    )

    train_end_idx = 79
    val_start_idx = 80

    df.loc[train_end_idx, "PM25"] = np.nan
    df.loc[train_end_idx, "CO"] = np.nan
    df.loc[val_start_idx, "PM25"] = 999.0
    df.loc[val_start_idx, "CO"] = 9999.0
    return df


class PrepareDatasetSyntheticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "synthetic.csv"
        build_synthetic_dataset().to_csv(self.csv_path, index=False)
        self.cfg = DataConfig(
            data_path=str(self.csv_path),
            target="PM25",
            time_col="Local Time",
            lookback=5,
            horizon=2,
            target_transform="none",
            train_end="2024-01-04 07:00:00",
            val_start="2024-01-04 08:00:00",
            val_end="2024-01-04 17:00:00",
            test_start="2024-01-04 18:00:00",
            freq="1h",
            seed=42,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_prepare_dataset_avoids_cross_split_leakage(self):
        artifacts = prepare_dataset(self.cfg)

        self.assertEqual(artifacts["train_df"]["PM25"].iloc[-1], 88.0)
        self.assertEqual(artifacts["train_df"]["CO"].iloc[-1], 178.0)
        self.assertNotEqual(artifacts["train_df"]["PM25"].iloc[-1], 999.0)
        self.assertNotEqual(artifacts["train_df"]["CO"].iloc[-1], 9999.0)

    def test_prepare_dataset_uses_prior_context_for_val_and_test_sequences(self):
        artifacts = prepare_dataset(self.cfg)

        val_times = pd.DatetimeIndex(artifacts["val_times"])
        test_times = pd.DatetimeIndex(artifacts["test_times"])

        self.assertEqual(artifacts["X_val_seq"].shape[0], len(artifacts["val_df"]) - self.cfg.horizon + 1)
        self.assertEqual(artifacts["X_test_seq"].shape[0], len(artifacts["test_df"]) - self.cfg.horizon + 1)
        self.assertEqual(val_times[0], pd.Timestamp(self.cfg.val_start))
        self.assertEqual(test_times[0], pd.Timestamp(self.cfg.test_start))
        self.assertTrue((val_times >= pd.Timestamp(self.cfg.val_start)).all())
        self.assertTrue((val_times <= pd.Timestamp(self.cfg.val_end)).all())
        self.assertTrue((test_times >= pd.Timestamp(self.cfg.test_start)).all())


class TargetTransformerTests(unittest.TestCase):
    def test_none_mode_is_true_identity(self):
        y = np.array([[-2.0], [0.0], [3.5]])
        transformer = TargetTransformer(mode="none")

        y_scaled = transformer.fit_transform(y)
        y_restored = transformer.inverse_transform(y_scaled)

        np.testing.assert_allclose(y_restored, y)

    def test_log_and_sqrt_reject_negative_targets(self):
        y = np.array([[-1.0], [2.0]])

        for mode in ("log", "sqrt"):
            transformer = TargetTransformer(mode=mode)
            with self.assertRaises(ValueError):
                transformer.fit_transform(y)


class PrepareDatasetSmokeTests(unittest.TestCase):
    def test_prepare_dataset_real_data_smoke(self):
        artifacts = prepare_dataset(CFG)

        self.assertEqual(artifacts["X_train_seq"].shape, (16746, 336, 43))
        self.assertEqual(artifacts["X_val_seq"].shape, (8713, 336, 43))
        self.assertEqual(artifacts["X_test_seq"].shape, (8689, 336, 43))
        self.assertEqual(artifacts["y_train_seq"].shape, (16746, 72))
        self.assertEqual(artifacts["y_val_seq"].shape, (8713, 72))
        self.assertEqual(artifacts["y_test_seq"].shape, (8689, 72))


if __name__ == "__main__":
    unittest.main()
