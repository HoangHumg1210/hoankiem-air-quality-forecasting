import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_utils import CFG, DataConfig, prepare_walk_forward_datasets


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


class WalkForwardSyntheticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "synthetic_walk_forward.csv"
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

    def test_walk_forward_defaults_to_two_folds_over_val_and_test(self):
        folds = prepare_walk_forward_datasets(self.cfg)

        self.assertEqual(len(folds), 2)
        self.assertEqual(folds[0]["fold"], 1)
        self.assertEqual(folds[1]["fold"], 2)
        self.assertEqual(pd.Timestamp(folds[0]["walk_forward"]["val_start"]), pd.Timestamp("2024-01-04 08:00:00"))
        self.assertEqual(pd.Timestamp(folds[0]["walk_forward"]["val_end"]), pd.Timestamp("2024-01-04 17:00:00"))
        self.assertEqual(pd.Timestamp(folds[1]["walk_forward"]["val_start"]), pd.Timestamp("2024-01-04 18:00:00"))
        self.assertEqual(pd.Timestamp(folds[1]["walk_forward"]["val_end"]), pd.Timestamp("2024-01-05 03:00:00"))
        self.assertEqual(folds[0]["X_val_seq"].shape[0], 9)
        self.assertEqual(folds[1]["X_val_seq"].shape[0], 9)

    def test_walk_forward_supports_rolling_train_window(self):
        folds = prepare_walk_forward_datasets(
            self.cfg,
            eval_size=4,
            step_size=4,
            max_folds=3,
            expanding=False,
            train_size=80,
        )

        self.assertEqual(len(folds), 3)
        self.assertEqual(folds[0]["walk_forward"]["train_size"], 80)
        self.assertEqual(folds[1]["walk_forward"]["train_size"], 80)
        self.assertEqual(folds[2]["walk_forward"]["train_size"], 80)
        self.assertEqual(pd.Timestamp(folds[0]["walk_forward"]["val_start"]), pd.Timestamp("2024-01-04 08:00:00"))
        self.assertEqual(pd.Timestamp(folds[1]["walk_forward"]["val_start"]), pd.Timestamp("2024-01-04 12:00:00"))
        self.assertEqual(pd.Timestamp(folds[2]["walk_forward"]["val_start"]), pd.Timestamp("2024-01-04 16:00:00"))


class WalkForwardSmokeTests(unittest.TestCase):
    def test_prepare_walk_forward_real_data_smoke(self):
        folds = prepare_walk_forward_datasets(CFG, max_folds=2)

        self.assertEqual(len(folds), 2)
        self.assertEqual(folds[0]["X_val_seq"].shape, (8713, 336, 43))
        self.assertEqual(folds[1]["X_val_seq"].shape[0], 8689)
        self.assertEqual(folds[1]["X_val_seq"].shape[1], 336)
        self.assertEqual(folds[0]["X_test_seq"].shape[0], 0)
        self.assertEqual(folds[1]["X_test_seq"].shape[0], 0)


if __name__ == "__main__":
    unittest.main()
