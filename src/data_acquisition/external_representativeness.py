"""Cross-source and cross-point representativeness check for PM2.5.

Question answered: can one coordinate represent the whole Hoan Kiem district?

Outputs under results/external_check/
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXT_DIR = ROOT / "data" / "external" / "open_meteo"
OUT_DIR = ROOT / "results" / "external_check"
PROJECT_CSV = ROOT / "data" / "processed" / "cleaned_data.csv"

CENTRAL = "hoankiem_center"
POINTS_NAMES = ["hoankiem_center", "hang_bai", "dong_xuan", "hang_bac",
                "ba_dinh", "hai_ba_trung", "long_bien"]
HOANKIEM_SET = {"hoankiem_center", "hang_bai", "dong_xuan", "hang_bac"}
RENAME = {"pm2_5": "PM25", "pm10": "PM10", "nitrogen_dioxide": "NO2",
          "ozone": "O3", "sulphur_dioxide": "SO2", "carbon_monoxide": "CO"}


def load_project() -> pd.DataFrame:
    df = pd.read_csv(PROJECT_CSV)
    df["Local Time"] = pd.to_datetime(df["Local Time"])
    return df.set_index("Local Time")[["PM25", "PM10", "NO2", "O3", "SO2", "CO"]]


def load_external(name: str) -> pd.DataFrame:
    df = pd.read_csv(EXT_DIR / f"airq_{name}.csv")
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").rename(columns=RENAME)


def _metrics(a: pd.Series, b: pd.Series) -> dict[str, float | int]:
    pair = pd.concat([a, b], axis=1).dropna()
    if pair.empty:
        return {"n": 0}
    x, y = pair.iloc[:, 0], pair.iloc[:, 1]
    diff = x - y
    corr = x.corr(y)
    return {
        "n": int(len(pair)),
        "mean_a": float(x.mean()),
        "mean_b": float(y.mean()),
        "mean_diff": float(diff.mean()),
        "mae": float(diff.abs().mean()),
        "rmse": float(np.sqrt((diff ** 2).mean())),
        "corr": float(corr) if pd.notna(corr) else float("nan"),
        "nmae_pct": float(diff.abs().mean() / y.mean() * 100) if y.mean() else float("nan"),
    }


def compare_sources() -> pd.DataFrame:
    project = load_project()
    rows = []
    for name in POINTS_NAMES:
        ext = load_external(name)
        overlap = project.index.intersection(ext.index)
        if overlap.empty:
            continue
        for pollutant in ["PM25", "PM10", "NO2", "O3"]:
            if pollutant not in ext.columns:
                continue
            rows.append({"point": name, "pollutant": pollutant,
                         **_metrics(ext.loc[overlap, pollutant], project.loc[overlap, pollutant])})
    return pd.DataFrame(rows)


def representativeness() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = {n: load_external(n)["PM25"] for n in POINTS_NAMES}
    wide = pd.DataFrame(frames).dropna(how="all")
    pair_rows = [{"point_a": a, "point_b": b, **_metrics(wide[a], wide[b])}
                 for a, b in combinations(sorted(frames), 2)]
    stats = [{"point": n, "n": int(s.dropna().size), "mean": float(s.mean()),
              "median": float(s.median()), "p95": float(s.quantile(0.95)), "max": float(s.max())}
             for n, s in frames.items()]
    return pd.DataFrame(pair_rows), pd.DataFrame(stats)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    src = compare_sources()
    src.to_csv(OUT_DIR / "compare_sources.csv", index=False, encoding="utf-8")

    pair_df, point_df = representativeness()
    pair_df.to_csv(OUT_DIR / "representativeness.csv", index=False, encoding="utf-8")
    point_df.to_csv(OUT_DIR / "point_stats.csv", index=False, encoding="utf-8")

    inner = pair_df[pair_df["point_a"].isin(HOANKIEM_SET) & pair_df["point_b"].isin(HOANKIEM_SET)]
    pm25_src = src[(src["pollutant"] == "PM25") & (src["point"] == CENTRAL)]

    summary = {
        "window": ["2022-01-13", "2024-12-31"],
        "source_check_pm25": pm25_src.drop(columns=["point", "pollutant"]).round(3).to_dict("records"),
        "pm25_pairwise_mae_mean_all_points": round(float(pair_df["mae"].mean()), 2),
        "pm25_pairwise_mae_max_all_points": round(float(pair_df["mae"].max()), 2),
        "pm25_pairwise_nmae_pct_mean_all_points": round(float(pair_df["nmae_pct"].mean()), 1),
        "pm25_pairwise_corr_min_all_points": round(float(pair_df["corr"].min()), 3),
        "pm25_within_hoankiem_mae_mean": round(float(inner["mae"].mean()), 2) if len(inner) else None,
        "pm25_within_hoankiem_corr_min": round(float(inner["corr"].min()), 3) if len(inner) else None,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
