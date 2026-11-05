"""Wind-sector analysis: PM2.5 by 8 wind directions x 4 seasons.

Uses only the project dataset (no external API).
Output: results/wind_sector/
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "wind_sector"
PROJECT_CSV = ROOT / "data" / "processed" / "cleaned_data.csv"

SECTORS = [(0, "N"), (45, "NE"), (90, "E"), (135, "SE"), (180, "S"), (225, "SW"), (270, "W"), (315, "NW")]
SEASON_MAP = {12: "Đông", 1: "Đông", 2: "Đông",
              3: "Xuân", 4: "Xuân", 5: "Xuân",
              6: "Hè", 7: "Hè", 8: "Hè",
              9: "Thu", 10: "Thu", 11: "Thu"}


def bearing_to_sector(deg: pd.Series) -> pd.Series:
    idx = ((deg % 360) / 45).round().astype(int) % 8
    labels = np.array([s[1] for s in SECTORS])
    return pd.Series(labels[idx], index=deg.index)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROJECT_CSV)
    df["t"] = pd.to_datetime(df["Local Time"])
    df = df.set_index("t").sort_index()

    df["sector"] = bearing_to_sector(df["Wind Direction"])
    df["season"] = df.index.month.map(SEASON_MAP)

    overall = df.groupby("sector")["PM25"].agg(["median", "mean", "count"])
    overall = overall.reindex([s[1] for s in SECTORS])
    overall.to_csv(OUT / "pm25_by_sector.csv", encoding="utf-8")

    table = df.pivot_table(index="sector", columns="season", values="PM25", aggfunc="median")
    table = table.reindex([s[1] for s in SECTORS])[["Xuân", "Hè", "Thu", "Đông"]]
    table.to_csv(OUT / "pm25_by_sector_season.csv", encoding="utf-8")

    counts = df.pivot_table(index="sector", columns="season", values="PM25", aggfunc="count")
    counts = counts.reindex([s[1] for s in SECTORS])[["Xuân", "Hè", "Thu", "Đông"]]

    # wind-speed split: calm vs windy tells more than direction alone
    calm = df[df["Wind Speed"] < 1.5]
    windy = df[df["Wind Speed"] >= 3.0]
    speed_summary = {
        "calm_hours": int(len(calm)),
        "calm_pm25_median": round(float(calm["PM25"].median()), 2),
        "windy_hours": int(len(windy)),
        "windy_pm25_median": round(float(windy["PM25"].median()), 2),
        "all_median": round(float(df["PM25"].median()), 2),
    }

    # figure: polar plot of median PM2.5 per sector
    fig = plt.figure(figsize=(11, 4.6))
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    angles = np.deg2rad([s[0] for s in SECTORS])
    values = np.r_[overall["median"].values, overall["median"].values[0]]
    ang = np.r_[angles, angles[0]]
    ax.plot(ang, values, color="#0e7490", lw=2)
    ax.fill(ang, values, color="#0e7490", alpha=0.22)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels([s[1] for s in SECTORS], fontsize=9)
    ax.set_title("PM2.5 trung vị theo hướng gió\n(µg/m³, 2022–2026)", fontsize=10.5, pad=18)

    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(table.values, cmap="YlOrRd", aspect="auto")
    ax2.set_xticks(range(table.shape[1]))
    ax2.set_xticklabels(table.columns, fontsize=9)
    ax2.set_yticks(range(table.shape[0]))
    ax2.set_yticklabels(table.index, fontsize=9)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            if np.isfinite(v):
                ax2.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5,
                         color="white" if v > np.nanmedian(table.values) else "#1e293b")
    ax2.set_title("PM2.5 trung vị: hướng gió × mùa", fontsize=10.5)
    ax2.set_xlabel("Mùa")
    ax2.set_ylabel("Hướng gió")
    fig.colorbar(im, ax=ax2, label="µg/m³", fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(OUT / "wind_sector_pm25.png", bbox_inches="tight")
    plt.close(fig)

    dirty = overall["median"].idxmax()
    clean = overall["median"].idxmin()
    gap = float(overall["median"].max() - overall["median"].min())
    winter = table["Đông"].dropna()
    summary = {
        "worst_sector": dirty,
        "worst_median": round(float(overall["median"].max()), 2),
        "cleanest_sector": clean,
        "cleanest_median": round(float(overall["median"].min()), 2),
        "sector_gap": round(gap, 2),
        "winter_worst_sector": winter.idxmax(),
        "winter_worst_median": round(float(winter.max()), 2),
        "winter_cleanest_sector": winter.idxmin(),
        "winter_cleanest_median": round(float(winter.min()), 2),
        **speed_summary,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nBy sector:\n" + overall.round(2).to_string())
    print("\nBy sector x season (median):\n" + table.round(1).to_string())
    print(f"\nOutputs -> {OUT}")


if __name__ == "__main__":
    main()
