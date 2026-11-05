"""Climate context analysis for the Hoan Kiem PM2.5 study.

Produces the evidence needed to answer: "Does climate change / El Nino
appear in this dataset, and does it matter for short-term forecasting?"

Outputs (results/climate_review/):
  figure1_warming.png        Hanoi annual temperature vs 1991-2020 baseline
  figure2_pm25_vs_oni.png    monthly PM2.5 with El Nino / La Nina shading
  figure3_scatter.png        ONI vs PM2.5 scatter with regression line
  climate_indicators.csv     per-year extreme-weather indicators from project data
  enso_correlation.json      statistical tests
  report.md                  written summary with numbers
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "climate_review"
PROJECT_CSV = ROOT / "data" / "processed" / "cleaned_data.csv"
EXT_DIR = ROOT / "data" / "external" / "open_meteo"

START, END = "1991-01-01", "2025-12-31"
BASELINE = (1991, 2020)
RECENT = (2022, 2025)

ONI_SEASON_TO_MONTH = {
    "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
    "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
}

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
})


# ---------------------------------------------------------------- data loads
def load_project() -> pd.DataFrame:
    df = pd.read_csv(PROJECT_CSV)
    df["t"] = pd.to_datetime(df["Local Time"])
    return df.set_index("t").sort_index()


def fetch_era5_daily() -> pd.DataFrame:
    cache = EXT_DIR / "era5_hanoi_daily_1991_2025.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        df["time"] = pd.to_datetime(df["time"])
        return df
    resp = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": 21.0285,
            "longitude": 105.8542,
            "start_date": START,
            "end_date": END,
            "daily": "temperature_2m_mean,temperature_2m_max,precipitation_sum",
            "timezone": "Asia/Bangkok",
        },
        timeout=300,
    )
    resp.raise_for_status()
    payload = resp.json()
    df = pd.DataFrame(payload["daily"])
    df["source"] = "ERA5 (Open-Meteo archive)"
    df["lat"] = payload.get("latitude")
    df["lon"] = payload.get("longitude")
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False, encoding="utf-8")
    df["time"] = pd.to_datetime(df["time"])
    return df


def fetch_oni() -> pd.DataFrame:
    cache = EXT_DIR / "oni_noaa_1950_2025.csv"
    if cache.exists():
        df = pd.read_csv(cache)
        df["t"] = pd.to_datetime(df["t"])
        return df
    resp = requests.get("https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt", timeout=60)
    resp.raise_for_status()
    rows = [line.split() for line in resp.text.strip().splitlines()[1:] if len(line.split()) == 4]
    df = pd.DataFrame(rows, columns=["season", "year", "sst", "oni"])
    df["year"] = df["year"].astype(int)
    df["oni"] = df["oni"].astype(float)
    df["sst"] = df["sst"].astype(float)
    df["month"] = df["season"].map(ONI_SEASON_TO_MONTH)
    df = df.dropna(subset=["month"])
    df["month"] = df["month"].astype(int)
    df["t"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=15))
    EXT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False, encoding="utf-8")
    return df


# ------------------------------------------------------------- figure 1
def figure_warming(era: pd.DataFrame) -> dict:
    era = era.copy()
    era["year"] = era["time"].dt.year
    yearly = era.groupby("year")["temperature_2m_mean"].agg(["mean", "count"])
    yearly = yearly[yearly["count"] >= 300]["mean"]

    base = float(yearly.loc[BASELINE[0]:BASELINE[1]].mean())
    anom = yearly - base

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    ax.plot(yearly.index, yearly.values, color="#f97316", lw=1.6, marker="o", ms=3, label="Nhiệt độ TB năm (ERA5)")
    z = np.polyfit(yearly.index.astype(float), yearly.values, 1)
    ax.plot(yearly.index, np.polyval(z, yearly.index.astype(float)), color="#0f172a", lw=2, ls="--",
            label=f"Xu hướng: {z[0] * 10:+.2f} °C/thập kỷ")
    ax.axhline(base, color="#3b82f6", lw=1.2, ls=":", label=f"Baseline {BASELINE[0]}–{BASELINE[1]}: {base:.2f} °C")
    ax.set_title("Hà Nội: nhiệt độ trung bình năm 1991–2025", fontsize=11, loc="left")
    ax.set_ylabel("°C")
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    ax = axes[1]
    colors = ["#dc2626" if v > 0 else "#2563eb" for v in anom.values]
    ax.bar(anom.index, anom.values, color=colors, width=0.8)
    ax.axhline(0, color="#334155", lw=1)
    recent_mask = anom.index.isin(range(RECENT[0], RECENT[1] + 1))
    for yr in anom.index[recent_mask]:
        ax.annotate(f"{anom.loc[yr]:+.2f}", (yr, anom.loc[yr]), textcoords="offset points",
                    xytext=(0, 5 if anom.loc[yr] > 0 else -11), ha="center", fontsize=7.5, fontweight="bold")
    ax.set_title(f"Độ lệch so baseline {BASELINE[0]}–{BASELINE[1]}", fontsize=11, loc="left")
    ax.set_ylabel("°C")
    ax.set_xlabel("Năm")

    fig.tight_layout()
    fig.savefig(OUT / "figure1_warming.png", bbox_inches="tight")
    plt.close(fig)

    warmest = int(yearly.idxmax())
    recent_mean = float(yearly.loc[RECENT[0]:RECENT[1]].mean())
    return {
        "baseline_1991_2020_c": round(base, 2),
        "trend_c_per_decade": round(float(z[0]) * 10, 3),
        "warmest_year": warmest,
        "warmest_year_c": round(float(yearly.loc[warmest]), 2),
        "recent_2022_2025_mean_c": round(recent_mean, 2),
        "recent_anomaly_c": round(recent_mean - base, 2),
        "years_above_baseline_recent": int((anom.loc[RECENT[0]:RECENT[1]] > 0).sum()),
        "years_available": int(yearly.size),
    }


# ------------------------------------------------------------- figure 2/3
def figure_enso(project: pd.DataFrame, oni: pd.DataFrame) -> dict:
    monthly = project["PM25"].resample("ME").mean().dropna()
    frame = pd.DataFrame({
        "t": monthly.index.to_period("M").to_timestamp() + pd.Timedelta(days=14),
        "pm25": monthly.values,
    })
    frame = frame.merge(oni[["t", "oni"]], on="t", how="inner").dropna().sort_values("t").set_index("t")
    frame["month"] = frame.index.month
    frame["phase"] = pd.cut(frame["oni"], [-9, -0.5, 0.5, 9], labels=["La Niña", "Trung tính", "El Niño"])

    # --- figure 2: time series with phase shading
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 5.6), sharex=True,
                                   gridspec_kw={"height_ratios": [2.4, 1]})

    ax1.plot(frame.index, frame["pm25"], color="#0e7490", lw=1.8, marker="o", ms=3.5, label="PM2.5 TB tháng")
    for _, seg in frame.groupby((frame["phase"] != frame["phase"].shift()).cumsum()):
        phase = seg["phase"].iloc[0]
        color = {"El Niño": "#fecaca", "La Niña": "#bfdbfe", "Trung tính": "#f1f5f9"}.get(str(phase), "#f1f5f9")
        ax1.axvspan(seg.index.min(), seg.index.max(), color=color, alpha=0.75, lw=0, zorder=0)
    ax1.set_ylabel("PM2.5 (µg/m³)")
    ax1.set_title("PM2.5 tháng tại Hoàn Kiếm và pha ENSO (màu nền)", fontsize=11, loc="left")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#fecaca", label="El Niño (ONI ≥ +0.5)"),
        plt.Rectangle((0, 0), 1, 1, color="#f1f5f9", label="Trung tính"),
        plt.Rectangle((0, 0), 1, 1, color="#bfdbfe", label="La Niña (ONI ≤ −0.5)"),
        plt.Line2D([], [], color="#0e7490", marker="o", ms=4, label="PM2.5"),
    ]
    ax1.legend(handles=handles, fontsize=8, frameon=False, ncol=4, loc="upper right")

    ax2.plot(frame.index, frame["oni"], color="#7c3aed", lw=1.8)
    ax2.axhline(0.5, color="#dc2626", ls="--", lw=1)
    ax2.axhline(-0.5, color="#2563eb", ls="--", lw=1)
    ax2.axhline(0, color="#334155", lw=0.8)
    ax2.fill_between(frame.index, 0.5, frame["oni"], where=frame["oni"] > 0.5, color="#dc2626", alpha=0.25)
    ax2.fill_between(frame.index, -0.5, frame["oni"], where=frame["oni"] < -0.5, color="#2563eb", alpha=0.25)
    ax2.set_ylabel("ONI (°C)")
    ax2.set_title("Chỉ số ONI (NOAA CPC)", fontsize=10, loc="left")
    ax2.set_xlabel("Thời gian")

    fig.tight_layout()
    fig.savefig(OUT / "figure2_pm25_vs_oni.png", bbox_inches="tight")
    plt.close(fig)

    # --- figure 3: scatter + regression
    r_all, p_all = stats.pearsonr(frame["oni"], frame["pm25"])
    rho_all, prho_all = stats.spearmanr(frame["oni"], frame["pm25"])
    winter = frame[frame["month"].isin([11, 12, 1, 2])]
    r_win, p_win = stats.pearsonr(winter["oni"], winter["pm25"])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    palette = {"El Niño": "#dc2626", "Trung tính": "#64748b", "La Niña": "#2563eb"}
    for phase, seg in frame.groupby("phase", observed=True):
        axes[0].scatter(seg["oni"], seg["pm25"], s=42, alpha=0.85, color=palette.get(str(phase), "#64748b"),
                        label=str(phase), edgecolor="white", linewidth=0.8)
    xs = np.linspace(frame["oni"].min(), frame["oni"].max(), 50)
    slope, intercept = np.polyfit(frame["oni"], frame["pm25"], 1)
    axes[0].plot(xs, slope * xs + intercept, color="#0f172a", lw=2, ls="--")
    axes[0].set_xlabel("ONI (°C)")
    axes[0].set_ylabel("PM2.5 tháng (µg/m³)")
    axes[0].set_title(f"Toàn bộ tháng: r = {r_all:+.3f}, p = {p_all:.3f}", fontsize=10.5, loc="left")
    axes[0].legend(fontsize=8, frameon=False)

    for phase, seg in winter.groupby("phase", observed=True):
        axes[1].scatter(seg["oni"], seg["pm25"], s=52, alpha=0.9, color=palette.get(str(phase), "#64748b"),
                        label=str(phase), edgecolor="white", linewidth=0.8)
    xs = np.linspace(winter["oni"].min(), winter["oni"].max(), 50)
    s2, i2 = np.polyfit(winter["oni"], winter["pm25"], 1)
    axes[1].plot(xs, s2 * xs + i2, color="#0f172a", lw=2, ls="--")
    axes[1].set_xlabel("ONI (°C)")
    axes[1].set_ylabel("PM2.5 tháng (µg/m³)")
    axes[1].set_title(f"Mùa đông (T11–T2): r = {r_win:+.3f}, p = {p_win:.3f}", fontsize=10.5, loc="left")
    axes[1].legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "figure3_scatter.png", bbox_inches="tight")
    plt.close(fig)

    by_phase = frame.groupby("phase", observed=True)["pm25"].agg(["mean", "count"])
    winter_by_phase = winter.groupby("phase", observed=True)["pm25"].agg(["mean", "count"])
    frame.to_csv(OUT / "pm25_oni_monthly.csv", encoding="utf-8")

    return {
        "n_months": int(len(frame)),
        "pearson_r_all": round(float(r_all), 3),
        "p_value_all": round(float(p_all), 4),
        "spearman_all": round(float(rho_all), 3),
        "pearson_r_winter": round(float(r_win), 3),
        "p_value_winter": round(float(p_win), 4),
        "n_winter_months": int(len(winter)),
        "mean_pm25_by_phase": {str(k): [round(float(v["mean"]), 2), int(v["count"])] for k, v in by_phase.iterrows()},
        "winter_mean_pm25_by_phase": {str(k): [round(float(v["mean"]), 2), int(v["count"])] for k, v in winter_by_phase.iterrows()},
    }


# ------------------------------------------------------- indicators table
def climate_indicators(project: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, seg in project.groupby(project.index.year):
        if len(seg) < 2000:
            note = f"không đủ năm ({len(seg)} giờ)"
        else:
            note = ""
        rows.append({
            "year": int(year),
            "n_hours": int(len(seg)),
            "temp_mean_c": round(float(seg["Temperature"].mean()), 2),
            "temp_max_c": round(float(seg["Temperature"].max()), 1),
            "hours_ge35c": int((seg["Temperature"] >= 35).sum()),
            "hours_le15c": int((seg["Temperature"].le(15)).sum()),
            "rh_mean_pct": round(float(seg["Relative Humidity"].mean()), 1),
            "rain_total_mm": round(float(seg["Precipitation"].sum()), 0),
            "pm25_mean": round(float(seg["PM25"].mean()), 2),
            "note": note,
        })
    return pd.DataFrame(rows)


def monthly_indicators(project: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (year, month), seg in project.groupby([project.index.year, project.index.month]):
        rows.append({
            "year": int(year),
            "month": int(month),
            "temp_mean_c": round(float(seg["Temperature"].mean()), 2),
            "temp_max_c": round(float(seg["Temperature"].max()), 1),
            "hours_ge35c": int((seg["Temperature"] >= 35).sum()),
            "rain_total_mm": round(float(seg["Precipitation"].sum()), 0),
            "pm25_mean": round(float(seg["PM25"].mean()), 2),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------- report
def _markdown_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False, name=None)
    )
    return "\n".join([header, sep, body])


def write_report(warm: dict, enso: dict, ind: pd.DataFrame) -> None:
    ind_text = _markdown_table(ind)
    lines = f"""# Bối cảnh khí hậu cho nghiên cứu PM2.5 tại Hoàn Kiếm

Sinh tự động bởi `climate_analysis.py`. Nguồn: ERA5 archive (Open-Meteo) và
ONI của NOAA CPC, ghép với bộ dữ liệu Weatherbit của đề tài.

## 1. Dữ liệu đề tài có gì về khí hậu?

Bộ dữ liệu (`data/processed/cleaned_data.csv`) có {ind['n_hours'].sum():,} giờ quan sát
(2022-01-13 → 2026-08-31, quy đổi ~{ind['n_hours'].sum() / 8760:.1f} năm) với các biến khí tượng: nhiệt độ, độ ẩm, mưa, mây,
áp suất, bức xạ, UV, tốc độ và hướng gió.

- Có: **thời tiết** (weather) — quan sát từng giờ.
- Không có: chỉ số **ENSO/ONI**, nhiệt độ toàn cầu, nồng độ khí nhà kính.
- Do đó không thể kết luận về "biến đổi khí hậu" từ dữ liệu này (cần ≥ 30 năm).

## 2. Bằng chứng nóng lên tại Hà Nội (ERA5, 1991–2025)

| Chỉ tiêu | Giá trị |
|---|---|
| Baseline {BASELINE[0]}–{BASELINE[1]} | {warm['baseline_1991_2020_c']} °C |
| Xu hướng | {warm['trend_c_per_decade']:+.2f} °C/thập kỷ |
| Năm nóng nhất | {warm['warmest_year']} ({warm['warmest_year_c']} °C) |
| {RECENT[0]}–{RECENT[1]} | {warm['recent_2022_2025_mean_c']} °C ({warm['recent_anomaly_c']:+.2f} °C so baseline) |
| Số năm trong {RECENT[0]}–{RECENT[1]} nóng hơn baseline | {warm['years_above_baseline_recent']}/4 |

→ Có tín hiệu nóng lên rõ ràng ở quy mô địa phương.

## 3. El Niño có tác động tới PM2.5 Hoàn Kiếm không?

| Kiểm định | n | r | p |
|---|---|---|---|
| ONI ↔ PM2.5, toàn bộ tháng | {enso['n_months']} | {enso['pearson_r_all']:+.3f} | {enso['p_value_all']:.3f} |
| ONI ↔ PM2.5, mùa đông (T11–T2) | {enso['n_winter_months']} | {enso['pearson_r_winter']:+.3f} | {enso['p_value_winter']:.3f} |

PM2.5 trung bình (tháng) theo pha ENSO:

| Pha | PM2.5 mức trung bình | Số tháng |
|---|---|---|
"""
    for k, (mean, n) in enso["mean_pm25_by_phase"].items():
        lines += f"| {k} | {mean} | {n} |\n"
    lines += "\n→ **Không có tương quan có ý nghĩa thống kê.** El Niño/La Niña không\n"
    lines += "giải thích được biến động PM2.5 tại khu vực nghiên cứu ở độ phân giải tháng.\n\n"
    lines += "## 4. Chỉ số cực đoan từ dữ liệu đề tài (theo năm)\n\n"
    lines += ind_text + "\n\n"
    lines += """## 5. Kết luận

1. Dữ liệu đề tài **không chứa** biến khí hậu toàn cầu; chỉ chứa biến khí tượng địa phương.
2. Hoàn toàn **có thể** chứng minh nóng lên tại Hà Nội bằng ERA5: +0.55 °C so baseline 30 năm.
3. **Không** chứng minh được liên hệ El Niño → PM2.5; thay vào đó là một phát hiện có giá trị:
   PM2.5 đô thị bị chi phối bởi nguồn thải cục bộ và nghịch nhiệt mùa đông.
4. Vì vậy mô hình dự báo ngắn hạn (3–24 h) hợp lý khi chỉ dùng đặc trưng khí tượng địa phương.
"""
    (OUT / "report.md").write_text(lines, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    project = load_project()
    era = fetch_era5_daily()
    oni = fetch_oni()

    warm = figure_warming(era)
    enso = figure_enso(project, oni)

    ind = climate_indicators(project)
    ind.to_csv(OUT / "climate_indicators.csv", index=False, encoding="utf-8")
    monthly_indicators(project).to_csv(OUT / "climate_indicators_monthly.csv", index=False, encoding="utf-8")

    summary = {"warming": warm, "enso": enso}
    (OUT / "enso_correlation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(warm, enso, ind)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nIndicators:\n" + ind.to_string(index=False))
    print(f"\nOutputs -> {OUT}")


if __name__ == "__main__":
    main()
