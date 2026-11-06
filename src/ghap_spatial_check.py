import argparse
from pathlib import Path

import pandas as pd


def find_root():
    for p in Path(__file__).resolve().parents:
        if (p / "data").is_dir() and (p / "notebooks").is_dir():
            return p
    return Path.cwd()


ROOT = find_root()
RAW = ROOT / "data" / "external" / "ghap"
OUT = ROOT / "results" / "ghap_check"
WEATHERBIT = ROOT / "data" / "processed" / "cleaned_data.csv"
BBOX = [105.840, 21.015, 105.865, 21.043]   # khung bao gần đúng, dùng khi không có ranh giới quận


# ---------------------------------------------------------------- 1. Tải dữ liệu
def fetch(project):
    import ee
    ee.Initialize(project=project)

    gaul = (ee.FeatureCollection("FAO/GAUL/2015/level2")
            .filter(ee.Filter.eq("ADM0_NAME", "Viet Nam"))
            .filter(ee.Filter.stringContains("ADM2_NAME", "Hoan Kiem")))
    district = gaul.geometry() if gaul.size().getInfo() > 0 else ee.Geometry.Rectangle(BBOX)
    ghap = ee.ImageCollection("projects/sat-io/open-datasets/GHAP/GHAP_D1K_PM25")

    def sample(img):
        img = ee.Image(img)
        date = img.date().format("YYYY-MM-dd")
        pts = img.select([0], ["pm25"]).sample(region=district, projection=img.projection(),
                                               geometries=True)
        return pts.map(lambda f: f.set("date", date))

    RAW.mkdir(parents=True, exist_ok=True)
    for month in pd.date_range("2017-01", "2022-12", freq="MS"):
        path = RAW / f"ghap_{month:%Y_%m}.csv"
        if path.exists():
            continue
        end = month + pd.offsets.MonthBegin()
        feats = (ghap.filterDate(f"{month:%Y-%m-%d}", f"{end:%Y-%m-%d}")
                 .map(sample).flatten().getInfo()["features"])
        pd.DataFrame([{"date": f["properties"]["date"],
                       "lon": f["geometry"]["coordinates"][0],
                       "lat": f["geometry"]["coordinates"][1],
                       "pm25": f["properties"]["pm25"]} for f in feats]).to_csv(path, index=False)
        print(f"{month:%Y-%m}: {len(feats)} giá trị")


# ---------------------------------------------------------------- 2. Phân tích
def load():
    df = pd.concat([pd.read_csv(f) for f in sorted(RAW.glob("ghap_*.csv"))], ignore_index=True)
    # File tải bằng bản script cũ có cả các ô quanh quận: chỉ giữ ô trong quận
    if "in_district" in df.columns:
        df = df[df["in_district"] == 1]
    df = df.rename(columns={"pm25_raw": "pm25"})
    df["date"] = pd.to_datetime(df["date"])
    if df["pm25"].median() > 150:          # GHAP lưu PM2.5 x 10
        df["pm25"] = df["pm25"] * 0.1
    df["o"] = df["lat"].round(3).astype(str) + "," + df["lon"].round(3).astype(str)
    return df.pivot_table(index="date", columns="o", values="pm25")   # hàng = ngày, cột = ô 1 km


def stats(wide):
    day_mean = wide.mean(axis=1)
    spatial_var = wide.sub(day_mean, axis=0).pow(2).stack().mean()
    total_var = wide.stack().var(ddof=0)
    corr = wide.corr()
    return {
        "Số ngày": len(wide),
        "PM2.5 TB": day_mean.mean(),
        "Dao động theo ngày (độ lệch chuẩn)": day_mean.std(),
        "Chênh giữa các ô trong ngày (độ lệch chuẩn, median)": wide.std(axis=1).median(),
        "Phần biến động do khác biệt giữa các ô (%)": 100 * spatial_var / total_var,
        "Tương quan giữa các ô (thấp nhất)": corr.where(corr < 0.99999).min().min(),
    }


def compare_weatherbit(day_mean):
    wb = pd.read_csv(WEATHERBIT, usecols=["Local Time", "PM25"], parse_dates=["Local Time"])
    utc_day = (wb["Local Time"] - pd.Timedelta(hours=7)).dt.normalize()   # GHAP tính theo ngày UTC
    wb = wb.groupby(utc_day)["PM25"].mean()
    return pd.concat([wb.rename("Weatherbit"), day_mean.rename("GHAP")], axis=1).dropna()


def analyze():
    wide = load()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"{wide.shape[1]} ô 1 km trong quận, {wide.index.min():%Y-%m-%d} → {wide.index.max():%Y-%m-%d}")

    table = pd.DataFrame({year: stats(g) for year, g in wide.groupby(wide.index.year)})
    table["Tất cả"] = pd.Series(stats(wide))
    table = table.astype(object)                 # hàng = chỉ số, cột = năm
    table.loc["Số ngày"] = table.loc["Số ngày"].astype(int)
    table.loc[table.index != "Số ngày"] = table.loc[table.index != "Số ngày"].astype(float).round(3)
    table.index.name = "Chỉ số"
    table.to_csv(OUT / "ket_qua.csv", encoding="utf-8-sig")
    print("\nĐộ đồng đều trong quận theo năm:")
    print(table.to_string())

    pair = compare_weatherbit(wide.mean(axis=1)) if WEATHERBIT.exists() else pd.DataFrame()
    if len(pair) > 30:
        err = pair["Weatherbit"] - pair["GHAP"]
        cmp = pd.Series({"Số ngày so sánh": len(pair),
                         "Tương quan": pair.corr().iloc[0, 1],
                         "Weatherbit thấp (-) / cao (+) hơn GHAP (µg/m³)": err.mean(),
                         "MAE (µg/m³)": err.abs().mean()}, dtype=object).map(
                             lambda v: v if isinstance(v, int) else round(v, 3))
        cmp.to_csv(OUT / "weatherbit_vs_ghap.csv", header=["Giá trị"], encoding="utf-8-sig")
        print("\nWeatherbit so với GHAP (trung bình ngày):")
        print(cmp.to_string())

    plot(wide, pair)
    print(f"\nĐã lưu vào {OUT}")


def plot(wide, pair):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    year = wide.index.year.max()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    wide[wide.index.year == year].plot(ax=axes[0], legend=False, linewidth=0.8)
    axes[0].set_title(f"PM2.5 từng ô 1 km trong quận, năm {year}")
    axes[0].set_ylabel("µg/m³")
    axes[0].set_xlabel("")

    if len(pair):
        axes[1].scatter(pair["GHAP"], pair["Weatherbit"], s=8, alpha=0.5)
        top = pair.max().max()
        axes[1].plot([0, top], [0, top], color="black", linewidth=1)
        axes[1].set_xlabel("GHAP (µg/m³)")
        axes[1].set_ylabel("Weatherbit (µg/m³)")
        axes[1].set_title("Trung bình ngày: Weatherbit so với GHAP (đường chéo = khớp)")
    else:
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "ket_qua.png", dpi=130)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["fetch", "analyze"])
    parser.add_argument("--project")
    args = parser.parse_args()
    fetch(args.project) if args.cmd == "fetch" else analyze()