# Cấu trúc dự án — Hoàn Kiếm Air Quality Forecasting

```
hoankiem-air-quality-forecasting/
│
├── app.py                          # Streamlit demo dự báo
├── inference.py                    # Suy luận từ bundle
├── train_final_model.py            # Train model chính (cần Python 3.11 + TF)
├── bundle_registry.py              # Quản lý model bundle
├── promote_best_model.py           # Chọn model tốt nhất
├── backfill_bundle_train_range.py  # Nạp lại khoảng train
├── requirements.txt
│
├── src/
│   └── data_acquisition/           # Script lấy & phân tích dữ liệu ngoài
│       ├── fetch_external_sources.py       # Open-Meteo 7 điểm
│       ├── fetch_extra_datasets.py         # ERA5-Land, NASA POWER, CAMS, Ensemble
│       ├── fetch_firms_history.py          # FIRMS lịch sử (cần MAP_KEY)
│       ├── fetch_external_drivers.py       # FIRMS 7 ngày + OSM
│       ├── external_representativeness.py  # Kiểm tra 1 điểm có đại diện quận
│       ├── climate_analysis.py             # Nóng lên + El Niño
│       ├── wind_sector_analysis.py         # Hướng gió
│       └── feature_audit.py                # Phân loại feature dùng được / không
│
├── data/
│   ├── raw/                        # Dữ liệu gốc Weatherbit
│   ├── processed/
│   │   └── cleaned_data.csv        # 40 608 giờ × 20 cột (dữ liệu chính)
│   ├── submission/
│   └── external/                   # Dữ liệu lấy thêm (28 file, ~48 MB)
│       ├── open_meteo/             # 3 nguồn từ Open-Meteo + NOAA + NASA POWER
│       │   ├── cams_full_hoankiem_hourly.csv     CAMS 40 608h × 12
│       │   ├── era5land_hoankiem_hourly.csv      ERA5-Land 40 608h × 10
│       │   ├── nasa_power_hoankiem_hourly.csv    NASA POWER 40 608h × 13
│       │   ├── era5_hanoi_daily_1991_2025.csv    35 năm 12 784 ngày
│       │   ├── oni_noaa_1950_2025.csv            ONI 76 năm
│       │   ├── ensemble_forecast_spread.csv      51 thành viên × 168h
│       │   ├── airq_*.csv (7 file)               CAMS 7 điểm 2022-2024
│       │   ├── weather_*.csv (7 file)            Weather 7 điểm
│       │   ├── points.json                       7 toạ độ điểm
│       │   └── manifest.json
│       └── firms/                  # NASA FIRMS + OpenStreetMap
│           ├── hanoi_fires_2022_2026.csv         66 208 điểm cháy lịch sử
│           ├── hanoi_fire_daily_features.csv     Feature ngày đã tính
│           ├── osm_grid_1km.csv                  9 ô 1km hạ tầng
│           └── raw/                               Dữ liệu thô
│               ├── osm_cells.csv
│               └── snapshot_7d/                   3 vệ tinh × 7 ngày
│                   ├── hanoi_box_hotspots_7d.csv   21 điểm
│                   ├── hanoi_daily_hotspots_7d.csv
│                   ├── viirs_snpp_7d.csv
│                   ├── viirs_noaa20_7d.csv
│                   └── modis_7d.csv
│
├── results/
│   ├── README.md                   # MỤC LỤC (đọc đầu tiên)
│   ├── 05_gru/                     # Model chính: 4 kiến trúc GRU
│   ├── baselines/                  # Ridge, XGBoost
│   ├── feature_selection/          # Chọn đặc trưng
│   ├── stability/                  # Walk-forward, seed
│   ├── external_check/             # Đại diện không gian + kiểm tra chéo
│   ├── climate_review/             # Khí hậu + El Niño (3 biểu đồ)
│   ├── wind_sector/                # Hướng gió × mùa
│   ├── external_drivers/           # FIRMS, OSM, ablation
│   └── feature_audit/              # Phân loại feature
│
├── best_model_bundle/              # Model tốt nhất (GRU, MAE 5.53)
├── model_registry/                 # Các model đã train
├── notebooks/                      # EDA, model thử nghiệm
├── tests/                          # Unit test
└── reports/                        # Hình cho báo cáo
```

## Tác dụng từng nhóm file

| Nhóm | Tác dụng | Cần gì để chạy |
|---|---|---|
| `train_final_model.py` | Train model chính | Python 3.11 + TensorFlow |
| `app.py` + `inference.py` | Demo dự báo | Python 3.11 + TF (đọc bundle) |
| `src/data_acquisition/fetch_*.py` | Tải dữ liệu ngoài | Python 3.13 + requests |
| `src/data_acquisition/*_analysis.py` | Phân tích & kiểm định | Python 3.13 + pandas/matplotlib/scipy |
| `src/data_acquisition/feature_audit.py` | Audit feature | Python 3.13 |

## Hai môi trường Python

| Môi trường | Có gì | Dùng cho |
|---|---|---|
| Python 3.11 | pandas, sklearn, tensorflow | `train_final_model.py`, ablation |
| Python 3.13 | pandas, matplotlib, scipy, requests | Tất cả script phân tích |

## Quy ước đặt tên data

| Tiền tố | Ý nghĩa |
|---|---|
| `airq_*` | Chất lượng không khí (CAMS) |
| `weather_*` | Khí tượng (ERA5) |
| `era5land_*` | ERA5-Land (độ ẩm đất, lớp kấy xáo) |
| `nasa_power_*` | NASA POWER (đối chiếu khí tượng) |
| `cams_full_*` | CAMS phủ đủ 2022–2026 |
| `hanoi_fires_*` | Điểm cháy FIRMS |
| `osm_*` | Hạ tầng OpenStreetMap |
| `oni_*` | Chỉ số El Niño NOAA |
| `ensemble_*` | Dự báo tổ hợp |

## Chạy lại toàn bộ phân tích

```powershell
python src/data_acquisition/fetch_external_sources.py
python src/data_acquisition/fetch_extra_datasets.py
python src/data_acquisition/fetch_firms_history.py <FIRMS_MAP_KEY>
python src/data_acquisition/external_representativeness.py
python src/data_acquisition/climate_analysis.py
python src/data_acquisition/wind_sector_analysis.py
python src/data_acquisition/feature_audit.py
```
