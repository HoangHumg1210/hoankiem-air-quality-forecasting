# Thứ tự chạy notebook

## 5 bước chính

```
01_extract  →  02_merge  →  03_clean  →  04_eda  →  05_features  →  06_feature_selection
```

| # | Notebook | Cell | Làm gì | Đầu ra |
|---|---|---|---|---|
| 1 | `01_extract.ipynb` | 34 | Gọi Weatherbit API theo tháng, 13/01/2022 → 31/08/2026 | `data/raw/air_quality_weather_data.csv` |
| 2 | `02_merge.ipynb` | 22 | Ghép ERA5-Land, NASA POWER, CAMS, FIRMS | `data/processed/cleaned_data_v2.csv` |
| 3 | `03_clean.ipynb` | 33 | Khôi phục trục giờ, xử lý trùng, điền thiếu ngắn | `data/processed/cleaned_data.csv` |
| 4 | `04_eda.ipynb` | 116 | Khám phá, 29 biểu đồ, 22 đoạn nhận xét | Không tạo file |
| 5 | `05_features.ipynb` | 22 | **Feature engineering** từ insight của EDA | `data/processed/feature_data.csv`, `split_dates.csv` |
| 6 | `06_feature_selection.ipynb` | 52 | **Feature selection** bằng GRU nhỏ, 3 fold, 2 seed | `results/feature_selection/feature_config.json` |

## Mỗi feature ở bước 5 đến từ insight nào

| Insight (notebook 04) | Feature tạo ra (notebook 05) |
|---|---|
| Chu kỳ ngày, tuần, mùa rõ | `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos` |
| Gió Tây ô nhiễm hơn; gió yếu làm bụi tích tụ | `wind_sin/cos`, `wind_calm`, `wind_roll6` |
| PM10 gần đồng biến PM2.5 (0.95) | dùng PM10 trực tiếp + `PM10_PM25_ratio` |
| Áp suất cao/đổi nhanh = nghịch nhiệt | `Pressure_anom_24` |
| Mưa lớn mới rửa được bụi | `hours_since_rain`, `rain_24h_sum`, `Accumulated Hours of Rain` |
| Ngày lễ làm giao thông giảm | `IsHoliday` |

## Còn lại (chưa đổi tên)

| Notebook | Vai trò | Chạy sau |
|---|---|---|
| `03B_stability_weekend.ipynb` | Kiểm định `is_weekend` bằng paired test. **Kết quả đã có**: Gain −0.025, 0/6 cặp tốt hơn → loại bỏ | 06 |
| `05A_baselines.ipynb` | Ridge, XGBoost làm mốc | 06 |
| `05B_GRU_official.ipynb` | GRU chính thức | 06 |
| `04_model_calendar_decoder.ipynb` | 5 kiến trúc: GRU/LSTM/TCN/CNN-GRU/Transformer | 06 |
| `04_walkforward_error_analysis.ipynb` | Phân tích lỗi theo tháng, horizon, đỉnh | sau model |

## Đường dẫn

Mọi notebook dùng chung mẫu:

```python
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
```

Mở notebook từ thư mục `notebooks/` → `ROOT` là thư mục gốc dự án.
Chạy được cả local và Colab, không cần sửa đường dẫn.

## API key

`01_extract.ipynb` đọc từ biến môi trường:

```powershell
$env:WEATHERBIT_API_KEY = "..."
```

Không ghi key trực tiếp vào notebook.

## Bước 5 và 6 dùng bước 3 giờ

`05_features` tạo feature trên chuỗi 1 giờ. `06_feature_selection` chọn cửa sổ trong
`[24, 48, 72, 96]` giờ. Nếu đổi sang bước 3 giờ thì phải sửa cả `WINDOWS`,
`PRODUCTION_LAGS` và `step_hours` trong `best_model_bundle/config.json`.
