# Kết quả phân tích — mục lục

Toàn bộ kết quả được nhóm theo **câu hỏi nghiên cứu**, không phải theo thời điểm tạo.
Thư mục `data/external/` chứa dữ liệu thô đã tải; `results/` chứa kết quả phân tích.

---

## 1. Dự báo PM2.5 (kết quả chính của đề tài)

| Thư mục | Nội dung |
|---|---|
| `results/feature_selection/` | Chọn đặc trưng, cấu hình feature |
| `results/baselines/` | Ridge, XGBoost 3/6/12/24h, so sánh với baseline |
| `results/05_gru/` | 4 kiến trúc GRU/TCN/Transformer + tuning + predictions |
| `results/stability/` | Walk-forward, seed stability, so sánh cặp |
| `03B_walkforward_analysis/` | Phân tích lỗi walk-forward theo tháng/horizon/peak |

**Kết quả chốt:** `results/05_gru/gru_vs_baselines.csv`, model tốt nhất `best_model_bundle/` (Seq2Seq GRU + Attention, MAE 5.53 / RMSE 9.63).

---

## 2. Đại diện không gian — "1 điểm có đại diện cả quận?"

| File | Nội dung |
|---|---|
| `results/external_check/summary.json` | Kết luận tổng hợp |
| `results/external_check/representativeness.csv` | MAE/tương quan giữa 7 điểm |
| `results/external_check/point_stats.csv` | Thống kê PM2.5 từng điểm |
| `results/external_check/compare_sources.csv` | Weatherbit vs CAMS cùng toạ độ |
| `results/external_check/grid_summary.json` | Độ phân giải lưới thực tế (11 km/ô) |
| `results/external_check/grid_probe.csv` | 208 điểm quét → 12 ô lưới |

**Kết luận:** 7 điểm trong quận cho kết quả giống hệt (r = 1.000, MAE = 0) vì cùng 1 ô lưới 11 km. Hai nguồn khác nhau lệch MAE 20.6 µg/m³.

---

## 3. Bối cảnh khí hậu — "biến đổi khí hậu / El Niño có trong dữ liệu?"

| File | Nội dung |
|---|---|
| `results/climate_review/report.md` | **Đọc file này trước** — báo cáo đầy đủ |
| `figure1_warming.png` | Nhiệt độ Hà Nội 1991–2025 + xu hướng + anomaly |
| `figure2_pm25_vs_oni.png` | PM2.5 với nền màu El Niño / La Niña |
| `figure3_scatter.png` | Scatter ONI vs PM2.5 (cả năm + mùa đông) |
| `enso_correlation.json` | Toàn bộ kiểm định thống kê |
| `climate_indicators.csv` | Chỉ số cực đoan theo năm |
| `climate_indicators_monthly.csv` | Chỉ số cực đoan theo tháng |

**Kết luận:** Nóng lên có thật (+0.55 °C so baseline 30 năm). El Niño **không** tương quan với PM2.5 (r = −0.09, p = 0.50).

---

## 4. Hướng gió & nguồn thải địa phương

| File | Nội dung |
|---|---|
| `results/wind_sector/wind_sector_pm25.png` | Biểu đồ tròn + heatmap hướng gió × mùa |
| `results/wind_sector/pm25_by_sector.csv` | PM2.5 theo 8 hướng gió |
| `results/wind_sector/pm25_by_sector_season.csv` | PM2.5 theo hướng × mùa |
| `results/wind_sector/summary.json` | Kết luận + thống kê gió lặng/mạnh |

**Kết luận:** Mùa đông gió Tây 59.4 vs Đông Bắc 39.4 µg/m³ (chênh 20.0). Lặng gió 38.7 vs gió mạnh 29.3 (−24%).

---

## 5. Nguồn thải bên ngoài (FIRMS, OSM)

| File | Nội dung |
|---|---|
| `results/external_drivers/summary.json` | Tổng hợp FIRMS + OSM |
| `results/external_drivers/osm_grid_1km.csv` | Mật độ đường/trạm xăng/công trình 9 ô 1 km |
| `results/external_drivers/feature_ablation.json` | So sánh MAE base vs extended feature |
| `data/external/firms/hanoi_box_hotspots_7d.csv` | 21 điểm cháy vùng Hà Nội (7 ngày) |

**Kết luận:** Mật độ đường chênh 6.6 lần trong 3 km. Thêm 8 feature mới giảm MAE 8.58% (10.70 → 9.78).

---

## 6. Slide báo cáo

| File | Nội dung |
|---|---|
| `sldie.html` | Slide 01 — Đặt vấn đề |
| `slide_climate.html` | Slide 02 — Bối cảnh khí hậu |

---

## Script tái tạo kết quả

| Script | Tạo ra |
|---|---|
| `train_final_model.py` | `best_model_bundle/`, `model_registry/` |
| `fetch_external_sources.py` | `data/external/open_meteo/` |
| `external_representativeness.py` | `results/external_check/` |
| `climate_analysis.py` | `results/climate_review/` |
| `wind_sector_analysis.py` | `results/wind_sector/` |
| `fetch_external_drivers.py` | `data/external/firms/`, `results/external_drivers/` |

## Lưu ý môi trường

- Python 3.13: có `pandas`, `matplotlib`, `scipy`, `requests`.
- Python 3.11: có thêm `scikit-learn`, `tensorflow`. Dùng 3.11 cho các script có model.
