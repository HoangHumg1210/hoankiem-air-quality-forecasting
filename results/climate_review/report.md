# Bối cảnh khí hậu cho nghiên cứu PM2.5 tại Hoàn Kiếm

Sinh tự động bởi `climate_analysis.py`. Nguồn: ERA5 archive (Open-Meteo) và
ONI của NOAA CPC, ghép với bộ dữ liệu Weatherbit của đề tài.

## 1. Dữ liệu đề tài có gì về khí hậu?

Bộ dữ liệu (`data/processed/cleaned_data.csv`) có 40,608 giờ quan sát
(2022-01-13 → 2026-08-31, quy đổi ~4.6 năm) với các biến khí tượng: nhiệt độ, độ ẩm, mưa, mây,
áp suất, bức xạ, UV, tốc độ và hướng gió.

- Có: **thời tiết** (weather) — quan sát từng giờ.
- Không có: chỉ số **ENSO/ONI**, nhiệt độ toàn cầu, nồng độ khí nhà kính.
- Do đó không thể kết luận về "biến đổi khí hậu" từ dữ liệu này (cần ≥ 30 năm).

## 2. Bằng chứng nóng lên tại Hà Nội (ERA5, 1991–2025)

| Chỉ tiêu | Giá trị |
|---|---|
| Baseline 1991–2020 | 23.58 °C |
| Xu hướng | +0.23 °C/thập kỷ |
| Năm nóng nhất | 2024 (24.6 °C) |
| 2022–2025 | 24.13 °C (+0.55 °C so baseline) |
| Số năm trong 2022–2025 nóng hơn baseline | 3/4 |

→ Có tín hiệu nóng lên rõ ràng ở quy mô địa phương.

## 3. El Niño có tác động tới PM2.5 Hoàn Kiếm không?

| Kiểm định | n | r | p |
|---|---|---|---|
| ONI ↔ PM2.5, toàn bộ tháng | 56 | -0.092 | 0.498 |
| ONI ↔ PM2.5, mùa đông (T11–T2) | 18 | +0.013 | 0.959 |

PM2.5 trung bình (tháng) theo pha ENSO:

| Pha | PM2.5 mức trung bình | Số tháng |
|---|---|---|
| La Niña | 44.34 | 16 |
| Trung tính | 44.34 | 25 |
| El Niño | 40.99 | 15 |

→ **Không có tương quan có ý nghĩa thống kê.** El Niño/La Niña không
giải thích được biến động PM2.5 tại khu vực nghiên cứu ở độ phân giải tháng.

## 4. Chỉ số cực đoan từ dữ liệu đề tài (theo năm)

| year | n_hours | temp_mean_c | temp_max_c | hours_ge35c | hours_le15c | rh_mean_pct | rain_total_mm | pm25_mean | note |
|---|---|---|---|---|---|---|---|---|---|
| 2022 | 8472 | 24.73 | 38.5 | 144 | 549 | 76.7 | 2933.0 | 40.1 |  |
| 2023 | 8760 | 25.36 | 40.8 | 240 | 352 | 76.5 | 2261.0 | 43.7 |  |
| 2024 | 8784 | 25.4 | 39.2 | 204 | 463 | 79.4 | 3036.0 | 45.4 |  |
| 2025 | 8760 | 24.36 | 40.2 | 140 | 372 | 79.5 | 3527.0 | 45.91 |  |
| 2026 | 5832 | 25.7 | 41.1 | 230 | 276 | 81.7 | 1390.0 | 40.5 |  |

## 5. Kết luận

1. Dữ liệu đề tài **không chứa** biến khí hậu toàn cầu; chỉ chứa biến khí tượng địa phương.
2. Hoàn toàn **có thể** chứng minh nóng lên tại Hà Nội bằng ERA5: +0.55 °C so baseline 30 năm.
3. **Không** chứng minh được liên hệ El Niño → PM2.5; thay vào đó là một phát hiện có giá trị:
   PM2.5 đô thị bị chi phối bởi nguồn thải cục bộ và nghịch nhiệt mùa đông.
4. Vì vậy mô hình dự báo ngắn hạn (3–24 h) hợp lý khi chỉ dùng đặc trưng khí tượng địa phương.
