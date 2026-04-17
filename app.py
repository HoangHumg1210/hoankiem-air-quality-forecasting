from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bundle_registry import load_registry_metrics as load_bundle_registry_metrics
from inference import (
    DEFAULT_TIMESTAMP_COL,
    forecast_recursive,
    keras_archive_contains_lambda,
    load_model_bundle,
    prepare_raw_frame,
)


APP_DIR = Path(__file__).resolve().parent
REGISTRY_DIR = APP_DIR / "model_registry"
BEST_BUNDLE_DIR = APP_DIR / "best_model_bundle"
RAW_DATA_PATH = APP_DIR / "data" / "processed" / "data2225_done.csv"
FUTURE_LOOKBACK_DAYS = 14
DEFAULT_HISTORY_WINDOW = 24


# AQI config clean override for UI/runtime.
AQI_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 500.0, 301, 500),
]

AQI_LEVELS = [
    {"range": (0, 50), "label": "Tốt", "color": "#63B32E", "bg": "#EAF5E1"},
    {"range": (51, 100), "label": "Trung bình", "color": "#CDBB25", "bg": "#F8F2D8"},
    {"range": (101, 150), "label": "Nhạy cảm", "color": "#D98A1F", "bg": "#F8E7D2"},
    {"range": (151, 200), "label": "Xấu", "color": "#CC4B4C", "bg": "#F6DCDD"},
    {"range": (201, 300), "label": "Rất Xấu", "color": "#7A4A38", "bg": "#EADFD7"},
    {"range": (301, 500), "label": "Nguy hiểm", "color": "#70253A", "bg": "#E8D8DE"},
]


def aqi_from_pm25(value: float) -> int:
    pm25 = float(np.clip(value, 0.0, AQI_BREAKPOINTS[-1][1]))
    for c_low, c_high, aqi_low, aqi_high in AQI_BREAKPOINTS:
        if c_low <= pm25 <= c_high:
            ratio = (pm25 - c_low) / (c_high - c_low) if c_high > c_low else 0.0
            return int(round(aqi_low + ratio * (aqi_high - aqi_low)))
    return AQI_BREAKPOINTS[-1][3]


def band_info(value: float) -> tuple[str, str, str]:
    aqi_value = aqi_from_pm25(value)
    for level in AQI_LEVELS:
        low, high = level["range"]
        if low <= aqi_value <= high:
            color = str(level["color"])
            bg = str(level.get("bg", f"{color}22"))
            return str(level["label"]), color, bg
    return "Nguy hiểm", "#7E0023", "#7E002322"


def _state_get(container, key: str, default=None):
    if container is None:
        return default
    if hasattr(container, "get"):
        return container.get(key, default)
    return getattr(container, key, default)


def get_selected_chart_point(chart_state, history_df: pd.DataFrame, forecast_df: pd.DataFrame) -> dict | None:
    selection = _state_get(chart_state, "selection", chart_state)
    points = _state_get(selection, "points", [])
    if not points:
        return None

    point = points[-1]
    curve_number = _state_get(point, "curve_number", _state_get(point, "curveNumber"))
    point_index = _state_get(point, "point_index", _state_get(point, "pointIndex"))
    if curve_number is None or point_index is None:
        return None

    curve_number = int(curve_number)
    point_index = int(point_index)

    if curve_number == 0 and 0 <= point_index < len(history_df):
        row = history_df.iloc[point_index]
        return {
            "source": "actual",
            "source_label": "Giá trị thực tế",
            "timestamp": pd.to_datetime(row[DEFAULT_TIMESTAMP_COL]),
            "value": float(row["PM25"]),
        }

    if curve_number == 1 and 0 <= point_index < len(forecast_df):
        row = forecast_df.iloc[point_index]
        return {
            "source": "forecast",
            "source_label": "Dự báo",
            "timestamp": pd.to_datetime(row[DEFAULT_TIMESTAMP_COL]),
            "value": float(row["y_pred"]),
        }

    if curve_number == 2 and "y_true" in forecast_df.columns and 0 <= point_index < len(forecast_df):
        row = forecast_df.iloc[point_index]
        if pd.notna(row.get("y_true")):
            return {
                "source": "future_actual",
                "source_label": "Thực tế tương lai",
                "timestamp": pd.to_datetime(row[DEFAULT_TIMESTAMP_COL]),
                "value": float(row["y_true"]),
            }

    return None


def get_selected_model_from_mae_chart(chart_state, metrics_df: pd.DataFrame) -> str | None:
    selection = _state_get(chart_state, "selection", chart_state)
    points = _state_get(selection, "points", [])
    if not points:
        return None

    point = points[-1]
    model_name = _state_get(point, "x")
    if model_name is not None:
        model_name = str(model_name)
        if model_name in set(metrics_df["Model"].astype(str)):
            return model_name

    point_index = _state_get(point, "point_index", _state_get(point, "pointIndex"))
    if point_index is None:
        return None

    chart_df = metrics_df.sort_values("MAE", na_position="last").reset_index(drop=True)
    point_index = int(point_index)
    if 0 <= point_index < len(chart_df):
        return str(chart_df.iloc[point_index]["Model"])
    return None


def load_registry_metrics() -> pd.DataFrame:
    return load_bundle_registry_metrics(
        APP_DIR,
        registry_dir=REGISTRY_DIR,
        best_bundle_dir=BEST_BUNDLE_DIR,
    )


def load_raw_data() -> pd.DataFrame:
    raw_df = pd.read_csv(RAW_DATA_PATH)
    raw_df[DEFAULT_TIMESTAMP_COL] = pd.to_datetime(raw_df[DEFAULT_TIMESTAMP_COL])
    return raw_df.sort_values(DEFAULT_TIMESTAMP_COL).reset_index(drop=True)


def load_bundle_timeline(bundle_dir: str | Path) -> pd.DataFrame:
    timeline_path = Path(bundle_dir)
    if not timeline_path.is_absolute():
        timeline_path = APP_DIR / timeline_path
    csv_path = timeline_path / "test_timeline.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    timeline_df = pd.read_csv(csv_path)
    if "timestamp" in timeline_df.columns:
        timeline_df["timestamp"] = pd.to_datetime(timeline_df["timestamp"])
    return timeline_df


def resolve_bundle_dir(metrics_df: pd.DataFrame, model_name: str) -> Path:
    row = metrics_df.loc[metrics_df["Model"] == model_name].iloc[0]
    return APP_DIR / row["Bundle Dir"]


def is_forecast_runtime_supported(bundle_dir: str | Path) -> bool:
    bundle_path = Path(bundle_dir)
    if not bundle_path.is_absolute():
        bundle_path = APP_DIR / bundle_path

    return not keras_archive_contains_lambda(bundle_path / "model.keras")


def get_best_supported_forecast_model(metrics_df: pd.DataFrame) -> str | None:
    ranked = metrics_df.sort_values("MAE", na_position="last").reset_index(drop=True)
    for _, row in ranked.iterrows():
        bundle_dir = APP_DIR / row["Bundle Dir"]
        if is_forecast_runtime_supported(bundle_dir):
            return str(row["Model"])
    return None


def load_bundle_runtime_limits(bundle_dir: str | Path) -> tuple[int, int]:
    bundle_path = Path(bundle_dir)
    if not bundle_path.is_absolute():
        bundle_path = APP_DIR / bundle_path

    config_path = bundle_path / "config.json"
    if not config_path.exists():
        return 24, 3

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 24, 3

    rollout_horizon = max(int(config.get("rollout_horizon", 24)), 1)
    step_hours = max(int(config.get("step_hours", 3)), 1)
    return rollout_horizon, step_hours


def _coerce_binary_like(value: float, source_series: pd.Series) -> float:
    valid_values = {float(item) for item in source_series.dropna().unique()}
    if valid_values.issubset({0.0, 1.0}):
        return float(int(round(float(value))))
    return float(value)


def _estimate_future_value(
    source_series: pd.Series,
    target_ts: pd.Timestamp,
    *,
    step_hours: int,
) -> float:
    if source_series.dropna().empty:
        return 0.0

    step_delta = pd.Timedelta(hours=step_hours)
    steps_per_day = max(int(round(24 / step_hours)), 1)
    candidate_steps = [steps_per_day, steps_per_day * 2, steps_per_day * 3, steps_per_day * 7]

    for offset_steps in candidate_steps:
        candidate_ts = target_ts - (offset_steps * step_delta)
        if candidate_ts in source_series.index:
            candidate_value = source_series.loc[candidate_ts]
            if pd.notna(candidate_value):
                return _coerce_binary_like(float(candidate_value), source_series)

    recent_window = source_series.dropna().tail(max(FUTURE_LOOKBACK_DAYS * steps_per_day, steps_per_day * 7))
    if recent_window.empty:
        return 0.0

    same_weekday_hour = recent_window[
        (recent_window.index.dayofweek == target_ts.dayofweek)
        & (recent_window.index.hour == target_ts.hour)
    ]
    if not same_weekday_hour.empty:
        return _coerce_binary_like(float(same_weekday_hour.median()), source_series)

    same_hour = recent_window[recent_window.index.hour == target_ts.hour]
    if not same_hour.empty:
        return _coerce_binary_like(float(same_hour.median()), source_series)

    return _coerce_binary_like(float(recent_window.median()), source_series)


def build_future_covariates_frame(prepared_history: pd.DataFrame, bundle, horizon_steps: int) -> pd.DataFrame:
    if horizon_steps <= 0:
        raise ValueError("Horizon steps must be greater than zero.")

    step_delta = pd.Timedelta(hours=bundle.step_hours)
    future_index = pd.date_range(
        start=prepared_history.index.max() + step_delta,
        periods=horizon_steps,
        freq=step_delta,
    )

    future_frame = pd.DataFrame(index=future_index)
    required_future_cols = [col for col in bundle.required_raw_columns if col != "PM25"]

    for col in required_future_cols:
        if col not in prepared_history.columns:
            raise ValueError(f"History is missing required raw column for future extrapolation: {col}")

        source_series = prepared_history[col].astype(float)
        future_frame[col] = [
            _estimate_future_value(source_series, timestamp, step_hours=bundle.step_hours)
            for timestamp in future_index
        ]

    future_frame["PM25"] = np.nan
    future_frame.index.name = DEFAULT_TIMESTAMP_COL
    return future_frame.reset_index()


def load_latest_forecast(bundle_dir: Path, horizon_steps: int) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    bundle = load_model_bundle(bundle_dir)
    raw_df = load_raw_data()
    prepared = prepare_raw_frame(raw_df, step_hours=bundle.step_hours)

    if len(prepared) <= bundle.lookback:
        raise ValueError("Không đủ dữ liệu lịch sử để chạy forecast cho bundle đã chọn.")

    max_steps = min(horizon_steps, bundle.rollout_horizon)
    future_frame = build_future_covariates_frame(prepared, bundle, max_steps)

    history_frame = prepared.copy()

    forecast_df = forecast_recursive(
        bundle,
        history_frame.reset_index(),
        future_frame,
        horizon=max_steps,
        timestamp_col=DEFAULT_TIMESTAMP_COL,
    )

    history_plot_df = history_frame.tail(min(14, len(history_frame))).reset_index()
    return history_plot_df, forecast_df, bundle.step_hours


def load_latest_forecast_or_timeline(bundle_dir: Path, horizon_steps: int) -> tuple[pd.DataFrame, pd.DataFrame, int, str]:
    history_df, forecast_df, step_hours = load_latest_forecast(bundle_dir, horizon_steps)
    return history_df, forecast_df, step_hours, "forecast_ngoai_du_lieu"


def make_forecast_chart(history_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    zones = [
        (0, 30, "Tốt", "rgba(107, 221, 0, 0.30)", "#58A81C"),
        (30, 60, "Trung bình", "rgba(240, 222, 140, 0.32)", "#B49B12"),
        (60, 90, "Nhạy cảm", "rgba(235, 191, 126, 0.34)", "#D37D18"),
        (90, 120, "Không tốt", "rgba(230, 163, 168, 0.34)", "#C54B57"),
        (120, 250, "Rất xấu", "rgba(191, 163, 146, 0.34)", "#8A5B45"),
        (250, 500, "Nguy hiểm", "rgba(171, 136, 150, 0.34)", "#7A2E4A"),
    ]

    y_values = pd.concat(
        [
            history_df["PM25"],
            forecast_df["y_pred"],
            forecast_df["y_true"] if "y_true" in forecast_df.columns else pd.Series(dtype=float),
        ],
        ignore_index=True,
    ).dropna()
    observed_max = float(y_values.max()) if not y_values.empty else 80.0
    y_max = max(90.0, observed_max + 18.0)
    for bound in [90.0, 120.0, 150.0, 200.0, 250.0, 300.0, 500.0]:
        if y_max <= bound:
            y_max = bound
            break

    visible_zones = [(low, min(high, y_max), label, fill, text_color) for low, high, label, fill, text_color in zones if low < y_max]
    for lower, upper, _, fill, _ in visible_zones:
        fig.add_hrect(y0=lower, y1=upper, fillcolor=fill, line_width=0)

    fig.add_trace(
        go.Scatter(
            x=history_df[DEFAULT_TIMESTAMP_COL],
            y=history_df["PM25"],
            mode="lines+markers",
            name="Giá trị thực tế",
            line=dict(color="#3C82D7", width=2.6),
            marker=dict(size=6, color="#3C82D7"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df[DEFAULT_TIMESTAMP_COL],
            y=forecast_df["y_pred"],
            mode="lines+markers",
            name="Dự báo",
            line=dict(color="#21B07A", width=2.6, dash="dot"),
            marker=dict(size=6, color="#21B07A"),
        )
    )

    if "y_true" in forecast_df.columns and forecast_df["y_true"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=forecast_df[DEFAULT_TIMESTAMP_COL],
                y=forecast_df["y_true"],
                mode="lines+markers",
                name="Thực tế tương lai",
                line=dict(color="#EF4444", width=2),
                marker=dict(size=6, color="#EF4444"),
            )
        )

    current_x = history_df[DEFAULT_TIMESTAMP_COL].iloc[-1]
    peak = forecast_df.loc[forecast_df["y_pred"].idxmax()]
    fig.add_vline(x=current_x, line_dash="dot", line_color="rgba(76, 139, 245, 0.85)", line_width=1.6)
    fig.add_annotation(
        x=current_x,
        y=y_max - 6,
        text="Hiện tại",
        showarrow=False,
        bgcolor="#EAF2FF",
        bordercolor="#BCD1F7",
        borderwidth=1,
        font=dict(size=10, color="#3F79D8"),
    )
    fig.add_annotation(
        x=peak[DEFAULT_TIMESTAMP_COL],
        y=peak["y_pred"] + 6,
        text=f"{peak[DEFAULT_TIMESTAMP_COL].strftime('%d/%m %H:%M')}<br><b>{peak['y_pred']:.1f} µg/m³</b>",
        showarrow=True,
        arrowcolor="#21B07A",
        bgcolor="#ECFFF7",
        bordercolor="#9EE8CB",
        borderwidth=1,
        font=dict(size=11, color="#0D8A5E"),
    )

    for lower, upper, label, _, text_color in visible_zones:
        fig.add_annotation(
            xref="paper",
            x=0.01,
            y=(lower + upper) / 2,
            yref="y",
            text=label,
            showarrow=False,
            xanchor="left",
            font=dict(size=12, color=text_color),
        )

    fig.update_layout(
        height=360,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.10, x=0, bgcolor="rgba(255,255,255,0.0)"),
        xaxis=dict(title="", showgrid=False, tickformat="%d/%m %H:%M"),
        yaxis=dict(
            title="PM2.5 (µg/m³)",
            gridcolor="rgba(0,0,0,0.05)",
            range=[0, y_max],
        ),
    )
    return fig


def make_mae_chart(metrics_df: pd.DataFrame, selected_model: str) -> go.Figure:
    chart_df = metrics_df.sort_values("MAE", na_position="last").reset_index(drop=True)
    colors = ["#67C9A8" if model == selected_model else "#8FB5F9" for model in chart_df["Model"]]
    text_colors = ["#22A06B" if model == selected_model else "#3F79D8" for model in chart_df["Model"]]
    y_max = float(chart_df["MAE"].max()) if not chart_df.empty else 1.0
    fig = go.Figure(
        go.Bar(
            x=chart_df["Model"],
            y=chart_df["MAE"],
            text=chart_df["MAE"].round(2),
            textposition="outside",
            cliponaxis=False,
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.6)", width=1)),
            textfont=dict(color=text_colors, size=15),
            hovertemplate="<b>%{x}</b><br>MAE: %{y:.2f} µg/m³<extra></extra>",
            selectedpoints=[idx for idx, model in enumerate(chart_df["Model"]) if model == selected_model],
            selected=dict(marker=dict(opacity=1.0)),
            unselected=dict(marker=dict(opacity=0.92)),
        )
    )
    fig.update_layout(
        height=430,
        margin=dict(l=8, r=8, t=28, b=96),
        paper_bgcolor="white",
        plot_bgcolor="white",
        clickmode="event+select",
        xaxis=dict(title="", tickfont=dict(size=12), tickangle=32, automargin=True),
        yaxis=dict(title="", gridcolor="rgba(63,121,216,0.12)", range=[0, y_max * 1.18], automargin=True),
    )
    return fig


def make_backtest_chart(timeline_df: pd.DataFrame, selected_model: str) -> go.Figure:
    plot_df = timeline_df.sort_values("timestamp").copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp"],
            y=plot_df["y_true"],
            mode="lines+markers",
            name="Thực tế",
            line=dict(color="#378ADD", width=3),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp"],
            y=plot_df["y_pred"],
            mode="lines+markers",
            name=selected_model,
            line=dict(color="#1D9E75", width=3, dash="dot"),
            marker=dict(size=6),
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        xaxis=dict(title="", tickformat="%d/%m %H:%M"),
        yaxis=dict(title="PM2.5 (µg/m³)", gridcolor="rgba(0,0,0,0.05)"),
    )
    return fig


def metric_card(
    title: str,
    value: str,
    badge: str,
    accent: str,
    badge_bg: str,
    *,
    icon: str = "•",
    footer: str = "",
) -> None:
    st.markdown(
        f"""
        <div class="metric-card" style="--metric-accent:{accent}; --metric-badge-bg:{badge_bg}; border-left-color:{accent}">
            <div class="metric-head">
                <div class="metric-icon" style="background:{badge_bg}; color:{accent}">{escape(icon)}</div>
                <div class="metric-title">{title}</div>
            </div>
            <div class="metric-value" style="color:{accent}">{value}</div>
            <div class="metric-foot">
                <div class="metric-badge" style="background:{badge_bg}; color:{accent}">{badge}</div>
                <div class="metric-footer">{footer}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chip_card(time_text: str, value: float) -> None:
    label, color, bg = band_info(value)
    st.markdown(
        f"""
        <div class="hour-chip">
            <div class="hour-time">{time_text}</div>
            <div class="hour-value" style="color:{color}">{value:.1f}</div>
            <div class="hour-badge" style="background:{bg}; color:{color}">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_warning_html(forecast_df: pd.DataFrame, current_val: float, step_hours: int) -> str:
    peak_position = int(forecast_df["y_pred"].to_numpy().argmax())
    peak_row = forecast_df.iloc[peak_position]
    peak_label, _, _ = band_info(float(peak_row["y_pred"]))

    labels = forecast_df["y_pred"].apply(lambda value: band_info(float(value))[0]).tolist()
    start_position = peak_position
    end_position = peak_position

    while start_position > 0 and labels[start_position - 1] == peak_label:
        start_position -= 1
    while end_position < len(labels) - 1 and labels[end_position + 1] == peak_label:
        end_position += 1

    start_time_text = forecast_df.iloc[start_position][DEFAULT_TIMESTAMP_COL].strftime("%H:%M")
    end_time_text = forecast_df.iloc[end_position][DEFAULT_TIMESTAMP_COL].strftime("%H:%M")
    peak_time_text = peak_row[DEFAULT_TIMESTAMP_COL].strftime("%H:%M")

    window_steps = max(1, int(round(6 / step_hours)))
    six_hour_value = float(forecast_df["y_pred"].head(window_steps).iloc[-1])
    delta = six_hour_value - current_val
    if delta > 1.0:
        trend_text = "tăng"
    elif delta < -1.0:
        trend_text = "giảm"
    else:
        trend_text = "dao động nhẹ"

    advice_map = {
        "Tốt": [
            ("✅", "Chất lượng không khí đang ở mức an toàn cho các hoạt động ngoài trời."),
            ("🌿", "Tiếp tục theo dõi chỉ số theo khung giờ cao điểm để phát hiện biến động sớm."),
            ("💧", "Giữ không gian sống thông thoáng và vệ sinh khu vực có bụi mịn."),
        ],
        "Trung bình": [
            ("🟡", "Người nhạy cảm nên theo dõi triệu chứng nếu ở ngoài trời trong thời gian dài."),
            ("🏠", "Đóng cửa sổ vào giờ cao điểm bụi để giảm PM2.5 xâm nhập."),
            ("📊", "Theo dõi thêm các bản cập nhật tiếp theo để chủ động điều chỉnh lịch sinh hoạt."),
        ],
        "Nhạy cảm": [
            ("🚫", "Nhóm nhạy cảm nên hạn chế các hoạt động ngoài trời kéo dài."),
            ("🏢", "Đóng cửa sổ để giảm thiểu bụi mịn xâm nhập."),
            ("💗", "Theo dõi dự báo để chủ động bảo vệ sức khỏe."),
        ],
        "Xấu": [
            ("😷", "Nên đeo khẩu trang lọc bụi mịn nếu cần ra ngoài."),
            ("🏠", "Hạn chế vận động mạnh ngoài trời và ưu tiên ở trong nhà."),
            ("📣", "Theo dõi trẻ em, người già và người có bệnh hô hấp sát hơn bình thường."),
        ],
        "Rất Xấu": [
            ("⛔", "Tránh các hoạt động ngoài trời không cần thiết trong khung giờ cảnh báo."),
            ("🪟", "Giữ không khí trong nhà kín và cân nhắc dùng máy lọc không khí."),
            ("🩺", "Nếu có dấu hiệu khó thở hoặc kích ứng mạnh, cần nghỉ ngơi và theo dõi y tế."),
        ],
        "Nguy hiểm": [
            ("⚠️", "Hạn chế tối đa việc ra ngoài, đặc biệt với trẻ em, người già và người có bệnh nền."),
            ("🏥", "Sử dụng khẩu trang đạt chuẩn và tìm hỗ trợ y tế nếu có triệu chứng rõ rệt."),
            ("🔒", "Giữ môi trường trong nhà kín, dùng lọc không khí và tránh nguồn phát sinh bụi."),
        ],
    }
    advice_items = advice_map.get(peak_label, advice_map["Nguy hiểm"])
    advice_html = "".join(
        (
            '<div class="advice-item">'
            f'<span class="advice-icon">{icon}</span>'
            f"<span>{escape(text)}</span>"
            "</div>"
        )
        for icon, text in advice_items
    )

    return (
        '<div class="warning-box">'
        '<div class="warning-title">🔔 Cảnh báo &amp; Nhận định</div>'
        '<div class="warning-main">'
        f'PM2.5 dự kiến <strong>{trend_text}</strong> trong <strong>6 giờ tới</strong> và đạt đỉnh vào khoảng '
        f'<strong>{peak_time_text}</strong> với <span class="warning-highlight">{peak_row["y_pred"]:.1f} µg/m³</span>.'
        "</div>"
        '<div class="warning-main">'
        f'Chất lượng không khí sẽ ở mức <span class="warning-highlight">{peak_label}</span> '
        f'từ <strong>{start_time_text}</strong> đến <strong>{end_time_text}</strong>.'
        "</div>"
        '<div class="warning-section-label">📌 Khuyến nghị:</div>'
        f'<div class="advice-list">{advice_html}</div>'
        "</div>"
    )


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f4f6fb; color: #1c2e4a; }
        .block-container { max-width: 1400px; padding-top: 1.2rem; padding-bottom: 1.2rem; }
        section[data-testid="stSidebar"] { background: #0d1627; border-right: 1px solid rgba(255,255,255,0.05); }
        section[data-testid="stSidebar"] * { color: #ffffff !important; }
        section[data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #152238 !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
            border-radius: 8px !important;
            box-shadow: none !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] input {
            color: #ffffff !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] svg {
            fill: #ffffff !important;
        }
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stSlider label {
            color: #ffffff !important;
        }
        section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
            padding-left: 2px;
            padding-right: 2px;
        }
        .sidebar-brand { display: flex; align-items: center; gap: 11px; margin-bottom: 1.6rem; }
        .sidebar-logo { width: 44px; height: 44px; border-radius: 14px; background: rgba(255,255,255,0.08); display: flex; align-items: center; justify-content: center; font-size: 20px; font-weight: 700; }
        .sidebar-title { font-size: 20px; font-weight: 800; line-height: 1.1; }
        .sidebar-title span { color: #38bdf8; }
        .sidebar-note { background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.18); border-radius: 10px; padding: 12px; margin-top: 1rem; font-size: 12.5px; line-height: 1.55; }
        .sidebar-note code {
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 6px;
            padding: 1px 6px;
            color: #d9f3ff !important;
        }
        .sidebar-update { margin-top: 1rem; padding: 10px; border-radius: 8px; background: rgba(255,255,255,0.04); color: rgba(255,255,255,0.7) !important; font-size: 12px; line-height: 1.5; }
        .page-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; margin-bottom: 1.1rem; flex-wrap: wrap; }
        .page-title { font-size: 28px; font-weight: 800; color: #1c2e4a; margin: 0; }
        .page-subtitle { color: #68778f; margin-top: 4px; }
        .header-pills { display: flex; gap: 8px; flex-wrap: wrap; }
        .header-pill { background: #ffffff; border: 1px solid rgba(0,0,0,0.08); border-radius: 999px; padding: 6px 12px; font-size: 12px; color: #68778f; }
        .metric-card {
            background: #ffffff;
            border: 1px solid rgba(31, 51, 84, 0.10);
            border-left-width: 3px;
            border-radius: 16px;
            padding: 1.05rem 1.1rem;
            min-height: 134px;
            box-shadow: 0 10px 22px rgba(31, 51, 84, 0.06);
        }
        .metric-head { display: flex; align-items: center; gap: 12px; margin-bottom: 0.85rem; }
        .metric-icon {
            width: 40px;
            height: 40px;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            font-weight: 700;
            flex: 0 0 40px;
        }
        .metric-title {
            font-size: 14px;
            color: #52627c;
            line-height: 1.35;
            font-weight: 700;
        }
        .metric-value {
            font-size: 18px;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 0.8rem;
        }
        .metric-value strong { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.02em; }
        .metric-unit { font-size: 0.95rem; font-weight: 700; margin-left: 4px; opacity: 0.92; }
        .metric-foot { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 10px; }
        .metric-badge {
            display: inline-flex;
            align-items: center;
            min-height: 28px;
            padding: 5px 11px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 700;
            line-height: 1.2;
        }
        .metric-footer { font-size: 12px; color: #73819a; font-weight: 700; }
        .panel {
            background: #ffffff;
            border: 1px solid rgba(31, 51, 84, 0.09);
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 24px rgba(31, 51, 84, 0.06);
        }
        .panel-title { font-size: 16px; font-weight: 700; color: #1c2e4a; margin-bottom: 0.85rem; }
        .quality-title-row { display: flex; align-items: center; gap: 10px; margin-bottom: 0.9rem; }
        .quality-title-row .panel-title { margin-bottom: 0; }
        .quality-title-icon {
            width: 28px;
            height: 28px;
            border-radius: 10px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #e8f0ff;
            color: #3f79d8;
            font-size: 15px;
            font-weight: 800;
            flex: 0 0 28px;
        }
        .quality-meta { color: #60728d; font-size: 12px; font-weight: 700; margin: 0 0 0.55rem 2px; }
        .legend-row { display: flex; gap: 14px; flex-wrap: wrap; color: #68778f; font-size: 12px; margin-bottom: 0.65rem; }
        .legend-dot { width: 18px; height: 3px; border-radius: 999px; display: inline-block; margin-right: 5px; vertical-align: middle; }
        .forecast-toolbar-note { text-align:center; color:#68778f; font-size:12px; padding-top:8px; }
        .hour-chip {
            background: #f4f6fb;
            border: 1px solid rgba(0,0,0,0.08);
            border-radius: 10px;
            padding: 12px 8px;
            text-align: center;
            min-height: 168px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 8px;
        }
        .hour-time { font-size: 11px; color: #9aaabf; line-height: 1.45; min-height: 34px; }
        .hour-value { font-size: 18px; font-weight: 700; line-height: 1.15; }
        .hour-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 6px 10px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 600;
            line-height: 1.35;
            min-height: 42px;
            width: 100%;
            max-width: 100%;
        }
        .table-wrap { margin-top: 14px; overflow-x: auto; }
        .compare-wrap {
            overflow-x: auto;
            background: #f7faff;
            border: 1px solid #dbe5f2;
            border-radius: 12px;
            overflow: hidden;
        }
        .table-wrap table, .compare-wrap table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .table-wrap th, .compare-wrap th {
            text-align: left;
            padding: 10px 12px;
            color: #60728d;
            font-size: 12px;
            font-weight: 800;
            border-bottom: 1px solid rgba(71, 98, 138, 0.16);
            border-right: 1px solid rgba(71, 98, 138, 0.12);
            background: #eef4fb;
        }
        .table-wrap td, .compare-wrap td {
            padding: 11px 12px;
            border-bottom: 1px solid rgba(71, 98, 138, 0.10);
            border-right: 1px solid rgba(71, 98, 138, 0.10);
            color: #1c2e4a;
            background: #ffffff;
        }
        .compare-wrap th:last-child, .compare-wrap td:last-child { border-right: none; }
        .compare-wrap tbody tr:last-child td { border-bottom: none; }
        .best-row td { background: #e2f4ea; }
        .focus-row td { background: #eef4ff; }
        .best-row.focus-row td { background: #d8efe2; }
        .soft-badge { display: inline-block; padding: 4px 9px; border-radius: 999px; font-size: 11px; font-weight: 600; }
        .warning-box {
            background: linear-gradient(180deg, #fcf2e7 0%, #f9ebd9 100%);
            border: 1px solid #f4ca96;
            border-radius: 18px;
            padding: 1rem;
            height: 100%;
        }
        .warning-title { font-size: 16px; font-weight: 700; color: #c86c1a; margin-bottom: 12px; }
        .warning-main {
            background: rgba(255,255,255,0.78);
            border-radius: 12px;
            padding: 12px 14px;
            margin-bottom: 12px;
            color: #6b4423;
            font-size: 13px;
            line-height: 1.7;
        }
        .warning-section-label { font-size: 13px; font-weight: 700; color: #c86c1a; margin: 10px 0 8px; }
        .warning-highlight { color: #d16f14; font-weight: 700; }
        .advice-list { display: flex; flex-direction: column; gap: 8px; }
        .advice-item {
            display: flex;
            align-items: flex-start;
            gap: 10px;
            color: #8a5311;
            font-size: 13px;
            line-height: 1.55;
        }
        .advice-icon {
            flex: 0 0 24px;
            height: 24px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: rgba(255,255,255,0.75);
            border-radius: 8px;
            font-size: 14px;
        }
        .warning-box p, .warning-box li { color: #854F0B; font-size: 13px; line-height: 1.65; }
        .conclusion-box { background: #e6f1fb; border: 1px solid #b5d4f4; border-radius: 12px; padding: 1rem; height: 100%; }
        .conclusion-title { font-size: 16px; font-weight: 700; color: #185FA5; margin-bottom: 10px; }
        div[data-testid="stButton"] > button { border-radius: 8px; border: none; color: #1c2e4a; font-weight: 700; background: rgba(255,255,255,0.9); }
        div[data-testid="stButton"] > button:hover { color: #1c2e4a; background: #ffffff; }
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button {
            background: #22314b !important;
            color: #dbeafe !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
            background: #2a3b59 !important;
            color: #ffffff !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button:disabled,
        section[data-testid="stSidebar"] div[data-testid="stButton"] > button[disabled] {
            background: #22314b !important;
            color: rgba(219,234,254,0.55) !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            opacity: 1 !important;
            cursor: not-allowed !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app() -> None:
    st.set_page_config(page_title="Dự báo chất lượng không khí", layout="wide", initial_sidebar_state="expanded")
    apply_theme()

    metrics_df = load_registry_metrics()
    if metrics_df.empty:
        st.error("Không tìm thấy bundle model hợp lệ trong project.")
        return

    best_row = metrics_df.sort_values("MAE", na_position="last").iloc[0]
    model_options = metrics_df["Model"].tolist()
    best_supported_forecast_model = get_best_supported_forecast_model(metrics_df)

    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">AQI</div>
            <div class="sidebar-title">Dự báo chất lượng không khí<br><span>FORECAST</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio("Điều hướng", ["Dự báo ", "Chất lượng mô hình"], label_visibility="collapsed")
    default_model = (
        best_supported_forecast_model
        if page == "Dự báo " and best_supported_forecast_model is not None
        else str(best_row["Model"])
    )
    default_index = model_options.index(default_model)
    selected_model = st.sidebar.selectbox("Chọn mô hình", model_options, index=default_index)
    selected_model_effective = selected_model
    if page == "Dự báo " and not is_forecast_runtime_supported(resolve_bundle_dir(metrics_df, selected_model)):
        if best_supported_forecast_model is None:
            st.error(
                "Không có model nào forecast được trong runtime hiện tại. "
                "Các bundle hiện có đều chứa Lambda layer và cần export lại."
            )
            st.stop()

        selected_model_effective = best_supported_forecast_model
        st.sidebar.warning(
            "Model đang chọn không chạy được forecast ngoài dữ liệu trong runtime hiện tại "
            "vì bundle chứa Lambda layer. Dashboard tự chuyển sang model khả dụng tốt nhất."
        )

    sidebar_bundle_dir = resolve_bundle_dir(
        metrics_df,
        selected_model if page == "Chất lượng mô hình" else selected_model_effective,
    )
    max_horizon_steps, bundle_step_hours = load_bundle_runtime_limits(sidebar_bundle_dir)
    default_horizon_steps = min(24, max_horizon_steps)
    horizon_steps = st.sidebar.slider(
        f"Số bước dự báo (mỗi bước {bundle_step_hours} giờ, tối đa {max_horizon_steps * bundle_step_hours}h)",
        1,
        max_horizon_steps,
        default_horizon_steps,
        step=1,
    )
    forecast_page_size = st.sidebar.selectbox("Số mốc forecast mỗi trang", [6, 8, 10, 12], index=1)
    st.sidebar.button("Cập nhật dashboard", use_container_width=True)

    if page == "Chất lượng mô hình":
        st.sidebar.markdown(
            """
            <div class="sidebar-note">
                Metrics được đọc trực tiếp từ từng bundle trong <code>model_registry</code> và đã loại bản trùng model.
            </div>
            """,
            unsafe_allow_html=True,
        )

    quality_chart_key = "quality_mae_chart"
    if page == "Chất lượng mô hình":
        if st.session_state.get("quality_sidebar_model") != selected_model:
            st.session_state.quality_sidebar_model = selected_model
            st.session_state.quality_focus_model = selected_model

        clicked_model = get_selected_model_from_mae_chart(st.session_state.get(quality_chart_key), metrics_df)
        if clicked_model is not None:
            st.session_state.quality_focus_model = clicked_model

    active_model = (
        str(st.session_state.get("quality_focus_model", selected_model))
        if page == "Chất lượng mô hình"
        else selected_model_effective
    )
    bundle_dir = resolve_bundle_dir(metrics_df, active_model)
    selected_row = metrics_df.loc[metrics_df["Model"] == active_model].iloc[0]
    timeline_df = load_bundle_timeline(bundle_dir)

    try:
        history_df, forecast_df, step_hours, forecast_source = load_latest_forecast_or_timeline(bundle_dir, horizon_steps)
    except Exception as exc:
        if page == "Dự báo ":
            st.error(
                "Không thể tạo forecast ngoài dữ liệu cho model đang chọn. "
                "App đã chặn fallback sang test timeline để tránh hiển thị dự báo sai ngữ nghĩa."
            )
            st.caption(f"Lỗi chi tiết: {type(exc).__name__}: {exc}")
            st.stop()
        raise
    history_df = history_df.tail(DEFAULT_HISTORY_WINDOW).copy()

    chart_key = f"forecast_chart_{active_model}_{horizon_steps}".replace(" ", "_")
    current_val = float(history_df["PM25"].iloc[-1])
    future_val = float(forecast_df["y_pred"].iloc[0])
    delta = future_val - current_val
    delta_pct = (delta / current_val) * 100 if current_val else 0.0
    aqi_val = aqi_from_pm25(future_val)
    quality_label, quality_color, quality_bg = band_info(future_val)
    peak_row = forecast_df.loc[forecast_df["y_pred"].idxmax()]
    forecast_hours = len(forecast_df) * step_hours
    current_time_text = history_df[DEFAULT_TIMESTAMP_COL].iloc[-1].strftime("%H:%M")
    next_time_text = forecast_df[DEFAULT_TIMESTAMP_COL].iloc[0].strftime("%H:%M")

    selected_point = get_selected_chart_point(st.session_state.get(chart_key), history_df, forecast_df)
    if selected_point is None:
        focus_value = future_val
        focus_label = quality_label
        focus_color = quality_color
        focus_bg = quality_bg
        focus_delta = delta
        focus_delta_pct = delta_pct
        focus_aqi = aqi_val
        focus_time_text = next_time_text
        focus_source_label = "Dự báo"
        focus_icon = "↗"
        focus_title = f"Dự báo giờ tới ({next_time_text})"
        focus_footer = f"Mốc +{step_hours}h đầu tiên"
    else:
        focus_value = float(selected_point["value"])
        focus_label, focus_color, focus_bg = band_info(focus_value)
        focus_delta = focus_value - current_val
        focus_delta_pct = (focus_delta / current_val) * 100 if current_val else 0.0
        focus_aqi = aqi_from_pm25(focus_value)
        focus_time_text = selected_point["timestamp"].strftime("%H:%M")
        focus_source_label = str(selected_point["source_label"])
        focus_icon = "◉" if selected_point["source"] != "forecast" else "↗"
        focus_title = f"Điểm đang chọn ({focus_time_text})"
        focus_footer = f"{focus_source_label} · {selected_point['timestamp'].strftime('%d/%m %H:%M')}"

    title = "Hệ thống dự báo PM2.5" if page == "Dự báo " else "Đánh giá chất lượng mô hình"
    subtitle = (
        f"Dữ liệu thật từ bundle {selected_row['Bundle Key']} · horizon {len(forecast_df)} bước · mỗi bước {step_hours} giờ"
        if page == "Dự báo "
        else "So sánh metrics và backtest giữa các bundle model trong project"
    )

    st.markdown(
        f"""
        <div class="page-header">
            <div>
                <div class="page-title">{title}</div>
                <div class="page-subtitle">{subtitle}</div>
            </div>
            <div class="header-pills">
                <div class="header-pill">Model: <strong>{active_model}</strong></div>
                <div class="header-pill">MAE: <strong>{selected_row['MAE']:.2f}</strong></div>
                <div class="header-pill">Nguồn forecast: <strong>{forecast_source}</strong></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if page == "Dự báo ":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            now_label, now_color, now_bg = band_info(current_val)
            metric_card(
                "PM2.5 hiện tại",
                f"<strong>{current_val:.1f}</strong><span class=\"metric-unit\">µg/m³</span>",
                now_label,
                now_color,
                now_bg,
                icon="◎",
                footer=f"Cập nhật lúc {current_time_text}",
            )
        with col2:
            metric_card(
                focus_title,
                f"<strong>{focus_value:.1f}</strong><span class=\"metric-unit\">µg/m³</span>",
                focus_source_label,
                focus_color,
                focus_bg,
                icon=focus_icon,
                footer=focus_footer,
            )
        with col3:
            metric_card(
                "Thay đổi so với hiện tại",
                f"<strong>{focus_delta:+.1f}</strong><span class=\"metric-unit\">µg/m³</span>",
                f"{focus_delta_pct:+.2f}%",
                "#F28A1A",
                "#FCEAD8",
                icon="Δ",
                footer="Chọn điểm trên biểu đồ để so sánh",
            )
        with col4:
            metric_card(
                "Chất lượng không khí",
                f"<strong>{focus_label}</strong>",
                focus_label,
                focus_color,
                focus_bg,
                icon="◌",
                footer=f"AQI: {focus_aqi}",
            )

        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-title">✣ Diễn biến PM2.5 và dự báo trong {forecast_hours} giờ tới</div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            make_forecast_chart(history_df, forecast_df),
            use_container_width=True,
            key=chart_key,
            on_select="rerun",
            selection_mode="points",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        left, right = st.columns([2.2, 1], gap="large")
        
        with left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)

            # ===== State cho slider trái/phải =====
            page_size = int(forecast_page_size)
            total_items = len(forecast_df)

            if "forecast_start_idx" not in st.session_state:
                st.session_state.forecast_start_idx = 0

            max_start = max(0, total_items - page_size)
            start_idx = st.session_state.forecast_start_idx
            end_idx = min(start_idx + page_size, total_items)

            # ===== Header + counter + nút xem chi tiết =====
            header_left, header_mid, header_right = st.columns([3.8, 2.0, 1.3])
            with header_left:
                st.markdown('<div class="panel-title">Dự báo PM2.5 theo giờ</div>', unsafe_allow_html=True)
            with header_mid:
                st.markdown(
                    f"""
                    <div class="forecast-toolbar-note">
                        Hiển thị {start_idx + 1}–{end_idx} / {total_items} mốc dự báo
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with header_right:
                if "show_forecast_detail" not in st.session_state:
                    st.session_state.show_forecast_detail = False

                if st.button(
                    "Xem chi tiết" if not st.session_state.show_forecast_detail else "Ẩn chi tiết",
                    key="toggle_forecast_detail",
                    use_container_width=True,
                ):
                    st.session_state.show_forecast_detail = not st.session_state.show_forecast_detail

            nav1, nav2, nav3 = st.columns([1, 6, 1])

            with nav1:
                if st.button("◀", key="forecast_prev", use_container_width=True, disabled=st.session_state.forecast_start_idx == 0):
                    st.session_state.forecast_start_idx = max(0, st.session_state.forecast_start_idx - page_size)

            with nav3:
                if st.button("▶", key="forecast_next", use_container_width=True, disabled=st.session_state.forecast_start_idx >= max_start):
                    st.session_state.forecast_start_idx = min(max_start, st.session_state.forecast_start_idx + page_size)

            with nav2:
                st.empty()

            start_idx = st.session_state.forecast_start_idx
            end_idx = min(start_idx + page_size, total_items)
            visible_df = forecast_df.iloc[start_idx:end_idx].copy()

            # ===== Dãy chip forecast =====
            chip_cols = st.columns(page_size)
            for idx in range(page_size):
                with chip_cols[idx]:
                    if idx < len(visible_df):
                        row = visible_df.iloc[idx]
                        chip_card(row[DEFAULT_TIMESTAMP_COL].strftime("%d/%m/%Y-%H:%M"), float(row["y_pred"]))
                    else:
                        st.empty()

            # ===== Chi tiết bảng =====
            if st.session_state.show_forecast_detail:
                table_df = visible_df.copy()
                table_df["Thời gian"] = table_df[DEFAULT_TIMESTAMP_COL].dt.strftime("%d/%m/%Y-%H:%M")
                table_df["PM2.5 dự báo"] = table_df["y_pred"].round(1)
                table_df["Thay đổi"] = table_df["y_pred"].apply(
                    lambda value: f"{value - current_val:+.1f} ({((value - current_val) / current_val) * 100:+.1f}%)"
                    if current_val else "0.0 (0.0%)"
                )
                table_df['AQI'] = table_df['PM2.5 dự báo'].apply(aqi_from_pm25)

                rows = ""
                for _, row in table_df.iterrows():
                    label, color, bg = band_info(float(row["PM2.5 dự báo"]))
                    delta_val = float(row["PM2.5 dự báo"]) - current_val
                    delta_color = "#3B6D11" if delta_val >= 0 else "#A32D2D"

                    rows += f"""
                    <tr>
                        <td>{row['Thời gian']}</td>
                        <td>{row['PM2.5 dự báo']:.1f}</td>
                        <td style="color:{delta_color}">{row['Thay đổi']}</td>
                        <td>{int(row['AQI'])}</td>
                        <td><span class="soft-badge" style="background:{bg}; color:{color}">{label}</span></td>
                    </tr>
                    """

                st.markdown(
                    f"""
                    <div class="table-wrap">
                        <table>
                            <thead>
                                <tr>
                                    <th>Thời gian</th>
                                    <th>PM2.5 dự báo (µg/m³)</th>
                                    <th>Thay đổi</th>
                                    <th>AQI</th>
                                    <th>Mức chất lượng</th>
                                </tr>
                            </thead>
                            <tbody>{rows}</tbody>
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown(build_warning_html(forecast_df, current_val, step_hours), unsafe_allow_html=True)

    else:
        ranking_df = metrics_df.sort_values("MAE", na_position="last").reset_index(drop=True)
        current_rank = int(ranking_df.index[ranking_df["Model"] == active_model][0]) + 1
        rank_badge = "Tốt nhất theo MAE" if active_model == str(best_row["Model"]) else f"Top {current_rank}/{len(ranking_df)}"
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            
            metric_card("Model đang xem", str(active_model), rank_badge, "#1DC54F", "#E6F1FB")
        with q2:
            metric_card("MAE", f"{selected_row['MAE']:.2f} µg/m³", active_model, "#5391EE", "#E1F5EE")
        with q3:
            metric_card("RMSE", f"{selected_row['RMSE']:.2f} µg/m³", active_model, "#534AB7", "#EEEDFE")
        with q4:
            peak_mae = selected_row["Peak MAE"]
            peak_text = f"{peak_mae:.2f}" if pd.notna(peak_mae) else "N/A"
            metric_card("MAPE / Peak MAE", f"{selected_row['MAPE']:.2f}%", peak_text, "#EC9E0D", "#FAEEDA")

        left, right = st.columns([1.0, 1.18], gap="large")
        with left:
            rows = ""
            table_df = ranking_df.copy()
            for idx, row in table_df.iterrows():
                row_classes: list[str] = []
                badges: list[str] = []
                if idx == 0:
                    row_classes.append("best-row")
                    badges.append('<span class="soft-badge" style="background:#EAF3DE; color:#1DC54F"> 🏆 Best</span>')
                if str(row["Model"]) == active_model:
                    row_classes.append("focus-row")
                    badges.append('<span class="soft-badge" style="background:#EAF2FF; color:#3F79D8">Đang xem</span>')
                row_class = " ".join(row_classes)
                badge = f" {' '.join(badges)}" if badges else ""
                
                rows += f"""
                <tr class="{row_class}">
                    <td><strong>{row['Model']}</strong>{badge}</td>
                    <td>{row['MSE']:.2f}</td>
                    <td>{row['MAE']:.2f}</td>
                    <td>{row['RMSE']:.2f}</td>
                    <td>{row['MAPE']:.2f}%</td>
                    <td>{row['Peak MAE']:.2f}</td>
           
                </tr>
                """
            st.markdown(
                f"""
                <div class="panel">
                    <div class="quality-title-row">
                        <div class="quality-title-icon">▦</div>
                        <div class="panel-title">Bảng so sánh {len(table_df)} mô hình</div>
                    </div>
                    <div class="compare-wrap">
                        <table>
                            <thead>
                                <tr>
                                    <th>Model</th>
                                    <th>MSE</th>
                                    <th>MAE</th>
                                    <th>RMSE</th>
                                    <th>MAPE</th>
                                    <th>Peak MAE</th>
                                </tr>
                            </thead>
                            <tbody>{rows}</tbody>
                        </table>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            with st.container(border=True):
                st.markdown(
                    """
                    <div class="quality-title-row">
                        <div class="quality-title-icon">✦</div>
                        <div class="panel-title">So sánh MAE giữa các mô hình</div>
                    </div>
                    <div class="quality-meta">MAE (µg/m³)</div>
                    <div class="legend-row">
                        <span><span class="legend-dot" style="background:#67C9A8"></span>Model đang xem</span>
                        <span><span class="legend-dot" style="background:#8FB5F9"></span>Model còn lại</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    make_mae_chart(metrics_df, active_model),
                    use_container_width=True,
                    key=quality_chart_key,
                    on_select="rerun",
                    selection_mode="points",
                    config={"displayModeBar": False},
                )

        lower_left, lower_right = st.columns([2, 1], gap="large")
        with lower_left:
            st.markdown('<div class="panel"><div class="panel-title">Backtest gần nhất của model đang chọn</div>', unsafe_allow_html=True)
            if timeline_df.empty:
                st.info("Bundle này chưa có test_timeline.csv để vẽ backtest.")
            else:
                timeline_options = timeline_df["timestamp"].dropna().sort_values().drop_duplicates().tolist()
                default_start_idx = max(0, len(timeline_options) - min(48, len(timeline_options)))
                selected_start, selected_end = st.select_slider(
                    "Chọn khoảng thời gian backtest",
                    options=timeline_options,
                    value=(timeline_options[default_start_idx], timeline_options[-1]),
                    format_func=lambda ts: ts.strftime("%d/%m/%Y %H:%M"),
                    key=f"backtest_range_{active_model}",
                )
                filtered_timeline_df = timeline_df[
                    (timeline_df["timestamp"] >= selected_start)
                    & (timeline_df["timestamp"] <= selected_end)
                ].copy()
                st.caption(
                    f"Hiển thị {len(filtered_timeline_df)} mốc từ "
                    f"{selected_start.strftime('%d/%m/%Y %H:%M')} đến {selected_end.strftime('%d/%m/%Y %H:%M')}."
                )
                st.plotly_chart(make_backtest_chart(filtered_timeline_df, active_model), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with lower_right:
            peak_text = f"{selected_row['Peak MAE']:.2f}" if pd.notna(selected_row["Peak MAE"]) else "N/A"
            data_start = selected_row["Data Start"] if "Data Start" in selected_row else "N/A"
            data_end = selected_row["Data End"] if "Data End" in selected_row else "N/A"
            summary_line = (
                f"<strong>{active_model}</strong> hiện là model tốt nhất theo MAE trong {len(ranking_df)} mô hình."
                if active_model == str(best_row["Model"])
                else (
                    f"<strong>{active_model}</strong> hiện đứng hạng <strong>{current_rank}/{len(ranking_df)}</strong> "
                    f"theo MAE. Model tốt nhất hiện tại là <strong>{best_row['Model']}</strong>."
                )
            )
            st.markdown(
                f"""
                <div class="conclusion-box">
                    <div class="conclusion-title">Kết luận</div>
                    <p>Dữ liệu được sử dụng để huấn luyện model là dữ liệu từ {data_start} đến {data_end}.</p>
                    <p>Model đang xem là <strong>{active_model}</strong> với MAE <strong>{selected_row['MAE']:.2f}</strong>, MSE <strong>{selected_row['MSE']:.2f}</strong>, RMSE <strong>{selected_row['RMSE']:.2f}</strong>, Peak MAE <strong>{peak_text}</strong>.</p>
                    <p>{summary_line}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    


if __name__ == "__main__":
    render_app()
