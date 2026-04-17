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


AQI_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 500.0, 301, 500),
]

AQI_LEVELS = [
    {"range": (0, 50), "label": "Tốt", "color": "#16a34a", "bg": "#dcfce7"},
    {"range": (51, 100), "label": "Trung bình", "color": "#ca8a04", "bg": "#fef9c3"},
    {"range": (101, 150), "label": "Nhạy cảm", "color": "#ea580c", "bg": "#ffedd5"},
    {"range": (151, 200), "label": "Không tốt", "color": "#dc2626", "bg": "#fee2e2"},
    {"range": (201, 300), "label": "Rất Xấu", "color": "#9f1239", "bg": "#ffe4e6"},
    {"range": (301, 500), "label": "Nguy hiểm", "color": "#7c3aed", "bg": "#ede9fe"},
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
    return "Nguy hiểm", "#7c3aed", "#ede9fe"


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
        return {"source": "actual", "source_label": "Giá trị thực tế", "timestamp": pd.to_datetime(row[DEFAULT_TIMESTAMP_COL]), "value": float(row["PM25"])}
    if curve_number == 1 and 0 <= point_index < len(forecast_df):
        row = forecast_df.iloc[point_index]
        return {"source": "forecast", "source_label": "Dự báo", "timestamp": pd.to_datetime(row[DEFAULT_TIMESTAMP_COL]), "value": float(row["y_pred"])}
    if curve_number == 2 and "y_true" in forecast_df.columns and 0 <= point_index < len(forecast_df):
        row = forecast_df.iloc[point_index]
        if pd.notna(row.get("y_true")):
            return {"source": "future_actual", "source_label": "Thực tế tương lai", "timestamp": pd.to_datetime(row[DEFAULT_TIMESTAMP_COL]), "value": float(row["y_true"])}
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
    return load_bundle_registry_metrics(APP_DIR, registry_dir=REGISTRY_DIR, best_bundle_dir=BEST_BUNDLE_DIR)


def load_raw_data() -> pd.DataFrame:
    raw_df = pd.read_csv(RAW_DATA_PATH)
    raw_df[DEFAULT_TIMESTAMP_COL] = pd.to_datetime(raw_df[DEFAULT_TIMESTAMP_COL])
    return raw_df.sort_values(DEFAULT_TIMESTAMP_COL).reset_index(drop=True)


def format_optional_datetime(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        text = str(value).strip()
        return text or None
    return parsed.strftime("%d/%m/%Y %H:%M")


def get_project_data_range_text(step_hours: int) -> tuple[str, str]:
    prepared = prepare_raw_frame(load_raw_data(), step_hours=step_hours).reset_index()
    if prepared.empty:
        return "N/A", "N/A"
    start_ts = pd.to_datetime(prepared[DEFAULT_TIMESTAMP_COL].min(), errors="coerce")
    end_ts = pd.to_datetime(prepared[DEFAULT_TIMESTAMP_COL].max(), errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        return "N/A", "N/A"
    return start_ts.strftime("%d/%m/%Y %H:%M"), end_ts.strftime("%d/%m/%Y %H:%M")


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


def _estimate_future_value(source_series: pd.Series, target_ts: pd.Timestamp, *, step_hours: int) -> float:
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
    same_weekday_hour = recent_window[(recent_window.index.dayofweek == target_ts.dayofweek) & (recent_window.index.hour == target_ts.hour)]
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
    future_index = pd.date_range(start=prepared_history.index.max() + step_delta, periods=horizon_steps, freq=step_delta)
    future_frame = pd.DataFrame(index=future_index)
    required_future_cols = [col for col in bundle.required_raw_columns if col != "PM25"]
    for col in required_future_cols:
        if col not in prepared_history.columns:
            raise ValueError(f"History is missing required raw column for future extrapolation: {col}")
        source_series = prepared_history[col].astype(float)
        future_frame[col] = [_estimate_future_value(source_series, timestamp, step_hours=bundle.step_hours) for timestamp in future_index]
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
    forecast_df = forecast_recursive(bundle, history_frame.reset_index(), future_frame, horizon=max_steps, timestamp_col=DEFAULT_TIMESTAMP_COL)
    history_plot_df = history_frame.tail(min(14, len(history_frame))).reset_index()
    return history_plot_df, forecast_df, bundle.step_hours


def load_latest_forecast_or_timeline(bundle_dir: Path, horizon_steps: int) -> tuple[pd.DataFrame, pd.DataFrame, int, str]:
    history_df, forecast_df, step_hours = load_latest_forecast(bundle_dir, horizon_steps)
    return history_df, forecast_df, step_hours, "forecast_ngoai_du_lieu"


def make_forecast_chart(history_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    zones = [
        (0, 25, "Tốt", "rgba(22,163,74,0.13)", "#15803d"),
        (25, 50, "Trung bình", "rgba(202,138,4,0.13)", "#a16207"),
        (50, 75, "Nhạy cảm", "rgba(234,88,12,0.13)", "#c2410c"),
        (75, 100, "Không tốt", "rgba(220,38,38,0.14)", "#b91c1c"),
        (100, 200, "Rất xấu", "rgba(159,18,57,0.14)", "#9f1239"),
        (200, 500, "Nguy hiểm", "rgba(124,58,237,0.13)", "#6d28d9"),
    ]
    y_values = pd.concat([history_df["PM25"], forecast_df["y_pred"], forecast_df["y_true"] if "y_true" in forecast_df.columns else pd.Series(dtype=float)], ignore_index=True).dropna()
    observed_max = float(y_values.max()) if not y_values.empty else 80.0
    y_max = max(90.0, observed_max + 18.0)
    for bound in [90.0, 100.0, 120.0, 150.0, 200.0, 250.0, 300.0, 500.0]:
        if y_max <= bound:
            y_max = bound
            break

    visible_zones = [(low, min(high, y_max), label, fill, text_color) for low, high, label, fill, text_color in zones if low < y_max]
    for lower, upper, _, fill, _ in visible_zones:
        fig.add_hrect(y0=lower, y1=upper, fillcolor=fill, line_width=0)

    fig.add_trace(go.Scatter(x=history_df[DEFAULT_TIMESTAMP_COL], y=history_df["PM25"], mode="lines+markers", name="Giá trị thực tế", line=dict(color="#3b82f6", width=2.5), marker=dict(size=6, color="#3b82f6", line=dict(color="white", width=1.5))))
    fig.add_trace(go.Scatter(x=forecast_df[DEFAULT_TIMESTAMP_COL], y=forecast_df["y_pred"], mode="lines+markers", name="Dự báo", line=dict(color="#10b981", width=2.5, dash="dot"), marker=dict(size=6, color="#10b981", line=dict(color="white", width=1.5))))

    if "y_true" in forecast_df.columns and forecast_df["y_true"].notna().any():
        fig.add_trace(go.Scatter(x=forecast_df[DEFAULT_TIMESTAMP_COL], y=forecast_df["y_true"], mode="lines+markers", name="Thực tế tương lai", line=dict(color="#ef4444", width=2), marker=dict(size=6, color="#ef4444")))

    current_x = history_df[DEFAULT_TIMESTAMP_COL].iloc[-1]
    peak = forecast_df.loc[forecast_df["y_pred"].idxmax()]

    fig.add_vline(x=current_x, line_dash="dash", line_color="rgba(59,130,246,0.5)", line_width=1.5)
    fig.add_annotation(x=current_x, y=y_max * 0.97, text="Hiện tại", showarrow=False, bgcolor="white", bordercolor="#bfdbfe", borderwidth=1.5, borderpad=5, font=dict(size=11, color="#2563eb", family="sans-serif"), xanchor="center")
    fig.add_annotation(
        x=peak[DEFAULT_TIMESTAMP_COL], y=peak["y_pred"] + (y_max * 0.06),
        text=f"{peak[DEFAULT_TIMESTAMP_COL].strftime('%H:%M')}<br><b>{peak['y_pred']:.1f} µg/m³</b>",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor="#10b981",
        bgcolor="white", bordercolor="#6ee7b7", borderwidth=1.5, borderpad=6,
        font=dict(size=11, color="#065f46"), xanchor="center",
    )

    for lower, upper, label, _, text_color in visible_zones:
        fig.add_annotation(xref="paper", x=0.005, y=(lower + upper) / 2, yref="y", text=label, showarrow=False, xanchor="left", font=dict(size=11, color=text_color, family="sans-serif"))

    fig.update_layout(
        height=340,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=12), itemsizing="constant"),
        xaxis=dict(title="", showgrid=False, tickformat="%H:%M", tickfont=dict(size=11, color="#64748b")),
        yaxis=dict(title="PM2.5 (µg/m³)", gridcolor="rgba(0,0,0,0.05)", range=[0, y_max], tickfont=dict(size=11, color="#64748b"), title_font=dict(size=12, color="#475569")),
    )
    return fig


def make_mae_chart(metrics_df: pd.DataFrame, selected_model: str) -> go.Figure:
    chart_df = metrics_df.sort_values("MAE", na_position="last").reset_index(drop=True)
    colors = ["#10b981" if model == selected_model else "#93c5fd" for model in chart_df["Model"]]
    text_colors = ["#065f46" if model == selected_model else "#1d4ed8" for model in chart_df["Model"]]
    y_max = float(chart_df["MAE"].max()) if not chart_df.empty else 1.0
    fig = go.Figure(go.Bar(
        x=chart_df["Model"], y=chart_df["MAE"],
        text=chart_df["MAE"].round(2), textposition="outside", cliponaxis=False,
        marker=dict(color=colors, line=dict(color="white", width=1.5), cornerradius=6),
        textfont=dict(color=text_colors, size=14, family="sans-serif"),
        hovertemplate="<b>%{x}</b><br>MAE: %{y:.2f} µg/m³<extra></extra>",
        selectedpoints=[idx for idx, model in enumerate(chart_df["Model"]) if model == selected_model],
        selected=dict(marker=dict(opacity=1.0)),
        unselected=dict(marker=dict(opacity=0.85)),
    ))
    fig.update_layout(
        height=400,
        margin=dict(l=8, r=8, t=28, b=90),
        paper_bgcolor="white",
        plot_bgcolor="white",
        clickmode="event+select",
        xaxis=dict(title="", tickfont=dict(size=12, color="#475569"), tickangle=28, automargin=True, showgrid=False),
        yaxis=dict(title="MAE (µg/m³)", gridcolor="rgba(99,102,241,0.08)", range=[0, y_max * 1.22], automargin=True, tickfont=dict(size=11, color="#64748b")),
    )
    return fig


def make_backtest_chart(timeline_df: pd.DataFrame, selected_model: str) -> go.Figure:
    plot_df = timeline_df.sort_values("timestamp").copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["y_true"], mode="lines+markers", name="Thực tế", line=dict(color="#3b82f6", width=2.5), marker=dict(size=5, line=dict(color="white", width=1))))
    fig.add_trace(go.Scatter(x=plot_df["timestamp"], y=plot_df["y_pred"], mode="lines+markers", name=selected_model, line=dict(color="#10b981", width=2.5, dash="dot"), marker=dict(size=5, line=dict(color="white", width=1))))
    fig.update_layout(
        height=260,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=12)),
        xaxis=dict(title="", tickformat="%d/%m %H:%M", showgrid=False, tickfont=dict(size=11, color="#64748b")),
        yaxis=dict(title="PM2.5 (µg/m³)", gridcolor="rgba(0,0,0,0.05)", tickfont=dict(size=11, color="#64748b")),
    )
    return fig


def metric_card(title: str, value: str, badge: str, accent: str, badge_bg: str, *, icon: str = "◎", footer: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card" style="border-left-color:{accent}">
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


def render_sidebar_footer(page: str, last_update_text: str, num_models: int) -> None:
    top_gap_html = (
        '<div class="sidebar-footer-gap" aria-hidden="true"></div>'
        if page == "Chất lượng mô hình"
        else ""
    )
    model_text = "mô hình" if num_models == 1 else "mô hình"
    st.sidebar.markdown(
        (
            f'<div class="sidebar-footer">'
            f"{top_gap_html}"
            f'<div class="sidebar-footer-row">'
            f'<span class="sidebar-footer-icon">ⓘ</span>'
            f'<span class="sidebar-footer-label">Cập nhật cuối</span>'
            f"</div>"
            f'<div class="sidebar-footer-value">{escape(last_update_text)}</div>'
            f'<div class="sidebar-footer-note">'
            f"{num_models} {model_text} đã được huấn luyện và đánh giá trên cùng tập dữ liệu kiểm tra."
            f"</div>"
            f"</div>"
        ),
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str, pills: list[str]) -> None:
    pills_html = "".join(pills)
    st.markdown(
        (
            f'<div class="page-header">'
            f'<div class="page-header-copy">'
            f'<div class="page-title">{escape(title)}</div>'
            f'<div class="page-subtitle">{escape(subtitle)}</div>'
            f"</div>"
            f'<div class="header-pills-shell">'
            f'<div class="header-pills">{pills_html}</div>'
            f"</div>"
            f"</div>"
        ),
        unsafe_allow_html=True,
    )


def build_warning_html(forecast_df: pd.DataFrame, current_val: float, step_hours: int) -> str:
    peak_position = int(forecast_df["y_pred"].to_numpy().argmax())
    peak_row = forecast_df.iloc[peak_position]
    peak_label, peak_color, peak_bg = band_info(float(peak_row["y_pred"]))

    labels = forecast_df["y_pred"].apply(lambda v: band_info(float(v))[0]).tolist()
    start_pos = peak_position
    end_pos = peak_position
    while start_pos > 0 and labels[start_pos - 1] == peak_label:
        start_pos -= 1
    while end_pos < len(labels) - 1 and labels[end_pos + 1] == peak_label:
        end_pos += 1

    start_time_text = forecast_df.iloc[start_pos][DEFAULT_TIMESTAMP_COL].strftime("%H:%M")
    end_time_text = forecast_df.iloc[end_pos][DEFAULT_TIMESTAMP_COL].strftime("%H:%M")
    peak_time_text = peak_row[DEFAULT_TIMESTAMP_COL].strftime("%H:%M")

    window_steps = max(1, int(round(6 / step_hours)))
    six_hour_value = float(forecast_df["y_pred"].head(window_steps).iloc[-1])
    delta = six_hour_value - current_val
    trend_text = "tăng" if delta > 1.0 else ("giảm" if delta < -1.0 else "dao động nhẹ")

    advice_map = {
        "Tốt": [("✅", "Chất lượng không khí đang ở mức an toàn cho các hoạt động ngoài trời."), ("🌿", "Tiếp tục theo dõi theo khung giờ cao điểm để phát hiện biến động sớm."), ("💧", "Giữ không gian sống thông thoáng và vệ sinh khu vực có bụi mịn.")],
        "Trung bình": [("🟡", "Người nhạy cảm nên theo dõi triệu chứng nếu ở ngoài trời lâu."), ("🏠", "Đóng cửa sổ vào giờ cao điểm bụi để giảm PM2.5 xâm nhập."), ("📊", "Theo dõi bản cập nhật tiếp theo để chủ động điều chỉnh lịch sinh hoạt.")],
        "Nhạy cảm": [("🚫", "Nhóm nhạy cảm nên hạn chế hoạt động ngoài trời kéo dài."), ("🏢", "Đóng cửa sổ để giảm thiểu bụi mịn xâm nhập."), ("💗", "Theo dõi dự báo để chủ động bảo vệ sức khỏe.")],
        "Không tốt": [("😷", "Nên đeo khẩu trang lọc bụi mịn nếu cần ra ngoài."), ("🏠", "Hạn chế vận động mạnh ngoài trời và ưu tiên ở trong nhà."), ("📣", "Theo dõi trẻ em, người già và người có bệnh hô hấp sát hơn bình thường.")],
        "Rất Xấu": [("⛔", "Tránh hoạt động ngoài trời không cần thiết trong khung giờ cảnh báo."), ("🪟", "Giữ không khí trong nhà kín và cân nhắc dùng máy lọc không khí."), ("🩺", "Nếu có dấu hiệu khó thở hoặc kích ứng, cần nghỉ ngơi và theo dõi y tế.")],
        "Nguy hiểm": [("⚠️", "Hạn chế tối đa ra ngoài, đặc biệt trẻ em, người già và người có bệnh nền."), ("🏥", "Dùng khẩu trang đạt chuẩn và tìm hỗ trợ y tế nếu có triệu chứng rõ rệt."), ("🔒", "Giữ môi trường trong nhà kín, dùng lọc không khí và tránh nguồn phát sinh bụi.")],
    }
    advice_items = advice_map.get(peak_label, advice_map["Nguy hiểm"])
    advice_html = "".join(
        f'<div class="advice-item"><span class="advice-icon">{icon}</span><span>{escape(text)}</span></div>'
        for icon, text in advice_items
    )

    return (
        '<div class="warning-box">'
        '<div class="warning-title">🔔 Cảnh báo &amp; Nhận định</div>'
        '<div class="warning-main">'
        f'PM2.5 dự kiến <strong>{trend_text}</strong> trong <strong>6 giờ tới</strong> và đạt đỉnh vào khoảng '
        f'<strong>{peak_time_text}</strong> với <span class="warning-highlight">{peak_row["y_pred"]:.1f} µg/m³</span>.'
        '</div>'
        '<div class="warning-main">'
        f'Chất lượng không khí sẽ ở mức <span class="warning-highlight">{peak_label}</span> '
        f'từ <strong>{start_time_text}</strong> đến <strong>{end_time_text}</strong>.'
        '</div>'
        '<div class="warning-section-label">📌 Khuyến nghị:</div>'
        f'<div class="advice-list">{advice_html}</div>'
        '</div>'
    )


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* ─── Global ─── */
        html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
        .stApp { background: #f1f5f9; color: #0f172a; }
        .block-container { max-width: 1440px; padding-top: 1rem; padding-bottom: 1.5rem; }

        /* ─── Sidebar shell ─── */
        section[data-testid="stSidebar"] {
            background: #0f1e36 !important;
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

        /* ─── Sidebar brand ─── */
        .sidebar-brand { display: flex; align-items: center; gap: 10px; padding: 0 0 1.4rem 0; border-bottom: 1px solid rgba(255,255,255,0.07); margin-bottom: 1.2rem; }
        .sidebar-logo {
            width: 42px; height: 42px; border-radius: 12px;
            background: linear-gradient(135deg, #1d4ed8 0%, #06b6d4 100%);
            display: flex; align-items: center; justify-content: center;
            font-size: 18px; flex-shrink: 0;
        }
        .sidebar-title-main { font-size: 15px; font-weight: 800; color: #f8fafc !important; line-height: 1.15; }
        .sidebar-title-sub { font-size: 13px; font-weight: 700; color: #10b981 !important; letter-spacing: 0.05em; }

        /* ─── Sidebar nav radio → styled as nav items ─── */
        div[data-testid="stSidebar"] [data-testid="stRadio"] > label { display: none; }
        div[data-testid="stSidebar"] [data-testid="stRadio"] > div {
            flex-direction: column !important;
            gap: 4px !important;
        }
        div[data-testid="stSidebar"] [data-testid="stRadio"] > div > label {
            background: transparent !important;
            border-radius: 10px !important;
            padding: 10px 14px !important;
            cursor: pointer !important;
            display: flex !important;
            align-items: center !important;
            gap: 10px !important;
            color: rgba(226,232,240,0.75) !important;
            font-size: 14px !important;
            font-weight: 600 !important;
            transition: all 0.15s !important;
            margin: 0 !important;
        }
        div[data-testid="stSidebar"] [data-testid="stRadio"] > div > label:hover {
            background: rgba(255,255,255,0.07) !important;
            color: #f8fafc !important;
        }
        div[data-testid="stSidebar"] [data-testid="stRadio"] > div > label[data-checked="true"],
        div[data-testid="stSidebar"] [data-testid="stRadio"] > div > label[aria-checked="true"] {
            background: #10b981 !important;
            color: #ffffff !important;
        }
        div[data-testid="stSidebar"] [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {
            font-size: 14px !important;
            font-weight: 600 !important;
        }
        /* Hide the circle radio indicator */
        div[data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] { display: none !important; }
        div[data-testid="stSidebar"] [data-testid="stRadio"] > div > label > div:first-child { display: none !important; }

        /* ─── Sidebar section label ─── */
        .sidebar-section-label {
            font-size: 10.5px; font-weight: 700; letter-spacing: 0.10em;
            color: rgba(148,163,184,0.8) !important;
            text-transform: uppercase;
            margin: 1.2rem 0 0.5rem 2px;
        }

        /* ─── Sidebar controls ─── */
        section[data-testid="stSidebar"] [data-baseweb="select"] > div {
            background: #1e3a5f !important;
            border: 1px solid rgba(255,255,255,0.10) !important;
            border-radius: 10px !important;
            box-shadow: none !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"] input { color: #f0f9ff !important; }
        section[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #93c5fd !important; }
        section[data-testid="stSidebar"] .stSelectbox label,
        section[data-testid="stSidebar"] .stSlider label { color: #cbd5e1 !important; font-size: 13px !important; font-weight: 600 !important; }
        section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] { padding: 0 2px; }
        section[data-testid="stSidebar"] [data-testid="stSlider"] [data-testid="stThumbValue"] { color: #38bdf8 !important; font-weight: 700 !important; }



        /* ─── Sidebar footer ─── */
        .sidebar-footer {
            margin-top: 0.9rem;
            padding: 14px 15px;
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.72), rgba(15, 23, 42, 0.58));
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 14px;
            font-size: 12px;
            line-height: 1.6;
            color: #e2e8f0 !important;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
            overflow: hidden;
            word-break: break-word;
        }

        .sidebar-footer-gap {
            height: 8px;
        }

        .sidebar-footer-row {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 6px;
        }

        .sidebar-footer-icon {
            color: #93c5fd !important;
            font-size: 12px;
            line-height: 1;
        }

        .sidebar-footer-label {
            color: #cbd5e1 !important;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }

        .sidebar-footer-value {
            color: #ffffff !important;
            font-weight: 700;
            font-size: 13px;
            line-height: 1.45;
            margin-bottom: 6px;
        }

        .sidebar-footer-note {
            color: #cbd5e1 !important;
            font-size: 11.5px;
            line-height: 1.55;
        }

        /* ─── Page header ─── */
        .page-header {
            display: flex; justify-content: space-between; align-items: flex-start;
            gap: 18px; margin-bottom: 1.5rem; flex-wrap: wrap;
        }
        .page-header-copy { min-width: 280px; flex: 1 1 340px; }
        .page-title { font-size: 28px; font-weight: 800; color: #0f172a; margin: 0; letter-spacing: -0.02em; }
        .page-subtitle { color: #64748b; margin-top: 3px; font-size: 14.5px; font-weight: 500; }
        .header-pills-shell {
            flex: 1 1 640px;
            min-width: 560px;
            width: 100%;
            display: flex;
            justify-content: flex-end;
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 10px 28px rgba(15,23,42,0.05);
            padding: 12px 14px;
        }
        .header-pills { display: flex; gap: 8px; flex-wrap: nowrap; align-items: center; justify-content: flex-end; width: 100%; }
        .header-pill {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 999px;
            padding: 6px 14px;
            font-size: 13.5px;
            color: #475569;
            font-weight: 600;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            white-space: nowrap;
        }
        .header-pill strong { color: #0f172a; }
        .header-pill.accent { background: #f0fdf4; border-color: #bbf7d0; }
        .header-pill.accent strong { color: #065f46; }

        /* ─── Metric cards ─── */
        .metric-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-left-width: 4px;
            border-radius: 16px;
            padding: 1.1rem 1.15rem 1rem;
            min-height: 136px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.04);
            transition: box-shadow 0.2s;
        }
        .metric-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08), 0 16px 32px rgba(0,0,0,0.06); }
        .metric-head { display: flex; align-items: center; gap: 12px; margin-bottom: 0.8rem; }
        .metric-icon {
            width: 40px; height: 40px; border-radius: 12px;
            display: inline-flex; align-items: center; justify-content: center;
            font-size: 17px; font-weight: 700; flex-shrink: 0;
        }
        .metric-title { font-size: 14px; color: #64748b; font-weight: 600; line-height: 1.3; }
        .metric-value { font-size: 18px; font-weight: 800; line-height: 1.2; margin-bottom: 0.75rem; }
        .metric-value strong { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; }
        .metric-unit { font-size: 1rem; font-weight: 600; margin-left: 3px; opacity: 0.85; }
        .metric-foot { display: flex; flex-wrap: wrap; align-items: center; gap: 7px 10px; }
        .metric-badge {
            display: inline-flex; align-items: center;
            padding: 4px 12px; border-radius: 999px;
            font-size: 13px; font-weight: 700; line-height: 1.2;
        }
        .metric-footer { font-size: 12.5px; color: #94a3b8; font-weight: 500; }

        /* ─── Chart + quality panels ─── */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.forecast-chart-anchor),
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.quality-box-anchor) {
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 18px !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.04) !important;
            padding: 0.75rem 0.85rem 1rem !important;
            height: 100%;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.quality-summary-anchor) {
            background: #ffffff !important;
            border: 1px solid #dbe3ef !important;
            border-radius: 22px !important;
            box-shadow: 0 6px 20px rgba(15,23,42,0.06) !important;
            padding: 1rem 1rem 1.15rem !important;
            overflow: hidden !important;
            height: 100%;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.conclusion-box-anchor) {
            background: transparent !important;
            border: none !important;
            border-radius: 0 !important;
            box-shadow: none !important;
            padding: 0 !important;
            height: auto;
        }
        

        /* ─── Panel titles ─── */
        .panel-title { font-size: 16.5px; font-weight: 700; color: #0f172a; margin-bottom: 0.8rem; }
        .panel { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 18px; padding: 1.1rem 1.2rem; margin-bottom: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.04); }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.backtest-box-anchor) {
            background: #ffffff !important;
            border: 1px solid #dbe3ef !important;
            border-radius: 20px !important;
            box-shadow: 0 2px 10px rgba(15,23,42,0.05) !important;
            padding: 1rem 1rem 0.8rem !important;
            height: 100%;
        }

        /* ─── Forecast section title ─── */
        .forecast-section-title { display: flex; align-items: center; gap: 10px; font-size: 16.5px; font-weight: 700; color: #0f172a; }
        .forecast-section-icon { width: 24px; height: 24px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center; background: #eff6ff; color: #2563eb; font-size: 12px; font-weight: 800; flex-shrink: 0; }
        .forecast-toolbar-note { text-align: center; color: #3b5b93; font-size: 14.5px; font-weight: 700; }
        .forecast-table-title { font-size: 15px; font-weight: 700; color: #334155; margin: 10px 0 8px; }

        /* ─── Hour chips ─── */
        .hour-chip {
            background: #fffaf4;
            border: none;
            border-radius: 14px;
            padding: 9px 6px 8px;
            text-align: center;
            min-height: 78px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 4px;
            box-shadow: inset 0 0 0 1px rgba(246,234,211,0.65);
        }
        .hour-chip:hover { box-shadow: none; }
        .hour-time { font-size: 13px; color: #334155; font-weight: 700; }
        .hour-value { font-size: 16px; font-weight: 800; line-height: 1.1; }
        .hour-badge { display: inline-flex; align-items: center; justify-content: center; padding: 3px 6px; border-radius: 999px; font-size: 11.5px; font-weight: 700; line-height: 1.2; width: 100%; }

        /* ─── Quality / compare table ─── */
        .quality-title-row { display: flex; align-items: center; gap: 10px; margin-bottom: 0.85rem; }
        .quality-title-row .panel-title { margin-bottom: 0; }
        .quality-title-icon { width: 28px; height: 28px; border-radius: 9px; display: inline-flex; align-items: center; justify-content: center; background: #f0fdf4; color: #10b981; font-size: 14px; font-weight: 800; flex-shrink: 0; }
        .quality-meta { color: #94a3b8; font-size: 13px; font-weight: 600; margin: 0 0 0.5rem; }
        .legend-row { display: flex; gap: 14px; flex-wrap: wrap; color: #64748b; font-size: 13px; margin-bottom: 0.6rem; }
        .legend-dot { width: 20px; height: 3px; border-radius: 999px; display: inline-block; margin-right: 4px; vertical-align: middle; }

        /* ─── Tables ─── */
        .table-wrap { margin-top: 8px; overflow-x: auto; border-radius: 10px; overflow: hidden; border: 1px solid #e2e8f0; background: #ffffff; }
        .table-wrap table { width: 100%; border-collapse: collapse; font-size: 14px; }
        .table-wrap th { text-align: left; padding: 8px 13px; color: #475569; font-size: 12px; font-weight: 700; border-bottom: 1px solid #e2e8f0; background: #eff6ff; }
        .table-wrap td { padding: 8px 13px; border-bottom: 1px solid #f1f5f9; color: #1e293b; background: #ffffff; }
        .table-wrap tbody tr:last-child td { border-bottom: none; }
        .compare-wrap {
            overflow-x: auto;
            border: 1px solid #e6ded2;
            border-radius: 16px;
            overflow: hidden;
            background: #ffffff;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.95);
        }
        .compare-wrap table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            background: #ffffff;
        }
        .compare-wrap th {
            text-align: left;
            padding: 11px 14px;
            color: #5f4b34;
            font-size: 13px;
            font-weight: 800;
            letter-spacing: 0.01em;
            border-bottom: 1px solid #e6ded2;
            border-right: 1px solid #efe7dc;
            background: linear-gradient(180deg, #faf5ed 0%, #f3ecdf 100%);
        }
        .compare-wrap td {
            padding: 11px 14px;
            border-bottom: 1px solid #f2ece3;
            border-right: 1px solid #f2ece3;
            color: #1e293b;
            background: #ffffff;
            vertical-align: middle;
        }
        .compare-wrap tbody tr:nth-child(even) td {
            background: #fcfaf7;
        }
        .compare-wrap tbody tr:last-child td {
            border-bottom: none;
        }
        .compare-wrap th:last-child,
        .compare-wrap td:last-child {
            border-right: none;
        }
        .forecast-detail-table {
            margin-top: 10px;
            overflow-x: auto;
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid #d9e4f2;
            background: linear-gradient(180deg, #fcfdff 0%, #f8fbff 100%);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.9);
        }
        .forecast-detail-table table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        .forecast-detail-table th {
            text-align: left;
            padding: 11px 14px;
            color: #38506b;
            font-size: 13px;
            font-weight: 800;
            letter-spacing: 0.01em;
            border-bottom: 1px solid #d9e4f2;
            border-right: 1px solid #d9e4f2;
            background: linear-gradient(180deg, #eff5fc 0%, #e6eef8 100%);
        }
        .forecast-detail-table td {
            padding: 11px 14px;
            border-bottom: 1px solid #e9eff7;
            border-right: 1px solid #e9eff7;
            color: #1e293b;
            background: #ffffff;
            vertical-align: middle;
        }
        .forecast-detail-table tbody tr:nth-child(even) td {
            background: #f8fbff;
        }
        .forecast-detail-table tbody tr:hover td {
            background: #f1f7ff;
        }
        .forecast-detail-table tbody tr:last-child td {
            border-bottom: none;
        }
        .forecast-detail-table th:last-child,
        .forecast-detail-table td:last-child {
            border-right: none;
        }
        .forecast-detail-table th:nth-child(2),
        .forecast-detail-table th:nth-child(3),
        .forecast-detail-table th:nth-child(4),
        .forecast-detail-table td:nth-child(2),
        .forecast-detail-table td:nth-child(3),
        .forecast-detail-table td:nth-child(4) {
            text-align: center;
        }
        .forecast-detail-table th:nth-child(5),
        .forecast-detail-table td:nth-child(5) {
            text-align: center;
            white-space: nowrap;
        }
        .forecast-detail-table td strong {
            font-size: 16px;
            color: #0f172a;
        }
        .best-row td { background: #eefbf4 !important; }
        .focus-row td { background: #edf5ff !important; }
        .best-row.focus-row td { background: #e8f8ef !important; }
        .soft-badge { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 12.5px; font-weight: 600; }

        /* ─── Warning box ─── */
        .warning-box {
            background: linear-gradient(180deg, #fffbf5 0%, #fff7ed 100%);
            border: 1px solid #fed7aa;
            border-radius: 18px;
            padding: 1.1rem 1.15rem;
            height: 100%;
        }
        .warning-title { font-size: 16px; font-weight: 700; color: #c2410c; margin-bottom: 12px; display: flex; align-items: center; gap: 6px; }
        .warning-main { background: rgba(255,255,255,0.8); border-radius: 12px; padding: 12px 14px; margin-bottom: 10px; color: #7c2d12; font-size: 15px; line-height: 1.7; border: 1px solid rgba(254,215,170,0.5); }
        .warning-section-label { font-size: 15px; font-weight: 700; color: #c2410c; margin: 10px 0 8px; }
        .warning-highlight { color: #ea580c; font-weight: 700; }
        .advice-list { display: flex; flex-direction: column; gap: 7px; }
        .advice-item { display: flex; align-items: flex-start; gap: 9px; color: #92400e; font-size: 15px; line-height: 1.6; }
        .advice-icon { flex: 0 0 26px; height: 26px; display: inline-flex; align-items: center; justify-content: center; background: rgba(255,255,255,0.8); border-radius: 8px; font-size: 14px; }

        /* ─── Conclusion box ─── */
        .conclusion-box {
            background: linear-gradient(180deg, #f3fff7 0%, #ecfdf5 100%);
            border: 1px solid #bbf7d0;
            border-radius: 16px;
            padding: 1.1rem 1.15rem;
            height: 100%;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 8px 24px rgba(16,185,129,0.06);
        }
        .conclusion-title { font-size: 16px; font-weight: 700; color: #065f46; margin-bottom: 10px; }
        .conclusion-box p { color: #14532d; font-size: 14px; line-height: 1.7; margin: 0 0 8px; }
        .conclusion-box p:last-child { margin-bottom: 0; }
        .conclusion-box code { background: rgba(16,185,129,0.1); border-radius: 5px; padding: 1px 5px; color: #065f46 !important; font-size: 13px; }

        @media (max-width: 900px) {
            .header-pills-shell {
                min-width: 100%;
                justify-content: flex-start;
            }
            .header-pills {
                flex-wrap: wrap;
                justify-content: flex-start;
            }
        }

        /* ─── General buttons ─── */
        div[data-testid="stButton"] > button { border-radius: 9px; border: 1px solid #e2e8f0; color: #334155; font-weight: 600; background: #ffffff; font-size: 14px; transition: all 0.15s; }
        div[data-testid="stButton"] > button:hover { background: #f8fafc; border-color: #cbd5e1; color: #0f172a; }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.forecast-box-anchor) {
            background: #ffffff !important;
            border: 1px solid #dbe3ef !important;
            border-radius: 22px !important;
            box-shadow: 0 6px 20px rgba(15,23,42,0.06) !important;
            padding: 1rem 1rem 1.25rem !important;
            overflow: hidden !important;
        }

        /* tất cả button trong box forecast đều có khung */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.forecast-box-anchor) div[data-testid="stButton"] > button {
            background: #ffffff !important;
            color: #2563eb !important;
            border: 1px solid #dbeafe !important;
            border-radius: 12px !important;
            min-height: 38px !important;
            font-weight: 700 !important;
            box-shadow: 0 1px 2px rgba(15,23,42,0.04) !important;
            padding: 0.35rem 0.7rem !important;
            transition: all 0.18s ease !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.forecast-box-anchor) div[data-testid="stButton"] > button:hover {
            background: #eff6ff !important;
            border-color: #93c5fd !important;
            color: #1d4ed8 !important;
            box-shadow: 0 4px 12px rgba(37,99,235,0.10) !important;
        }

        div[data-testid="stVerticalBlockBorderWrapper"]:has(.forecast-box-anchor) div[data-testid="stButton"] > button:disabled {
            background: #f8fafc !important;
            color: #94a3b8 !important;
            border: 1px solid #e2e8f0 !important;
            box-shadow: none !important;
            opacity: 1 !important;
        }

        /* ─── Divider helper ─── */
        .sidebar-divider { border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 1rem 0; }
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

    # ── Sidebar brand ──────────────────────────────────────────────────────────
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">☁</div>
            <div>
                <div class="sidebar-title-main">Dự báo chất lượng không khí</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar navigation ─────────────────────────────────────────────────────
    page = st.sidebar.radio(
        "Điều hướng",
        ["Dự báo", "  Chất lượng mô hình"],
        label_visibility="collapsed",
    )
    page = "Dự báo " if page == "Dự báo" else "Chất lượng mô hình"

    st.sidebar.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-section-label">Tùy chỉnh dự báo</div>', unsafe_allow_html=True)

    # ── Model selector ─────────────────────────────────────────────────────────
    default_model = (
        best_supported_forecast_model
        if page == "Dự báo " and best_supported_forecast_model is not None
        else str(best_row["Model"])
    )
    model_display = [f"{m} (Best Model)" if m == str(best_row["Model"]) else m for m in model_options]
    default_index = model_options.index(default_model)
    selected_display = st.sidebar.selectbox("Chọn mô hình", model_display, index=default_index)
    selected_model = model_options[model_display.index(selected_display)]

    selected_model_effective = selected_model
    if page == "Dự báo " and not is_forecast_runtime_supported(resolve_bundle_dir(metrics_df, selected_model)):
        if best_supported_forecast_model is None:
            st.error("Không có model nào forecast được trong runtime hiện tại. Các bundle hiện có đều chứa Lambda layer và cần export lại.")
            st.stop()
        selected_model_effective = best_supported_forecast_model
        st.sidebar.warning("Model đang chọn không chạy được forecast ngoài dữ liệu trong runtime hiện tại vì bundle chứa Lambda layer. Dashboard tự chuyển sang model khả dụng tốt nhất.")

    sidebar_bundle_dir = resolve_bundle_dir(metrics_df, selected_model if page == "Chất lượng mô hình" else selected_model_effective)
    max_horizon_steps, bundle_step_hours = load_bundle_runtime_limits(sidebar_bundle_dir)
    max_hours = max_horizon_steps * bundle_step_hours
    default_hours = min(24, max_hours)

    horizon_hours = st.sidebar.slider(f"Số giờ dự báo (tối đa {max_hours}h)", bundle_step_hours, max_hours, default_hours, step=bundle_step_hours)
    horizon_steps = horizon_hours // bundle_step_hours

    history_window_options = [6, 12, 24, 48]
    history_window_labels = [f"{h} giờ" for h in history_window_options]
    selected_history_label = st.sidebar.selectbox("Số giờ lịch sử hiển thị", history_window_labels, index=2)
    selected_history_window = history_window_options[history_window_labels.index(selected_history_label)]

    forecast_page_size = st.sidebar.selectbox("Số mốc forecast mỗi trang", [6, 8, 10, 12], index=1)



    # Footer
    raw_df_for_footer = load_raw_data()
    last_update_str = raw_df_for_footer[DEFAULT_TIMESTAMP_COL].max().strftime("%H:%M:%S - %d/%m/%Y") if not raw_df_for_footer.empty else "N/A"
    num_models = len(metrics_df)
    render_sidebar_footer(page, last_update_str, num_models)

    # ── Active model resolution ────────────────────────────────────────────────
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
    forecast_source = "không áp dụng"

    # ── Forecast page data ─────────────────────────────────────────────────────
    if page == "Dự báo ":
        try:
            history_df, forecast_df, step_hours, forecast_source = load_latest_forecast_or_timeline(bundle_dir, horizon_steps)
        except Exception as exc:
            st.error("Không thể tạo forecast ngoài dữ liệu cho model đang chọn. App đã chặn fallback sang test timeline.")
            st.caption(f"Lỗi chi tiết: {type(exc).__name__}: {exc}")
            st.stop()

        history_df = history_df.tail(selected_history_window).copy()

        chart_key = f"forecast_chart_{active_model}_{horizon_steps}".replace(" ", "_")
        current_val = float(history_df["PM25"].iloc[-1])
        future_val = float(forecast_df["y_pred"].iloc[0])
        delta = future_val - current_val
        delta_pct = (delta / current_val) * 100 if current_val else 0.0
        aqi_val = aqi_from_pm25(future_val)
        quality_label, quality_color, quality_bg = band_info(future_val)
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

    # ── Page header ────────────────────────────────────────────────────────────
    if page == "Dự báo ":
        title = "Hệ thống dự báo PM2.5"
        subtitle = "Ứng dụng Deep Learning cho dự báo chất lượng không khí"
        extra_pills = [
            f'<div class="header-pill accent">Mô hình tốt nhất: <strong>{escape(str(best_row["Model"]))}</strong></div>'
        ]
    else:
        title = "Đánh giá chất lượng mô hình"
        subtitle = "So sánh hiệu suất dự báo của các mô hình đã train"
        extra_pills = []

    header_pills = [
        *extra_pills,
        f'<div class="header-pill">Mô hình đang chọn: <strong>{escape(str(active_model))}</strong></div>',
        f'<div class="header-pill">MAE: <strong>{selected_row["MAE"]:.2f}</strong></div>',
    ]
    render_page_header(title, subtitle, header_pills)

    # ══════════════════════════════════════════════════════════════════════════
    # FORECAST PAGE
    # ══════════════════════════════════════════════════════════════════════════
    if page == "Dự báo ":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            now_label, now_color, now_bg = band_info(current_val)
            metric_card("PM2.5 hiện tại", f"<strong>{current_val:.1f}</strong><span class=\"metric-unit\">µg/m³</span>", now_label, now_color, now_bg, icon="◎", footer=f"Cập nhật lúc {current_time_text}")
        with col2:
            arrow = "↑" if focus_delta > 0.5 else ("↓" if focus_delta < -0.5 else "→")
            metric_card(focus_title, f"<strong>{focus_value:.1f}</strong><span class=\"metric-unit\">µg/m³</span> {arrow}", focus_source_label, focus_color, focus_bg, icon=focus_icon, footer=focus_footer)
        with col3:
            delta_color = "#dc2626" if focus_delta >= 0 else "#16a34a"
            metric_card("Thay đổi so với hiện tại", f"<strong style='color:{delta_color}'>{focus_delta:+.1f}</strong><span class=\"metric-unit\">µg/m³</span>", f"{focus_delta_pct:+.1f}%", "#f97316", "#fff7ed", icon="Δ", footer="Chọn điểm trên biểu đồ để so sánh")
        with col4:
            metric_card("Chất lượng không khí", f"<strong>{focus_label}</strong>", focus_label, focus_color, focus_bg, icon="◌", footer=f"AQI: {focus_aqi}")

        st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)

        # Main forecast chart
        with st.container(border=True):
            st.markdown('<div class="forecast-chart-anchor"></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="panel-title">📈 Diễn biến PM2.5 và dự báo trong {forecast_hours} giờ tới</div>', unsafe_allow_html=True)
            st.markdown(
                '<div style="display:flex;gap:18px;margin-bottom:8px;font-size:14px;color:#64748b;font-weight:600">'
                # '<span><span style="display:inline-block;width:20px;height:3px;background:#3b82f6;border-radius:999px;vertical-align:middle;margin-right:5px"></span>Giá trị thực tế</span>'
                # '<span><span style="display:inline-block;width:20px;height:3px;background:#10b981;border-radius:999px;vertical-align:middle;margin-right:5px;border-top:2px dashed #10b981"></span>Dự báo</span>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(make_forecast_chart(history_df, forecast_df), use_container_width=True, key=chart_key, on_select="rerun", selection_mode="points")

        left, right = st.columns([1.7, 1.3], gap="large")

        with left:
            with st.container():
                st.markdown('<div class="forecast-box-anchor"></div>', unsafe_allow_html=True)

                page_size = int(forecast_page_size)
                total_items = len(forecast_df)

                if "forecast_start_idx" not in st.session_state:
                    st.session_state.forecast_start_idx = 0

                max_start = max(0, total_items - page_size)
                start_idx = st.session_state.forecast_start_idx
                end_idx = min(start_idx + page_size, total_items)

                if "show_forecast_detail" not in st.session_state:
                    st.session_state.show_forecast_detail = False

                def toggle_forecast_detail():
                    st.session_state.show_forecast_detail = not st.session_state.show_forecast_detail

                header_left, header_right = st.columns([5.2, 1.3], vertical_alignment="center")

                with header_left:
                    st.markdown(
                        '<div class="forecast-section-title">'
                        '<span class="forecast-section-icon">⏱</span>'
                        '<span>Dự báo PM2.5 theo giờ</span>'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                with header_right:
                    st.button(
                        "Ẩn chi tiết" if st.session_state.show_forecast_detail else "Xem chi tiết",
                        key="toggle_forecast_detail",
                        use_container_width=True,
                        on_click=toggle_forecast_detail,
                    )

                nav1, nav2, nav3 = st.columns([0.7, 6.6, 0.7], vertical_alignment="center")
                with nav1:
                    if st.button("‹", key="forecast_prev", use_container_width=True, disabled=st.session_state.forecast_start_idx == 0):
                        st.session_state.forecast_start_idx = max(0, st.session_state.forecast_start_idx - page_size)
                with nav2:
                    st.markdown(f'<div class="forecast-toolbar-note">Hiển thị {start_idx + 1}–{end_idx} / {total_items} mốc dự báo</div>', unsafe_allow_html=True)
                with nav3:
                    if st.button("›", key="forecast_next", use_container_width=True, disabled=st.session_state.forecast_start_idx >= max_start):
                        st.session_state.forecast_start_idx = min(max_start, st.session_state.forecast_start_idx + page_size)

                start_idx = st.session_state.forecast_start_idx
                end_idx = min(start_idx + page_size, total_items)
                visible_df = forecast_df.iloc[start_idx:end_idx].copy()

                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                chip_cols = st.columns(page_size, gap="small")
                for idx in range(page_size):
                    with chip_cols[idx]:
                        if idx < len(visible_df):
                            row = visible_df.iloc[idx]
                            chip_card(row[DEFAULT_TIMESTAMP_COL].strftime("%H:%M"), float(row["y_pred"]))
                        else:
                            st.empty()
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

                if st.session_state.show_forecast_detail:
                    st.markdown('<div class="forecast-table-title">Bảng dự báo chi tiết</div>', unsafe_allow_html=True)
                    table_df = visible_df.copy()
                    is_same_day = table_df[DEFAULT_TIMESTAMP_COL].dt.normalize().nunique() == 1
                    table_df["Thời gian"] = table_df[DEFAULT_TIMESTAMP_COL].dt.strftime("%H:%M" if is_same_day else "%d/%m %H:%M")
                    table_df["PM2.5 dự báo"] = table_df["y_pred"].round(1)
                    table_df["Thay đổi"] = table_df["y_pred"].apply(lambda v: f"{v - current_val:+.1f} ({((v - current_val) / current_val) * 100:+.1f}%)" if current_val else "0.0 (0.0%)")
                    table_df["AQI"] = table_df["PM2.5 dự báo"].apply(aqi_from_pm25)

                    rows = ""
                    for _, row in table_df.iterrows():
                        label, color, bg = band_info(float(row["PM2.5 dự báo"]))
                        delta_val = float(row["PM2.5 dự báo"]) - current_val
                        delta_color = "#dc2626" if delta_val >= 0 else "#16a34a"
                        rows += f"""
                        <tr>
                            <td>{row['Thời gian']}</td>
                            <td><strong>{row['PM2.5 dự báo']:.1f}</strong></td>
                            <td style="color:{delta_color};font-weight:600">{row['Thay đổi']}</td>
                            <td>{int(row['AQI'])}</td>
                            <td><span class="soft-badge" style="background:{bg}; color:{color}">{label}</span></td>
                        </tr>
                        """

                    st.markdown(
                        f"""
                        <div class="forecast-detail-table">
                            <table>
                                <thead><tr><th>Thời gian</th><th>PM2.5 dự báo (µg/m³)</th><th>Thay đổi</th><th>AQI</th><th>Mức chất lượng</th></tr></thead>
                                <tbody>{rows}</tbody>
                            </table>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with right:
            st.markdown(build_warning_html(forecast_df, current_val, step_hours), unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # MODEL QUALITY PAGE
    # ══════════════════════════════════════════════════════════════════════════
    else:
        ranking_df = metrics_df.sort_values("MAE", na_position="last").reset_index(drop=True)
        current_rank = int(ranking_df.index[ranking_df["Model"] == active_model][0]) + 1
        rank_badge = "Tốt nhất theo MAE" if active_model == str(best_row["Model"]) else f"Top {current_rank}/{len(ranking_df)}"

        q1, q2, q3, q4 = st.columns(4)
        with q1:
            metric_card("Model tốt nhất", str(best_row["Model"]), rank_badge, "#10b981", "#f0fdf4", icon="🏆")
        with q2:
            metric_card("MAE", f"<strong>{selected_row['MAE']:.2f}</strong><span class='metric-unit'>µg/m³</span>", active_model, "#3b82f6", "#eff6ff", icon="📉")
        with q3:
            metric_card("RMSE", f"<strong>{selected_row['RMSE']:.2f}</strong><span class='metric-unit'>µg/m³</span>", active_model, "#8b5cf6", "#f5f3ff", icon="📊")
        with q4:
            peak_mae = selected_row["Peak MAE"]
            peak_text = f"{peak_mae:.2f}" if pd.notna(peak_mae) else "N/A"
            peak_footer = f"{peak_text} µg/m³" if peak_text != "N/A" else "N/A"
            metric_card("MAPE", f"<strong>{selected_row['MAPE']:.2f}%</strong>", "Peak MAE", "#f59e0b", "#fffbeb", icon="📌", footer=peak_footer)

        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

        left, right = st.columns([1.2, 1.18], gap="large")

        with left:
            rows = ""
            for idx, row in ranking_df.iterrows():
                row_classes: list[str] = []
                badges: list[str] = []
                if idx == 0:
                    row_classes.append("best-row")
                    badges.append('<span class="soft-badge" style="background:#dcfce7; color:#16a34a">🏆 Best</span>')
                if str(row["Model"]) == active_model:
                    row_classes.append("focus-row")
                    badges.append('<span class="soft-badge" style="background:#dbeafe; color:#1d4ed8">Đang xem</span>')
                row_class = " ".join(row_classes)
                badge = f" {'&nbsp;'.join(badges)}" if badges else ""
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
                        <div class="panel-title">Bảng so sánh {len(ranking_df)} mô hình</div>
                    </div>
                    <div class="compare-wrap">
                        <table>
                            <thead><tr><th>Model</th><th>MSE</th><th>MAE</th><th>RMSE</th><th>MAPE</th><th>Peak MAE</th></tr></thead>
                            <tbody>{rows}</tbody>
                        </table>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
                st.markdown('<div class="quality-box-anchor"></div>', unsafe_allow_html=True)
                st.markdown(
                    """
                    <div class="quality-title-row">
                        <div class="quality-title-icon">✦</div>
                        <div class="panel-title">So sánh MAE giữa các mô hình</div>
                    </div>
                    <div class="quality-meta">MAE (µg/m³)</div>
                    <div class="legend-row">
                        <span><span class="legend-dot" style="background:#10b981"></span>Model đang xem</span>
                        <span><span class="legend-dot" style="background:#93c5fd"></span>Model còn lại</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.plotly_chart(make_mae_chart(metrics_df, active_model), use_container_width=True, key=quality_chart_key, on_select="rerun", selection_mode="points", config={"displayModeBar": False})

        with st.container(border=True):
            st.markdown('<div class="quality-summary-anchor"></div>', unsafe_allow_html=True)
            lower_left, lower_right = st.columns([2, 1], gap="large")

            with lower_left:
                
                    st.markdown('<div class="backtest-box-anchor"></div>', unsafe_allow_html=True)
                    st.markdown('<div class="panel-title">📉 Giá trị thực tế vs Dự báo (Model đang chọn)</div>', unsafe_allow_html=True)
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
                        filtered_timeline_df = timeline_df[(timeline_df["timestamp"] >= selected_start) & (timeline_df["timestamp"] <= selected_end)].copy()
                        st.caption(f"Hiển thị {len(filtered_timeline_df)} mốc từ {selected_start.strftime('%d/%m/%Y %H:%M')} đến {selected_end.strftime('%d/%m/%Y %H:%M')}.")
                        st.plotly_chart(make_backtest_chart(filtered_timeline_df, active_model), use_container_width=True)

            with lower_right:
                peak_text = f"{selected_row['Peak MAE']:.2f}" if pd.notna(selected_row["Peak MAE"]) else "N/A"
                _, active_step_hours = load_bundle_runtime_limits(bundle_dir)
                data_start = format_optional_datetime(selected_row.get("Data Start"))
                data_end = format_optional_datetime(selected_row.get("Data End"))
                if data_start and data_end:
                    training_range_html = f"<p>Dữ liệu huấn luyện từ <strong>{data_start}</strong> đến <strong>{data_end}</strong>.</p>"
                else:
                    project_start, project_end = get_project_data_range_text(active_step_hours)
                    training_range_html = (
                        "<p>Bundle chưa lưu metadata <code>train_data_start</code>/<code>train_data_end</code>.</p>"
                        f"<p>Dữ liệu project sau resample {active_step_hours}h: <strong>{project_start}</strong> – <strong>{project_end}</strong>.</p>"
                    )
                summary_line = (
                    f"<strong>{active_model}</strong> hiện là model tốt nhất theo MAE trong {len(ranking_df)} mô hình."
                    if active_model == str(best_row["Model"])
                    else f"<strong>{active_model}</strong> đứng hạng <strong>{current_rank}/{len(ranking_df)}</strong>. Model tốt nhất hiện tại là <strong>{best_row['Model']}</strong>."
                )
                with st.container():
                    st.markdown('<div class="conclusion-box-anchor"></div>', unsafe_allow_html=True)
                    st.markdown(
                        (
                            f'<div class="conclusion-box">'
                            f'<div class="conclusion-title">💡 Kết luận</div>'
                            f"{training_range_html}"
                            f"<p>MAE <strong>{selected_row['MAE']:.2f}</strong> · MSE <strong>{selected_row['MSE']:.2f}</strong> · RMSE <strong>{selected_row['RMSE']:.2f}</strong> · Peak MAE <strong>{peak_text}</strong>.</p>"
                            f"<p>{summary_line}</p>"
                            f"</div>"
                        ),
                        unsafe_allow_html=True,
                    )


if __name__ == "__main__":
    render_app()
