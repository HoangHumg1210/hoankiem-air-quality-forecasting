from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bundle_registry import dedupe_model_metrics, load_registry_metrics as load_bundle_registry_metrics
from inference import DEFAULT_TIMESTAMP_COL, forecast_recursive, load_model_bundle, prepare_raw_frame


APP_DIR = Path(__file__).resolve().parent
REGISTRY_DIR = APP_DIR / "model_registry"
BEST_BUNDLE_DIR = APP_DIR / "best_model_bundle"
RAW_DATA_PATH = APP_DIR / "data" / "processed" / "data2225_done.csv"
FUTURE_LOOKBACK_DAYS = 14


#  Quy đổi PM2.5 sang AQI
AQI_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 500.0, 301, 500),
]

AQI_LEVELS = [
    {"range": (0, 50), "label": "Tốt", "color": "#66cc66"},
    {"range": (51, 100), "label": "Trung bình", "color": "#eec900"},
    {"range": (101, 150), "label": "Không tốt cho nhóm nhạy cảm", "color": "#FF7F24"},
    {"range": (151, 200), "label": "Xấu", "color": "#CD2626"},
    {"range": (201, 300), "label": "Rất Xấu", "color": "#CD2626"},
    {"range": (301, 500), "label": "Nguy hiểm", "color": "#b03060"},
]
#  Quy đổi PM2.5 sang AQI
AQI_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 500.0, 301, 500),
]

AQI_LEVELS = [
    {"range": (0, 50), "label": "Tốt", "color": "#66cc66"},
    {"range": (51, 100), "label": "Trung bình", "color": "#eec900"},
    {"range": (101, 150), "label": "Không tốt cho nhóm nhạy cảm", "color": "#FF7F24"},
    {"range": (151, 200), "label": "Xấu", "color": "#CD2626"},
    {"range": (201, 300), "label": "Rất Xấu", "color": "#CD2626"},
    {"range": (301, 500), "label": "Nguy hiểm", "color": "#b03060"},
]

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
    {"range": (0, 50), "label": "Tốt", "color": "#19f019"},
    {"range": (51, 100), "label": "Trung bình", "color": "#f5d313"},
    {"range": (101, 150), "label": "Không tốt cho nhóm nhạy cảm", "color": "#F36D0E"},
    {"range": (151, 200), "label": "Xấu", "color": "#F70303"},
    {"range": (201, 300), "label": "Rất Xấu", "color": "#8D2544"},
    {"range": (301, 500), "label": "Nguy hiểm", "color": "#6B0320"},
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
            return str(level["label"]), color, f"{color}22"
    return "Nguy hiem", "#7E0023", "#7E002322"


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
    try:
        history_df, forecast_df, step_hours = load_latest_forecast(bundle_dir, horizon_steps)
        return history_df, forecast_df, step_hours, "forecast_ngoai_du_lieu"
    except Exception:
        timeline_df = load_bundle_timeline(bundle_dir)
        if timeline_df.empty:
            raise

        step_hours = 3
        max_steps = min(horizon_steps, len(timeline_df))
        forecast_df = timeline_df.tail(max_steps).copy()
        forecast_df = forecast_df.rename(columns={"timestamp": DEFAULT_TIMESTAMP_COL})
        forecast_df = forecast_df[[DEFAULT_TIMESTAMP_COL, "y_true", "y_pred"]].copy()
        forecast_df["step"] = range(1, len(forecast_df) + 1)

        raw_df = prepare_raw_frame(load_raw_data(), step_hours=step_hours).reset_index()
        start_ts = forecast_df[DEFAULT_TIMESTAMP_COL].min()
        history_df = raw_df[raw_df[DEFAULT_TIMESTAMP_COL] < start_ts].tail(14).copy()
        return history_df, forecast_df, step_hours, "test_timeline"


def make_forecast_chart(history_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    zones = [
        (0, 15, "Tốt", "rgba(234,243,222,0.85)"),
        (15, 35, "Trung bình", "rgba(250,238,218,0.75)"),
        (35, 55, "Nhạy cảm", "rgba(250,199,117,0.20)"),
        (55, 75, "Không tốt", "rgba(252,235,235,0.80)"),
        (75, 120, "Rất xấu", "rgba(250,236,231,0.85)"),
    ]
    for lower, upper, _, fill in zones:
        fig.add_hrect(y0=lower, y1=upper, fillcolor=fill, line_width=0)

    fig.add_trace(
        go.Scatter(
            x=history_df[DEFAULT_TIMESTAMP_COL],
            y=history_df["PM25"],
            mode="lines+markers",
            name="Lịch sử",
            line=dict(color="#378ADD", width=3),
            marker=dict(size=7, color="#378ADD"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_df[DEFAULT_TIMESTAMP_COL],
            y=forecast_df["y_pred"],
            mode="lines+markers",
            name="Dự báo",
            line=dict(color="#1D9E75", width=3, dash="dot"),
            marker=dict(size=7, color="#1D9E75"),
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
    fig.add_vline(x=current_x, line_dash="dot", line_color="rgba(55,138,221,0.55)", line_width=1.5)
    fig.add_annotation(
        x=current_x,
        y=max(float(forecast_df["y_pred"].max()) + 15, 70),
        text="Mốc dự báo",
        showarrow=False,
        bgcolor="#E6F1FB",
        bordercolor="#B5D4F4",
        font=dict(size=10, color="#185FA5"),
    )
    fig.add_annotation(
        x=peak[DEFAULT_TIMESTAMP_COL],
        y=peak["y_pred"] + 6,
        text=f"{peak[DEFAULT_TIMESTAMP_COL].strftime('%d/%m %H:%M')}<br><b>{peak['y_pred']:.1f} µg/m³</b>",
        showarrow=True,
        arrowcolor="#1D9E75",
        bgcolor="#ECFDF5",
        bordercolor="#A7F3D0",
        font=dict(size=11, color="#0F6E56"),
    )

    fig.update_layout(
        height=360,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.08, x=0),
        xaxis=dict(title="", showgrid=False, tickformat="%d/%m %H:%M"),
        yaxis=dict(title="PM2.5 (µg/m³)", gridcolor="rgba(0,0,0,0.05)"),
    )
    return fig


def make_mae_chart(metrics_df: pd.DataFrame, selected_model: str) -> go.Figure:
    colors = ["#378ADD" if model == selected_model else "#B5D4F4" for model in metrics_df["Model"]]
    fig = go.Figure(
        go.Bar(
            x=metrics_df["Model"],
            y=metrics_df["MAE"],
            text=metrics_df["MAE"].round(2),
            textposition="outside",
            marker=dict(color=colors),
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis=dict(title=""),
        yaxis=dict(title="MAE (µg/m³)", gridcolor="rgba(0,0,0,0.05)"),
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


def metric_card(title: str, value: str, badge: str, accent: str, badge_bg: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card" style="border-left-color:{accent}">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="color:{accent}">{value}</div>
            <div class="metric-badge" style="background:{badge_bg}; color:{accent}">{badge}</div>
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
        .metric-card { background: #ffffff; border: 1px solid rgba(0,0,0,0.08); border-left-width: 3px; border-radius: 12px; padding: 1rem; min-height: 122px; }
        .metric-title { font-size: 12px; color: #68778f; margin-bottom: 8px; }
        .metric-value { font-size: 22px; font-weight: 700; line-height: 1.25; margin-bottom: 10px; }
        .metric-badge { display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
        .panel { background: #ffffff; border: 1px solid rgba(0,0,0,0.08); border-radius: 12px; padding: 1rem 1.1rem; margin-bottom: 1rem; }
        .panel-title { font-size: 16px; font-weight: 700; color: #1c2e4a; margin-bottom: 0.85rem; }
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
        .table-wrap table, .compare-wrap table { width: 100%; border-collapse: collapse; font-size: 13px; }
        .table-wrap th, .compare-wrap th { text-align: left; padding: 8px 10px; color: #68778f; font-size: 12px; border-bottom: 1px solid rgba(0,0,0,0.08); }
        .table-wrap td, .compare-wrap td { padding: 8px 10px; border-bottom: 1px solid rgba(0,0,0,0.06); color: #1c2e4a; }
        .best-row { background: #ebfaf1; }
        .soft-badge { display: inline-block; padding: 4px 9px; border-radius: 999px; font-size: 11px; font-weight: 600; }
        .warning-box { background: #faeeda; border: 1px solid #fac775; border-radius: 12px; padding: 1rem; height: 100%; }
        .warning-title { font-size: 16px; font-weight: 700; color: #633806; margin-bottom: 10px; }
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
    default_index = model_options.index(best_row["Model"])
    selected_model = st.sidebar.selectbox("Chọn mô hình", model_options, index=default_index)
    horizon_steps = st.sidebar.slider("Số bước dự báo (mỗi bước 3 giờ, tối đa 72h)", 4, 24, 24, step=4)
    history_window = st.sidebar.selectbox("Số điểm lịch sử hiển thị", [14, 24, 32], index=0)
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

    bundle_dir = resolve_bundle_dir(metrics_df, selected_model)
    selected_row = metrics_df.loc[metrics_df["Model"] == selected_model].iloc[0]
    timeline_df = load_bundle_timeline(bundle_dir)

    history_df, forecast_df, step_hours, forecast_source = load_latest_forecast_or_timeline(bundle_dir, horizon_steps)
    history_df = history_df.tail(history_window).copy()

    current_val = float(history_df["PM25"].iloc[-1])
    future_val = float(forecast_df["y_pred"].iloc[0])
    delta = future_val - current_val
    delta_pct = (delta / current_val) * 100 if current_val else 0.0
    aqi_val = aqi_from_pm25(future_val)
    quality_label, _, _ = band_info(future_val)
    peak_row = forecast_df.loc[forecast_df["y_pred"].idxmax()]

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
                <div class="header-pill">Model: <strong>{selected_model}</strong></div>
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
            now_label, _, _ = band_info(current_val)
            metric_card("PM2.5 hiện tại", f"{current_val:.1f} µg/m³", now_label, "#185FA5", "#E6F1FB")
        with col2:
            metric_card(
                f"Dự báo bước kế tiếp (+{step_hours}h)",
                f"{future_val:.1f} µg/m³",
                quality_label,
                "#534AB7",
                "#EEEDFE",
            )
        with col3:
            metric_card(
                "Thay đổi so với hiện tại",
                f"{delta:+.1f} µg/m³",
                f"{delta_pct:+.2f}%",
                "#854F0B",
                "#FAEEDA",
            )
        with col4:
            metric_card("Chất lượng không khí", quality_label, f"AQI {aqi_val}", "#A32D2D", "#FCEBEB")

        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Lịch sử và dự báo theo model đang chọn</div>
                <div class="legend-row">
                    <span><span class="legend-dot" style="background:#378ADD"></span>Lịch sử</span>
                    <span><span class="legend-dot" style="background:#1D9E75"></span>Dự báo</span>
                    <span><span class="legend-dot" style="background:#EF4444"></span>Thực tế tương lai nếu có</span>
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(make_forecast_chart(history_df, forecast_df), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        left, right = st.columns([2.2, 1], gap="large")
        
        with left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)

            # ===== State cho slider trái/phải =====
            page_size = 8
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
            actual_note = (
                f"Sai số bước đầu: {abs(float(forecast_df['y_true'].iloc[0]) - future_val):.2f} µg/m³."
                if "y_true" in forecast_df.columns and forecast_df["y_true"].notna().any()
                else "Không có y_true vì đây là forecast ngoài phạm vi dữ liệu đã có."
            )
            st.markdown(
                f"""
                <div class="warning-box">
                    <div class="warning-title">Nhận định theo model</div>
                    <p>{selected_model} đang dự báo đỉnh tại <strong>{peak_row[DEFAULT_TIMESTAMP_COL].strftime('%d/%m %H:%M')}</strong> với <strong>{peak_row['y_pred']:.1f} µg/m³</strong>.</p>
                    <p>Bundle hiện dùng có MAE <strong>{selected_row['MAE']:.2f}</strong>, RMSE <strong>{selected_row['RMSE']:.2f}</strong>, Peak MAE <strong>{selected_row['Peak MAE']:.2f}</strong>.</p>
                    <p>{actual_note}</p>
                    <p><strong>Ghi chú:</strong> forecast 72h đang dùng covariates tương lai được ngoại suy từ lịch sử gần nhất của dataset.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        q1, q2, q3, q4 = st.columns(4)
        with q1:
            metric_card("Model tốt nhất", str(best_row["Model"]), "Theo MAE", "#185FA5", "#E6F1FB")
        with q2:
            metric_card("MAE", f"{selected_row['MAE']:.2f} µg/m³", selected_model, "#0F6E56", "#E1F5EE")
        with q3:
            metric_card("RMSE", f"{selected_row['RMSE']:.2f} µg/m³", selected_model, "#534AB7", "#EEEDFE")
        with q4:
            peak_mae = selected_row["Peak MAE"]
            peak_text = f"{peak_mae:.2f}" if pd.notna(peak_mae) else "N/A"
            metric_card("MAPE / Peak MAE", f"{selected_row['MAPE']:.2f}%", peak_text, "#854F0B", "#FAEEDA")

        left, right = st.columns([1.1, 1], gap="large")
        with left:
            rows = ""
            table_df = metrics_df.sort_values("MAE", na_position="last").reset_index(drop=True)
            for idx, row in table_df.iterrows():
                row_class = "best-row" if idx == 0 else ""
                badge = ' <span class="soft-badge" style="background:#EAF3DE; color:#3B6D11">best</span>' if idx == 0 else ""
                
                rows += f"""
                <tr class="{row_class}">
                    <td><strong>{row['Model']}</strong>{badge}</td>
                    <td>{row['MAE']:.2f}</td>
                    <td>{row['MSE']:.2f}</td>
                    <td>{row['RMSE']:.2f}</td>
                    <td>{row['MAPE']:.2f}%</td>
                    <td>{row['Peak MAE']:.2f}</td>
           
                </tr>
                """
            st.markdown('<div class="panel"><div class="panel-title">Bảng metrics các model</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
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
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown(
                """
                <div class="panel">
                    <div class="panel-title">So sánh MAE giữa các model</div>
                    <div class="legend-row">
                        <span><span class="legend-dot" style="background:#378ADD"></span>Model đang chọn</span>
                        <span><span class="legend-dot" style="background:#B5D4F4"></span>Model còn lại</span>
                    </div>
                """,
                unsafe_allow_html=True,
            )
            st.plotly_chart(make_mae_chart(metrics_df, selected_model), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

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
                    key=f"backtest_range_{selected_model}",
                )
                filtered_timeline_df = timeline_df[
                    (timeline_df["timestamp"] >= selected_start)
                    & (timeline_df["timestamp"] <= selected_end)
                ].copy()
                st.caption(
                    f"Hiển thị {len(filtered_timeline_df)} mốc từ "
                    f"{selected_start.strftime('%d/%m/%Y %H:%M')} đến {selected_end.strftime('%d/%m/%Y %H:%M')}."
                )
                st.plotly_chart(make_backtest_chart(filtered_timeline_df, selected_model), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with lower_right:
            peak_text = f"{selected_row['Peak MAE']:.2f}" if pd.notna(selected_row["Peak MAE"]) else "N/A"
            st.markdown(
                f"""
                <div class="conclusion-box">
                    <div class="conclusion-title">Kết luận</div>
                    <p>Dashboard hiện đã nối trực tiếp vào bundle model và forecast 72h sau điểm cuối của dữ liệu hiện có.</p>
                    <p>Model đang chọn là <strong>{selected_model}</strong> với MAE <strong>{selected_row['MAE']:.2f}</strong>, RMSE <strong>{selected_row['RMSE']:.2f}</strong>, Peak MAE <strong>{peak_text}</strong>.</p>
                    <p>Các covariates tương lai hiện được ngoại suy từ mẫu gần nhất theo giờ/ngày, nên phù hợp cho forecast offline nhưng chưa phải nguồn tương lai thực đo.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    


if __name__ == "__main__":
    render_app()
