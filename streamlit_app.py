from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.data_utils import CFG, DataConfig
from src.gru_walkforward import (
    forecast_next_hours,
    run_walk_forward_training,
    summarize_daily_forecast,
    train_final_model,
)


st.set_page_config(page_title="PM2.5 Forecast Dashboard", page_icon="🌤", layout="wide")


def pm25_to_aqi(pm25: float) -> int:
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    val = max(0.0, float(pm25))
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= val <= c_high:
            aqi = (i_high - i_low) * (val - c_low) / (c_high - c_low) + i_low
            return int(round(aqi))
    return 500


def aqi_label(aqi: int) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Moderate"
    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    if aqi <= 200:
        return "Unhealthy"
    if aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


@st.cache_data(show_spinner=False)
def run_training_pipeline(
    cfg_dict: dict,
    eval_size: int,
    step_size: int,
    max_folds: int,
    expanding: bool,
    epochs: int,
    batch_size: int,
):
    cfg = DataConfig(**cfg_dict)

    walk_forward_result = run_walk_forward_training(
        cfg=cfg,
        eval_size=eval_size,
        step_size=step_size,
        max_folds=max_folds,
        expanding=expanding,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    final_fit = train_final_model(
        cfg=cfg,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    hourly_forecast = forecast_next_hours(
        model=final_fit["model"],
        artifacts=final_fit["artifacts"],
        hours=72,
    )
    daily_forecast = summarize_daily_forecast(hourly_forecast)

    return {
        "walk_forward": walk_forward_result,
        "final_fit": final_fit,
        "hourly_forecast": hourly_forecast,
        "daily_forecast": daily_forecast,
    }


st.title("Du bao PM2.5 3 ngay toi")
st.caption("Su dung GRU + walk-forward validation de uu tien do sat voi du lieu thuc te.")

with st.sidebar:
    st.header("Cau hinh")

    eval_size = st.number_input("Eval size (gio/fold)", min_value=24, max_value=168, value=72, step=24)
    step_size = st.number_input("Step size (gio)", min_value=12, max_value=72, value=24, step=12)
    max_folds = st.number_input("So fold toi da", min_value=2, max_value=10, value=4, step=1)
    expanding = st.toggle("Expanding window", value=True)

    epochs = st.number_input("Epochs", min_value=5, max_value=100, value=30, step=5)
    batch_size = st.selectbox("Batch size", [16, 32, 64, 128], index=2)

    run_btn = st.button("Train + Forecast", type="primary")

if run_btn:
    with st.spinner("Dang huan luyen walk-forward va du bao 72h..."):
        results = run_training_pipeline(
            cfg_dict=CFG.__dict__,
            eval_size=int(eval_size),
            step_size=int(step_size),
            max_folds=int(max_folds),
            expanding=bool(expanding),
            epochs=int(epochs),
            batch_size=int(batch_size),
        )

    wf = results["walk_forward"]
    final_fit = results["final_fit"]
    hourly = results["hourly_forecast"].copy()
    daily = results["daily_forecast"].copy()

    latest_pm25 = float(hourly.iloc[0]["pm25_pred"])
    latest_aqi = pm25_to_aqi(latest_pm25)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PM2.5 gio dau", f"{latest_pm25:.1f} ug/m3")
    c2.metric("AQI gio dau", f"{latest_aqi}", aqi_label(latest_aqi))
    c3.metric("WF RMSE (mean)", f"{wf['summary']['rmse_mean']:.2f}")
    c4.metric("WF MAPE (mean)", f"{wf['summary']['mape_mean']:.2f}%")

    st.subheader("Walk-forward validation")
    st.dataframe(wf["fold_metrics"], use_container_width=True)

    st.subheader("Du bao theo gio (72h)")
    hourly["aqi"] = hourly["pm25_pred"].map(pm25_to_aqi)
    fig_hourly = px.line(
        hourly,
        x="timestamp",
        y="pm25_pred",
        markers=True,
        title="Du bao PM2.5 theo gio trong 3 ngay toi",
        labels={"timestamp": "Thoi gian", "pm25_pred": "PM2.5 (ug/m3)"},
    )
    st.plotly_chart(fig_hourly, use_container_width=True)

    st.subheader("Bang du bao theo gio")
    hour_table = hourly.copy()
    hour_table["timestamp"] = pd.to_datetime(hour_table["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
    hour_table["pm25_pred"] = hour_table["pm25_pred"].round(1)
    st.dataframe(hour_table.rename(columns={"timestamp": "gio", "pm25_pred": "pm25_du_bao", "aqi": "aqi"}), use_container_width=True)

    st.subheader("Tong hop theo ngay")
    daily["pm25_mean"] = daily["pm25_mean"].round(1)
    daily["pm25_max"] = daily["pm25_max"].round(1)
    daily["pm25_min"] = daily["pm25_min"].round(1)
    st.dataframe(daily.rename(columns={"date": "ngay", "pm25_mean": "tb", "pm25_max": "max", "pm25_min": "min"}), use_container_width=True)

    st.subheader("Chat gay o nhiem hien tai")
    latest_row = final_fit["artifacts"]["df"].iloc[-1]
    p1, p2, p3 = st.columns(3)
    p1.metric("PM2.5", f"{float(latest_row.get('PM25', 0.0)):.1f} ug/m3")
    p2.metric("PM10", f"{float(latest_row.get('PM10', 0.0)):.1f} ug/m3")
    p3.metric("NO2", f"{float(latest_row.get('NO2', 0.0)):.1f} ug/m3")
else:
    st.info("Chon cau hinh ben trai, sau do bam 'Train + Forecast' de chay walk-forward va hien thi dashboard.")
