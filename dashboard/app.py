# SOLAR-PAMPA Dashboard — Final Version
# Run with: python -m streamlit run dashboard/app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from pathlib import Path

# --- Page Config ---
st.set_page_config(
    page_title            = "SOLAR-PAMPA Dashboard",
    page_icon             = "☀️",
    layout                = "wide",
    initial_sidebar_state = "expanded"
)

# --- Paths ---
ROOT         = Path(__file__).parent.parent
PREDICTIONS  = ROOT / "models"  / "predictions.parquet"
METRICS      = ROOT / "reports" / "metrics.json"
FEATURES     = ROOT / "data"    / "processed" / "features_weekly.parquet"
FUTURE       = ROOT / "models"  / "future_forecast.parquet"

# --- Load Data ---
@st.cache_data
def load_data():
    predictions = pd.read_parquet(PREDICTIONS)
    predictions["week_start"] = pd.to_datetime(predictions["week_start"])

    features = pd.read_parquet(FEATURES)
    features["week_start"] = pd.to_datetime(features["week_start"])

    with open(METRICS) as f:
        metrics = json.load(f)

    future = pd.DataFrame()
    if FUTURE.exists():
        future = pd.read_parquet(FUTURE)
        future["week_start"] = pd.to_datetime(future["week_start"])

    return predictions, features, metrics, future

predictions, features, metrics, future = load_data()

# --- Sidebar ---
st.sidebar.image("https://flagcdn.com/w80/ar.png", width=60)
st.sidebar.title("☀️ SOLAR-PAMPA")
st.sidebar.markdown("**La Pampa, Argentina**")
st.sidebar.markdown("---")
show_ci       = st.sidebar.checkbox("Show confidence interval", value=True)
show_features = st.sidebar.checkbox("Show feature data",        value=False)

# --- Header ---
st.title("☀️ SOLAR-PAMPA — Solar Forecasting Dashboard")
st.markdown(
    "**Location:** La Pampa, Argentina &nbsp;|&nbsp; "
    "**Lat:** -37.5 &nbsp;|&nbsp; **Lon:** -66.5"
)
st.markdown("---")

# --- Metrics Row ---
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("MAE",         f"{metrics['mae']:.3f} kWh/m²")
col2.metric("RMSE",        f"{metrics['rmse']:.3f} kWh/m²")
col3.metric("MAPE",        f"{metrics['mape_pct']:.1f}%")
col4.metric("R²",          f"{metrics['r2']:.3f}")
col5.metric("CI Coverage", f"{metrics['ci_coverage_pct']:.1f}%")

st.markdown("---")

# --- Three Tabs ---
tab1, tab2, tab3 = st.tabs([
    "📚 Training History (2018–2022)",
    "🔮 Model Predictions (Test Set)",
    "🚀 Future Forecast (Next 4 Weeks)"
])

# ================================================================
# TAB 1 — Training History
# ================================================================
with tab1:
    st.subheader("📚 Full Solar History — Training Data")
    st.caption("All data the model learned from. Select any date range to explore.")

    feat_min  = features["week_start"].min().to_pydatetime()
    feat_max  = features["week_start"].max().to_pydatetime()

    hist_range = st.date_input(
        "History date range",
        value     = [feat_min, feat_max],
        min_value = feat_min,
        max_value = feat_max,
        key       = "hist_range"
    )

    if isinstance(hist_range, (list, tuple)) and len(hist_range) == 2:
        h_start, h_end = hist_range
    else:
        h_start, h_end = feat_min, feat_max

    hist_mask = (
        (features["week_start"].dt.date >= h_start) &
        (features["week_start"].dt.date <= h_end)
    )
    hist_df = features[hist_mask].copy()

    hist_fig = go.Figure()
    hist_fig.add_trace(go.Scatter(
        x         = hist_df["week_start"],
        y         = hist_df["ghi_weekly_kwh"].round(3),
        mode      = "lines",
        name      = "Weekly GHI",
        line      = dict(color="#F4A132", width=1.5),
        fill      = "tozeroy",
        fillcolor = "rgba(244,161,50,0.1)"
    ))
    hist_fig.update_layout(
        xaxis = dict(
            title      = "Week",
            tickformat = "%b %Y",
            tickangle  = -45,
            dtick      = "M3",
        ),
        yaxis     = dict(title="GHI (kWh/m²)"),
        height    = 400,
        hovermode = "x unified",
        margin    = dict(l=60, r=20, t=20, b=60),
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
    )
    st.plotly_chart(hist_fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Weeks shown", f"{len(hist_df)}")
    c2.metric("Avg GHI",     f"{hist_df['ghi_weekly_kwh'].mean():.2f} kWh/m²")
    c3.metric("Peak week",   f"{hist_df['ghi_weekly_kwh'].max():.2f} kWh/m²")
    c4.metric("Lowest week", f"{hist_df['ghi_weekly_kwh'].min():.2f} kWh/m²")

    if show_features:
        st.dataframe(
            hist_df.set_index("week_start"),
            use_container_width=True
        )

# ================================================================
# TAB 2 — Model Predictions
# ================================================================
with tab2:
    st.subheader("🔮 Model Predictions — Test Set")
    st.caption("Weeks the model never saw during training. "
               "This is the real performance evaluation.")

    date_min = predictions["week_start"].min().to_pydatetime()
    date_max = predictions["week_start"].max().to_pydatetime()

    pred_range = st.date_input(
        "Prediction date range",
        value     = [date_min, date_max],
        min_value = date_min,
        max_value = date_max,
        key       = "pred_range"
    )

    if isinstance(pred_range, (list, tuple)) and len(pred_range) == 2:
        p_start, p_end = pred_range
    else:
        p_start, p_end = date_min, date_max

    pred_mask = (
        (predictions["week_start"].dt.date >= p_start) &
        (predictions["week_start"].dt.date <= p_end)
    )
    df = predictions[pred_mask].copy()

    pred_fig = go.Figure()

    if show_ci and not df.empty:
        pred_fig.add_trace(go.Scatter(
            x         = pd.concat([df["week_start"],
                                   df["week_start"][::-1]]),
            y         = pd.concat([df["upper_bound"],
                                   df["lower_bound"][::-1]]),
            fill      = "toself",
            fillcolor = "rgba(70,130,180,0.2)",
            line      = dict(color="rgba(255,255,255,0)"),
            name      = "95% Confidence Interval",
            hoverinfo = "skip"
        ))

    pred_fig.add_trace(go.Scatter(
        x    = df["week_start"],
        y    = df["actual"].round(3),
        mode = "lines",
        name = "Actual",
        line = dict(color="#F4A132", width=2)
    ))
    pred_fig.add_trace(go.Scatter(
        x    = df["week_start"],
        y    = df["predicted"].round(3),
        mode = "lines",
        name = "Predicted",
        line = dict(color="steelblue", width=2, dash="dash")
    ))
    pred_fig.update_layout(
        xaxis = dict(
            title      = "Week",
            tickformat = "%b %Y",
            tickangle  = -45,
            dtick      = "M1",
        ),
        yaxis     = dict(title="GHI (kWh/m²)"),
        hovermode = "x unified",
        height    = 400,
        legend    = dict(orientation="h", y=1.12),
        margin    = dict(l=60, r=20, t=40, b=60),
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
    )
    st.plotly_chart(pred_fig, use_container_width=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test weeks", f"{len(df)}")
    c2.metric("MAE",        f"{metrics['mae']:.3f} kWh/m²")
    c3.metric("R²",         f"{metrics['r2']:.3f}")
    c4.metric("MAPE",       f"{metrics['mape_pct']:.1f}%")

# ================================================================
# TAB 3 — Future Forecast
# ================================================================
with tab3:
    st.subheader("🚀 Future Forecast — Next 4 Weeks")
    st.caption("Model predictions for weeks beyond the dataset. "
               "Based on seasonal patterns learned from 2018-2022.")

    if future.empty:
        st.warning("No future forecast found. Run pipeline.py to generate.")
    else:
        last_actual = predictions.tail(12)[
            ["week_start", "actual"]
        ].rename(columns={"actual": "ghi"})

        fut_fig = go.Figure()

        fut_fig.add_trace(go.Scatter(
            x    = last_actual["week_start"],
            y    = last_actual["ghi"].round(3),
            mode = "lines+markers",
            name = "Recent Actual",
            line = dict(color="#F4A132", width=2)
        ))

        fut_fig.add_trace(go.Scatter(
            x         = pd.concat([future["week_start"],
                                   future["week_start"][::-1]]),
            y         = pd.concat([future["upper_bound"],
                                   future["lower_bound"][::-1]]),
            fill      = "toself",
            fillcolor = "rgba(70,130,180,0.25)",
            line      = dict(color="rgba(255,255,255,0)"),
            name      = "95% Confidence Interval",
            hoverinfo = "skip"
        ))

        fut_fig.add_trace(go.Scatter(
            x    = future["week_start"],
            y    = future["predicted"].round(3),
            mode = "lines+markers",
            name = "Forecast",
            line = dict(color="steelblue", width=2, dash="dash"),
            marker = dict(size=8, symbol="diamond")
        ))

        fut_fig.update_layout(
            xaxis = dict(
                title      = "Week",
                tickformat = "%b %d, %Y",
                tickangle  = -45,
            ),
            yaxis     = dict(title="GHI (kWh/m²)"),
            hovermode = "x unified",
            height    = 420,
            legend    = dict(orientation="h", y=1.12),
            margin    = dict(l=60, r=20, t=40, b=60),
            plot_bgcolor  = "rgba(0,0,0,0)",
            paper_bgcolor = "rgba(0,0,0,0)",
        )
        st.plotly_chart(fut_fig, use_container_width=True)

        st.subheader("📋 Forecast Details")
        display = future[["week_start", "predicted",
                          "lower_bound", "upper_bound"]].copy()
        display.columns = ["Week", "Forecast (kWh/m²)",
                           "Lower Bound", "Upper Bound"]
        display["Week"] = display["Week"].dt.strftime("%Y-%m-%d")
        st.dataframe(display, use_container_width=True, hide_index=True)

# ================================================================
# MAP + RESIDUALS
# ================================================================
st.markdown("---")
col_map, col_res = st.columns([1, 1])

with col_map:
    st.subheader("🗺️ Target Region")
    map_fig = px.scatter_mapbox(
        lat                     = [-37.5],
        lon                     = [-66.5],
        zoom                    = 4,
        height                  = 350,
        size                    = [20],
        color_discrete_sequence = ["red"]
    )
    map_fig.update_layout(
        mapbox_style = "open-street-map",
        margin       = dict(r=0, t=0, l=0, b=0)
    )
    st.plotly_chart(map_fig, use_container_width=True)

with col_res:
    st.subheader("📊 Residual Distribution")
    hist_fig2 = px.histogram(
        predictions, x="residual", nbins=30,
        labels = {"residual": "Residual (kWh/m²)"},
        color_discrete_sequence = ["steelblue"]
    )
    hist_fig2.add_vline(x=0, line_dash="dash", line_color="gray")
    hist_fig2.update_layout(
        height        = 350,
        plot_bgcolor  = "rgba(0,0,0,0)",
        paper_bgcolor = "rgba(0,0,0,0)",
    )
    st.plotly_chart(hist_fig2, use_container_width=True)

st.markdown("---")
st.caption(
    "SOLAR-PAMPA | Solar Output Logistics & "
    "Atmospheric Response Prediction for La Pampa, Argentina"
)