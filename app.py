from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from senseguard.detector import analyze_sensor_health
from senseguard.evaluation import evaluate_detection
from senseguard.simulator import SCENARIOS, generate_scenario

st.set_page_config(page_title="SenseGuard | Sensor Health", page_icon="🟢", layout="wide")

st.title("SenseGuard")
st.caption("Event-aware sensor trust and fault detection · Synthetic PM2.5 prototype")
st.info(
    "Prototype status: rule-based baseline using simulated readings. The health score is a provisional heuristic, "
    "not a probability. This tool does not replace calibrated monitoring or official safety alarms."
)

with st.sidebar:
    st.header("Demo controls")
    scenario = st.selectbox("Scenario", list(SCENARIOS.keys()))
    n_points = st.slider("Samples per sensor", min_value=60, max_value=240, value=120, step=12)
    seed = st.number_input("Simulation seed", min_value=1, max_value=99999, value=7, step=1)
    st.caption(SCENARIOS[scenario])
    st.caption("Changing these controls updates the simulation.")

# Recompute deterministically for the selected controls on every rerun. The button
# provides an explicit action and also works naturally with Streamlit's rerun model.
@st.cache_data(show_spinner=False)
def build_results(selected_scenario: str, points: int, random_seed: int):
    raw = generate_scenario(selected_scenario, n_points=points, seed=random_seed)
    predictions = analyze_sensor_health(raw)
    metrics, evaluated = evaluate_detection(raw, predictions)
    return raw, predictions, metrics, evaluated

raw, results, metrics, evaluated = build_results(scenario, int(n_points), int(seed))

def format_metric(value):
    return "N/A" if value is None else f"{value:.0%}"

# Overview numbers use the latest row for each sensor, rather than adding a truth label to the detector.
latest = results.sort_values("timestamp").groupby("sensor_id", as_index=False).tail(1).sort_values("sensor_id")
cols = st.columns(4)
cols[0].metric("Sensors", str(raw["sensor_id"].nunique()))
cols[1].metric("Latest readings flagged", str(int(latest["suspected_fault"].sum())))
cols[2].metric("Latest needs review", str(int((latest["status"] == "Needs Review").sum())))
cols[3].metric("Shared-change candidates", str(int(results["shared_change_candidate"].sum())))

st.subheader("PM2.5 readings over time")
fig = go.Figure()
for sensor in sorted(results["sensor_id"].unique()):
    sensor_data = results[results["sensor_id"] == sensor]
    fig.add_trace(go.Scatter(
        x=sensor_data["timestamp"], y=sensor_data["pm25"], mode="lines",
        name=sensor, connectgaps=False,
        hovertemplate="%{x}<br>PM2.5: %{y:.1f} µg/m³<extra>%{fullData.name}</extra>",
    ))
flagged = results[results["suspected_fault"] & results["pm25"].notna()]
if not flagged.empty:
    fig.add_trace(go.Scatter(
        x=flagged["timestamp"], y=flagged["pm25"], mode="markers", name="Flagged for review",
        marker={"symbol": "x", "size": 11, "color": "#e45756", "line": {"width": 2}},
        customdata=flagged[["sensor_id", "reason"]],
        hovertemplate="%{x}<br>%{customdata[0]}: %{y:.1f}<br>%{customdata[1]}<extra>Review</extra>",
    ))
shared = results[results["shared_change_candidate"] & results["pm25"].notna()]
if not shared.empty:
    fig.add_trace(go.Scatter(
        x=shared["timestamp"], y=shared["pm25"], mode="markers", name="Shared-change candidate",
        marker={"symbol": "diamond-open", "size": 10, "color": "#2ca58d", "line": {"width": 2}},
        customdata=shared[["sensor_id", "reason"]],
        hovertemplate="%{x}<br>%{customdata[0]}: %{y:.1f}<br>%{customdata[1]}<extra>Shared change</extra>",
    ))
fig.update_layout(
    height=430, margin={"l": 10, "r": 10, "t": 20, "b": 10},
    xaxis_title="Timestamp", yaxis_title="PM2.5 (µg/m³)",
    legend_title="Series", hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)
st.caption("Red crosses indicate readings flagged by the rule-based baseline. Green diamonds indicate simultaneous upward changes across sensors. Both are evidence signals, not definitive diagnoses.")

left, right = st.columns([1.2, 1])
with left:
    st.subheader("Latest sensor-health assessment")
    view = latest[["sensor_id", "pm25", "trust_score", "status", "reason"]].copy()
    view["pm25"] = view["pm25"].round(1)
    view = view.rename(columns={"sensor_id": "Sensor", "pm25": "PM2.5", "trust_score": "Health score", "status": "Status", "reason": "Evidence"})
    st.dataframe(view, hide_index=True, use_container_width=True)
    st.caption("Health scores are provisional, rule-derived indicators based on current and recent evidence.")
with right:
    st.subheader("Evaluation on this synthetic scenario")
    m1, m2 = st.columns(2)
    m1.metric("Precision", format_metric(metrics["precision"]))
    m2.metric("Recall", format_metric(metrics["recall"]))
    m3, m4 = st.columns(2)
    m3.metric("F1 score", format_metric(metrics["f1"]))
    m4.metric("False-alarm rate", f"{metrics['false_alarm_rate']:.0%}")
    st.caption(
        "N/A means the selected scenario has no positive fault examples for that metric. "
        f"Counts: TP {metrics['true_positives']} · FP {metrics['false_positives']} · "
        f"FN {metrics['false_negatives']} · TN {metrics['true_negatives']} · "
        f"N={metrics['n_evaluated']}. These are generated on the selected scenario, not hidden-test or field results."
    )

with st.expander("Inspect flagged readings"):
    flagged_table = evaluated[evaluated["suspected_fault"] | evaluated["shared_change_candidate"]].copy()
    columns = ["timestamp", "sensor_id", "pm25", "status", "reason", "trust_score", "ground_truth_fault_type", "ground_truth_event"]
    st.dataframe(flagged_table[columns], hide_index=True, use_container_width=True)
    st.caption("Ground-truth columns are shown here only for local evaluation of synthetic data. They are not inputs to the detector.")

col1, col2 = st.columns(2)
with col1:
    st.download_button("Download synthetic input CSV", data=raw.to_csv(index=False).encode("utf-8"), file_name="senseguard_synthetic_input.csv", mime="text/csv", use_container_width=True)
with col2:
    st.download_button("Download analysis CSV", data=evaluated.to_csv(index=False).encode("utf-8"), file_name="senseguard_analysis_results.csv", mime="text/csv", use_container_width=True)

st.divider()
st.caption("Scope: synthetic air-quality data only. Not a certified monitor, not field-validated, and not a replacement for official safety alarms.")
