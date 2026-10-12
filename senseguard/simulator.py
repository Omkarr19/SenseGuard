"""Synthetic PM2.5 streams for the SenseGuard prototype.

Ground-truth fields are created only for evaluation. The detector receives
only timestamp, sensor_id, and pm25, preventing label leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SCENARIOS = {
    "Normal operation": "Normal readings with natural variation.",
    "Shared environmental change": "Several sensors rise together; this should not automatically be labelled as a sensor fault.",
    "Single-sensor spike": "One sensor produces an isolated sudden jump.",
    "Gradual drift": "One sensor slowly moves away from its peers.",
    "Frozen reading": "One sensor repeats the same measurement for a sustained period.",
    "Excessive noise": "One sensor becomes much noisier than comparable sensors.",
    "Missing readings": "One sensor intermittently fails to report a value.",
    "Combined faults": "One sensor develops drift and intermittent missing readings.",
}


def generate_scenario(scenario: str = "Normal operation", n_points: int = 120, seed: int = 7) -> pd.DataFrame:
    """Return long-form synthetic data with known evaluation labels.

    One point represents a five-minute sample. Values are illustrative and are
    not intended to be a physically validated air-quality model.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario {scenario!r}. Choose from: {', '.join(SCENARIOS)}")
    if n_points < 40:
        raise ValueError("n_points must be at least 40 so the fault windows are meaningful")

    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-01 08:00:00", periods=n_points, freq="5min")
    t = np.arange(n_points)
    mid = n_points // 2
    end = min(n_points, mid + max(12, n_points // 5))

    # Common baseline + small sensor-specific differences/noise.
    baseline = 22 + 2.2 * np.sin(2 * np.pi * t / 48) + 0.018 * t
    values = {
        "S-01": baseline + rng.normal(0, 1.4, n_points) - 0.3,
        "S-02": baseline + rng.normal(0, 1.6, n_points) + 0.6,
        "S-03": baseline + rng.normal(0, 1.5, n_points) + 0.1,
    }

    truth_fault = {sensor: np.zeros(n_points, dtype=bool) for sensor in values}
    fault_type = {sensor: np.full(n_points, "none", dtype=object) for sensor in values}
    truth_event = np.zeros(n_points, dtype=bool)
    missing = {sensor: np.zeros(n_points, dtype=bool) for sensor in values}

    if scenario == "Shared environmental change":
        # A temporary shared increase, affecting all sensors similarly.
        for sensor in values:
            values[sensor][mid:end] += 28
        truth_event[mid:end] = True

    elif scenario == "Single-sensor spike":
        values["S-03"][mid] += 55
        truth_fault["S-03"][mid] = True
        fault_type["S-03"][mid] = "spike"

    elif scenario == "Gradual drift":
        drift_start = mid
        drift = np.linspace(0, 32, n_points - drift_start)
        values["S-03"][drift_start:] += drift
        # Label the drift once it becomes meaningfully different from baseline.
        active = np.arange(n_points) >= drift_start + max(3, n_points // 20)
        active &= np.arange(n_points) < n_points
        truth_fault["S-03"][active] = True
        fault_type["S-03"][active] = "drift"

    elif scenario == "Frozen reading":
        frozen_value = values["S-03"][mid - 1]
        values["S-03"][mid:] = frozen_value
        truth_fault["S-03"][mid + 4:] = True
        fault_type["S-03"][mid + 4:] = "flatline"

    elif scenario == "Excessive noise":
        values["S-03"][mid:] += rng.normal(0, 13.5, n_points - mid)
        truth_fault["S-03"][mid:] = True
        fault_type["S-03"][mid:] = "noise"

    elif scenario == "Missing readings":
        idx = np.arange(mid, n_points, 3)
        values["S-03"][idx] = np.nan
        missing["S-03"][idx] = True
        truth_fault["S-03"][idx] = True
        fault_type["S-03"][idx] = "missing"

    elif scenario == "Combined faults":
        drift = np.linspace(0, 27, n_points - mid)
        values["S-03"][mid:] += drift
        active = np.arange(n_points) >= mid + max(3, n_points // 20)
        truth_fault["S-03"][active] = True
        fault_type["S-03"][active] = "drift"
        idx = np.arange(mid + 9, n_points, 11)
        values["S-03"][idx] = np.nan
        missing["S-03"][idx] = True
        truth_fault["S-03"][idx] = True
        fault_type["S-03"][idx] = "missing"

    rows = []
    for sensor, arr in values.items():
        for i, value in enumerate(arr):
            rows.append({
                "timestamp": timestamps[i],
                "sensor_id": sensor,
                "pm25": float(value) if np.isfinite(value) else np.nan,
                # These labels are evaluation-only; detector.py explicitly ignores them.
                "ground_truth_fault": bool(truth_fault[sensor][i]),
                "ground_truth_fault_type": str(fault_type[sensor][i]),
                "ground_truth_event": bool(truth_event[i]),
            })
    return pd.DataFrame(rows)
