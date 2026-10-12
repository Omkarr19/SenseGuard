"""Interpretable first-pass checks for sensor-health assessment.

This is a rule-based baseline, not a certified fault diagnosis system.
It intentionally ignores all ground-truth/evaluation columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"timestamp", "sensor_id", "pm25"}


def analyze_sensor_health(data: pd.DataFrame) -> pd.DataFrame:
    """Return row-level detection results from timestamp/sensor_id/pm25 only."""
    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    # Explicitly select only observable inputs; no ground-truth label leakage.
    source = data[["timestamp", "sensor_id", "pm25"]].copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"], errors="coerce")
    source["pm25"] = pd.to_numeric(source["pm25"], errors="coerce")
    source = source.dropna(subset=["timestamp", "sensor_id"])
    source = source.sort_values(["timestamp", "sensor_id"]).reset_index(drop=True)
    if source.empty:
        raise ValueError("No valid readings to analyse")

    # Aggregate duplicate sensor/timestamp pairs conservatively by median.
    wide = source.pivot_table(index="timestamp", columns="sensor_id", values="pm25", aggfunc="median")
    wide = wide.sort_index()
    if wide.shape[1] < 3:
        raise ValueError("At least three comparable sensors are recommended for a robust peer-consensus baseline")

    delta = wide.diff()
    peer_median = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    peer_change = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    network_change = delta.median(axis=1, skipna=True)
    network_median = wide.median(axis=1, skipna=True)
    peer_gap = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    rolling_std = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    rolling_gap_trend = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    relative_change_frame = pd.DataFrame(index=wide.index, columns=wide.columns, dtype=float)
    flatline = pd.DataFrame(False, index=wide.index, columns=wide.columns)
    noise = pd.DataFrame(False, index=wide.index, columns=wide.columns)
    drift = pd.DataFrame(False, index=wide.index, columns=wide.columns)
    spike = pd.DataFrame(False, index=wide.index, columns=wide.columns)

    window = 7
    for sensor in wide.columns:
        peers = wide.drop(columns=[sensor])
        # The cross-sensor median is a robust consensus when most sensors are
        # healthy. With only two channels, neither is a reliable majority.
        pmed = network_median
        peer_median[sensor] = pmed
        peer_change[sensor] = network_change
        peer_gap[sensor] = wide[sensor] - pmed

        # Compare this sensor's change to the network median change. This avoids
        # allowing one spike to distort the comparison baseline for healthy peers.
        relative_change = delta[sensor] - network_change
        relative_change_frame[sensor] = relative_change
        large_change = relative_change.abs() >= 22
        # If a large change immediately reverses the prior large change, treat
        # it as a likely recovery from an isolated spike rather than a second fault.
        reverses_previous = large_change.shift(1, fill_value=False) & (relative_change * relative_change.shift(1) < 0)
        spike[sensor] = large_change & ~reverses_previous

        # Frozen/stuck reading: at least four near-zero changes in five samples.
        zero_change = delta[sensor].abs().lt(1e-9).astype(float)
        flatline[sensor] = zero_change.rolling(5, min_periods=5).sum().ge(4)

        own_std = wide[sensor].rolling(window, min_periods=5).std()
        # Compare temporal variability with the peers' temporal variability,
        # not the cross-sectional spread between sensors. A shared step should
        # not look like three noisy sensors merely because the window spans it.
        peers_roll_std = peers.rolling(window, min_periods=5).std().median(axis=1)
        rolling_std[sensor] = own_std
        # Distinguish sustained erratic movement from a single spike whose
        # rolling standard deviation stays elevated for several samples.
        high_relative_change = relative_change.abs().ge(8).astype(float)
        repeated_large_changes = high_relative_change.rolling(6, min_periods=4).sum().ge(3)
        noise[sensor] = (own_std >= 8.0) & repeated_large_changes & (own_std >= 2.0 * peers_roll_std.fillna(0) + 1.0)

        gap = peer_gap[sensor]
        # Smooth the peer gap before estimating a trend so one isolated spike
        # does not resemble gradual drift. A gap alone is not definitive.
        smooth_gap = gap.rolling(5, min_periods=3).median()
        sustained_gap = gap.abs().ge(14).rolling(5, min_periods=3).sum().ge(3)
        gap_rate = (smooth_gap - smooth_gap.shift(6)) / 6.0
        rolling_gap_trend[sensor] = gap_rate
        drift[sensor] = gap.abs().ge(10) & gap_rate.abs().ge(0.35)
        drift[sensor] = drift[sensor].fillna(False)
        spike[sensor] = spike[sensor].fillna(False)
        flatline[sensor] = flatline[sensor].fillna(False)
        noise[sensor] = noise[sensor].fillna(False)
        # Store sustained peer divergence separately; include drift evidence below.
        peer_gap[sensor] = sustained_gap | drift[sensor]

    # Shared upward change: at least two sensors rise together by a meaningful amount.
    upward = delta.ge(12)
    shared_change_times = upward.sum(axis=1).ge(2)
    shared_change = pd.DataFrame(False, index=wide.index, columns=wide.columns)
    for sensor in wide.columns:
        shared_change[sensor] = shared_change_times & upward[sensor]

    # Construct result records. Each flag is evidence, not proof of a failure.
    records: list[dict] = []
    for sensor in wide.columns:
        for timestamp in wide.index:
            value = wide.at[timestamp, sensor]
            is_missing = pd.isna(value)
            is_flatline = bool(flatline.at[timestamp, sensor])
            is_spike = bool(spike.at[timestamp, sensor])
            is_noise = bool(noise.at[timestamp, sensor])
            is_divergent = bool(peer_gap.at[timestamp, sensor])
            is_shared = bool(shared_change.at[timestamp, sensor])

            reasons = []
            if is_missing:
                reasons.append("Missing reading")
            if is_flatline:
                reasons.append("Repeated unchanged readings")
            if is_spike:
                reasons.append("Isolated sudden change versus peers")
            if is_noise:
                reasons.append("Unusually high rolling variability")
            if is_divergent:
                if bool(drift.at[timestamp, sensor]):
                    reasons.append("Growing peer gap; possible drift")
                else:
                    reasons.append("Persistent peer divergence")
            if is_shared:
                reasons.append("Shared upward change across sensors")

            suspected_fault = bool(is_missing or is_flatline or is_spike or is_noise or is_divergent)
            if is_missing:
                status = "Data Missing"
            elif suspected_fault:
                status = "Needs Review"
            elif is_shared:
                status = "Shared Change Candidate"
            else:
                status = "No Issue Detected"

            # Transparent provisional penalty scale. Multiple indicators add, capped.
            penalty = 0.0
            penalty += 35.0 if is_missing else 0.0
            penalty += 30.0 if is_flatline else 0.0
            penalty += 24.0 if is_spike else 0.0
            penalty += 22.0 if is_noise else 0.0
            penalty += 18.0 if is_divergent else 0.0
            penalty = min(penalty, 65.0)
            records.append({
                "timestamp": timestamp,
                "sensor_id": sensor,
                "pm25": value,
                "missing_data": bool(is_missing),
                "spike": is_spike,
                "flatline": is_flatline,
                "excessive_noise": is_noise,
                "peer_divergence": is_divergent,
                "shared_change_candidate": is_shared,
                "suspected_fault": suspected_fault,
                "status": status,
                "reason": "; ".join(reasons) if reasons else "No major quality indicator detected in this window",
                "instant_penalty": penalty,
            })

    result = pd.DataFrame(records).sort_values(["timestamp", "sensor_id"]).reset_index(drop=True)
    # Health score combines the current evidence with recent evidence; it is an
    # intentionally provisional heuristic, not a probability or calibration claim.
    result["recent_penalty"] = (
        result.groupby("sensor_id")["instant_penalty"]
        .transform(lambda s: s.rolling(8, min_periods=1).mean())
    )
    result["trust_score"] = (
        100.0 - 0.65 * result["instant_penalty"] - 1.7 * result["recent_penalty"]
    ).clip(0, 100).round().astype(int)
    result["score_type"] = "Provisional heuristic"
    return result
