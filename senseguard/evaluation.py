"""Evaluation helpers. Ground-truth labels are joined only after detection."""
from __future__ import annotations

import pandas as pd


def evaluate_detection(raw_data: pd.DataFrame, predictions: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Return binary fault metrics and joined rows for inspection/export."""
    labels = raw_data[["timestamp", "sensor_id", "ground_truth_fault", "ground_truth_fault_type", "ground_truth_event"]].copy()
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    merged = predictions.merge(labels, on=["timestamp", "sensor_id"], how="left", validate="one_to_one")
    valid = merged["ground_truth_fault"].notna()
    true = merged.loc[valid, "ground_truth_fault"].astype(bool)
    pred = merged.loc[valid, "suspected_fault"].astype(bool)

    tp = int((true & pred).sum())
    fp = int((~true & pred).sum())
    fn = int((true & ~pred).sum())
    tn = int((~true & ~pred).sum())
    # If a scenario has no true positives and no predicted positives, these
    # metrics are undefined (N/A), not zero. False-alarm rate remains meaningful.
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) else (0.0 if precision is not None and recall is not None else None)
    false_alarm_rate = fp / (fp + tn) if fp + tn else 0.0
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": false_alarm_rate,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "n_evaluated": int(valid.sum()),
    }
    return metrics, merged
