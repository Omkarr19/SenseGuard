import pandas as pd

from senseguard.detector import analyze_sensor_health
from senseguard.evaluation import evaluate_detection
from senseguard.simulator import generate_scenario


def test_detector_ignores_ground_truth_columns():
    raw = generate_scenario("Single-sensor spike", n_points=72, seed=3)
    a = analyze_sensor_health(raw)
    b = analyze_sensor_health(raw.drop(columns=["ground_truth_fault", "ground_truth_fault_type", "ground_truth_event"]))
    pd.testing.assert_frame_equal(a, b)


def test_shared_event_is_not_whole_period_marked_as_fault():
    raw = generate_scenario("Shared environmental change", n_points=120, seed=7)
    predictions = analyze_sensor_health(raw)
    # A shared event is useful to show on the chart; it should not by itself
    # mark every sensor reading in the interval as a fault.
    event_rows = predictions[predictions["shared_change_candidate"]]
    assert not event_rows.empty
    assert not event_rows["suspected_fault"].all()


def test_missing_readings_are_detected():
    raw = generate_scenario("Missing readings", n_points=120, seed=7)
    predictions = analyze_sensor_health(raw)
    missing = predictions[predictions["pm25"].isna()]
    assert not missing.empty
    assert missing["missing_data"].all()
    assert missing["status"].eq("Data Missing").all()


def test_metrics_have_expected_keys_and_range():
    raw = generate_scenario("Single-sensor spike", n_points=120, seed=7)
    predictions = analyze_sensor_health(raw)
    metrics, _ = evaluate_detection(raw, predictions)
    for key in ("precision", "recall", "f1", "false_alarm_rate"):
        assert metrics[key] is not None
        assert 0.0 <= metrics[key] <= 1.0


def test_positive_class_metrics_are_na_when_no_faults_or_alerts():
    raw = generate_scenario("Normal operation", n_points=72, seed=7)
    predictions = analyze_sensor_health(raw)
    metrics, _ = evaluate_detection(raw, predictions)
    assert metrics["precision"] is None
    assert metrics["recall"] is None
    assert metrics["f1"] is None
    assert metrics["false_alarm_rate"] == 0.0


def test_all_scenarios_return_valid_shapes():
    for scenario in [
        "Normal operation", "Shared environmental change", "Single-sensor spike",
        "Gradual drift", "Frozen reading", "Excessive noise", "Missing readings", "Combined faults",
    ]:
        raw = generate_scenario(scenario, n_points=72, seed=21)
        predictions = analyze_sensor_health(raw)
        assert len(raw) == 72 * 3
        assert len(predictions) == len(raw)
        assert predictions["trust_score"].between(0, 100).all()
