# SenseGuard — Prototype starter

A small software-only prototype for the Code_Apex Sensor Trust / Drift / False-Alarm Detection problem. It uses synthetic PM2.5 readings from three comparable sensors and an interpretable rule-based baseline.

> **Important limitations:** all readings are simulated; the current health score is a provisional heuristic; the rules are a baseline, not a diagnosis; this is not field-validated or certified safety equipment. Never use it to suppress official alarms.

## Features in this starter

- Synthetic scenarios: normal, shared environmental change, single spike, gradual drift, frozen readings, excessive noise, missing readings, combined faults.
- Rule-based indicators for missing data, isolated spikes, prolonged flatlining, excessive variability and persistent peer divergence.
- Shared upward-change marker when at least two sensors rise together.
- Provisional 0–100 health score with evidence/reason codes and a `Needs Review` state.
- Streamlit dashboard with Plotly chart, current sensor table, metrics, and CSV downloads.
- Evaluation labels kept separate from detector inputs to avoid label leakage.
- Pytest checks for core assumptions.

The optional Isolation Forest model is deliberately **not part of the first baseline yet**. First validate the simulator, rule logic and evaluation path; then compare a model against the rules-only baseline.

## Setup on Windows (PowerShell)

1. Install Python 3.12 from https://www.python.org/downloads/ and enable **Add Python to PATH** during setup. Python 3.11 should also work.
2. Extract this folder to a simple path, for example `D:\Projects\senseguard_starter`.
3. Open PowerShell in that folder.
4. Create and activate a virtual environment:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If script activation is blocked by your environment, run commands using `\.venv\Scripts\python.exe` directly, or use Command Prompt activation `\.venv\Scripts\activate.bat` in `cmd.exe`.

5. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Start the dashboard:

   ```powershell
   python -m streamlit run app.py
   ```

   Streamlit will print a local URL, normally `http://localhost:8501`.

7. Select each scenario from the left sidebar and inspect the chart, explanations, and evaluation metrics.

## Run the tests

```powershell
python -m pytest -q
```

## Repository structure

```text
app.py                      Streamlit UI
senseguard/simulator.py     Synthetic data + evaluation labels
senseguard/detector.py      Rule-based assessment (labels are ignored)
senseguard/evaluation.py    Metrics after predictions are produced
tests/test_detector.py      Core behaviour checks
requirements.txt            Dependencies
```

## How the score works

The current score subtracts documented penalties for detected evidence such as missing data, flatlining, isolated spikes, excessive variability and persistent peer divergence. Penalties combine current evidence with a rolling recent penalty. The result is a transparent, **heuristic demo score**, not a probability that a sensor is correct and not a calibrated measure of physical accuracy. We should tune it only against development scenarios and report results on held-out scenarios separately.

## Suggested next development steps

1. Review detections in every supplied scenario and note false positives/false negatives.
2. Tune the rules using a development seed/scenario set; reserve other seeds and new combinations for final evaluation.
3. Add explicit detection-delay evaluation for injected faults.
4. Add Isolation Forest only if it improves held-out results over the rules-only baseline.
5. Add a separate live-data adapter only if we later obtain hardware. For the proposed air-quality hardware extension, use comparable PM sensors; a BME280 gives temperature/humidity/pressure context but does not measure PM2.5.

## Data safety / honesty

Do not describe the app as real-time, hardware-connected, accurate on real-world sensors, or production-ready unless those capabilities have actually been implemented and independently validated. Metrics shown in the app are calculated on the selected synthetic scenario only.
