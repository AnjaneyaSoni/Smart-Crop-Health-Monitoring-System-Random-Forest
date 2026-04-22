# 🌾 CropPulse — Real-Time Crop Health Monitor

A **Streamlit** dashboard for real-time crop health monitoring using simulated IoT sensor data, ML-powered predictions, and smart alerting.

## Overview

CropPulse continuously tracks six agricultural sensors — **Humidity, Temperature, Soil Moisture, Light, Leaf Wetness, and Leaf Color** — and computes a weighted **Health Score** (0–100%) for each configured crop based on how far readings deviate from optimal ranges.

## How It Works

1. **Data Generation** — In demo mode, realistic sensor readings are synthesized using diurnal cycles (day/night temperature swings, bell-curve light, morning dew), slow drift, and correlated noise. Three stress profiles are available: 🟢 Mild, 🟡 Moderate, 🔴 Severe.

2. **Health Scoring** — Each sensor value is compared against the crop's optimal range. A weighted average across all sensors produces the overall health score:
   - 🟢 ≥80% Excellent · 🟡 60–80% Good · 🟠 40–60% Fair · 🔴 <40% Critical

3. **Alerting** — When a sensor leaves its optimal range, an alert is generated with severity (`warning` / `critical`) based on the deviation magnitude. Alerts are deduplicated per sensor.

4. **Prediction** — The app forecasts sensor values **6 hours ahead** using either:
   - A trained **Random Forest** model (`crop_rf_model.pkl`) that takes 30 statistical features (mean, std, min, max, slope × 6 sensors) from the last 100 readings, or
   - A **mock trend-based** fallback (linear extrapolation) when no model file is present.

5. **Live Dashboard** — Uses Streamlit's `@st.fragment` with auto-refresh to update only the dashboard area (gauges, radar, timelines, prediction charts, alert log) without flickering the sidebar.

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit application |
| `train_model.py` | Train the Random Forest prediction model |
| `crop_rf_model.pkl` | Serialized RF model (optional) |
| `crop_health_data/` | Persistent storage (sensor CSV, alerts, plant configs) |

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Supported Crops

Corn, Wheat, Rice, and Tomato — each with pre-configured optimal sensor ranges and weights. Custom crops can be added from the sidebar.
