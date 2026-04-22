import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

#   ML imports (graceful fallback if sklearn not installed)  
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    import pickle
    RF_AVAILABLE = True
except ImportError:
    RF_AVAILABLE = False

MODEL_FILE  = Path("crop_rf_model.pkl")
SENSOR_COLS = ["humidity", "temperature", "soil_moisture", "light", "leaf_wetness", "leaf_color"]
WINDOW      = 100   # readings used to compute features
HORIZON     = 6     # hours ahead to predict

 
# FEATURE ENGINEERING
 

def compute_features(df_window: pd.DataFrame) -> np.ndarray:
    feats = []
    x = np.arange(len(df_window))
    for s in SENSOR_COLS:
        vals = df_window[s].values.astype(float)
        feats.append(vals.mean())
        feats.append(vals.std() + 1e-9)
        feats.append(vals.min())
        feats.append(vals.max())
        slope = np.polyfit(x, vals, 1)[0] if len(vals) > 1 else 0.0
        feats.append(slope)
    return np.array(feats, dtype=float)

def generate_training_csv(df: pd.DataFrame, horizon: int = HORIZON,
                           readings_per_hour: int = 1) -> pd.DataFrame:
    step    = horizon * readings_per_hour
    records = []
    total   = len(df)
    for i in range(total - WINDOW - step + 1):
        window = df.iloc[i : i + WINDOW]
        future = df.iloc[i + WINDOW : i + WINDOW + step]
        feats  = compute_features(window)
        labels = [future[s].mean() for s in SENSOR_COLS]
        feat_names = [f"{s}_{stat}" for s in SENSOR_COLS for stat in ["mean","std","min","max","slope"]]
        row = dict(zip(feat_names, feats))
        for s, lbl in zip(SENSOR_COLS, labels):
            row[f"target_{s}"] = lbl
        records.append(row)
    return pd.DataFrame(records)

def load_model():
    if MODEL_FILE.exists() and RF_AVAILABLE:
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    return None

def predict_future(df: pd.DataFrame, model):
    if len(df) < WINDOW:
        return None
    window = df[SENSOR_COLS].tail(WINDOW)
    feats  = compute_features(window).reshape(1, -1)
    preds  = model.predict(feats)[0]
    return {s: float(np.clip(preds[i], SENSORS[s]["min"], SENSORS[s]["max"]))
            for i, s in enumerate(SENSOR_COLS)}

def mock_predict_future(df: pd.DataFrame) -> dict:
    result = {}
    window = df[SENSOR_COLS].tail(WINDOW)
    x      = np.arange(len(window))
    for s in SENSOR_COLS:
        vals  = window[s].values.astype(float)
        slope = np.polyfit(x, vals, 1)[0] if len(vals) > 1 else 0
        pred  = vals[-1] + slope * HORIZON
        result[s] = float(np.clip(pred, SENSORS[s]["min"], SENSORS[s]["max"]))
    return result

st.set_page_config(
    page_title="CropPulse — Crop Health Monitor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

 
# DATA STORAGE
 

DATA_DIR         = Path("crop_health_data")
DATA_DIR.mkdir(exist_ok=True)
PLANTS_FILE      = DATA_DIR / "plants.json"
SENSOR_DATA_FILE = DATA_DIR / "sensor_data.csv"
ALERTS_FILE      = DATA_DIR / "alerts.json"

 
# SENSORS CONFIG
 

SENSORS = {
    "humidity":      {"name": "Humidity",      "unit": "%",   "icon": "💧", "color": "#3b82f6", "min": 0,  "max": 100},
    "temperature":   {"name": "Temperature",   "unit": "°C",  "icon": "🌡️", "color": "#ef4444", "min": 0,  "max": 50},
    "soil_moisture": {"name": "Soil Moisture", "unit": "%",   "icon": "🌱", "color": "#8b5cf6", "min": 0,  "max": 100},
    "light":         {"name": "Light",         "unit": "lux", "icon": "☀️", "color": "#f59e0b", "min": 0,  "max": 2000},
    "leaf_wetness":  {"name": "Leaf Wetness",  "unit": "%",   "icon": "💦", "color": "#06b6d4", "min": 0,  "max": 100},
    "leaf_color":    {"name": "Leaf Color",    "unit": "",    "icon": "🍃", "color": "#10b981", "min": 0,  "max": 100},
}

DEFAULT_PLANTS = {
    "Corn": {
        "humidity":      {"optimal_min": 60, "optimal_max": 80,   "weight": 0.20},
        "temperature":   {"optimal_min": 20, "optimal_max": 30,   "weight": 0.25},
        "soil_moisture": {"optimal_min": 40, "optimal_max": 70,   "weight": 0.20},
        "light":         {"optimal_min": 800,"optimal_max": 1500, "weight": 0.15},
        "leaf_wetness":  {"optimal_min": 30, "optimal_max": 60,   "weight": 0.10},
        "leaf_color":    {"optimal_min": 70, "optimal_max": 100,  "weight": 0.10},
    },
    "Wheat": {
        "humidity":      {"optimal_min": 55, "optimal_max": 75,   "weight": 0.20},
        "temperature":   {"optimal_min": 15, "optimal_max": 25,   "weight": 0.25},
        "soil_moisture": {"optimal_min": 35, "optimal_max": 65,   "weight": 0.20},
        "light":         {"optimal_min": 700,"optimal_max": 1300, "weight": 0.15},
        "leaf_wetness":  {"optimal_min": 25, "optimal_max": 55,   "weight": 0.10},
        "leaf_color":    {"optimal_min": 65, "optimal_max": 95,   "weight": 0.10},
    },
    "Rice": {
        "humidity":      {"optimal_min": 70, "optimal_max": 90,   "weight": 0.20},
        "temperature":   {"optimal_min": 22, "optimal_max": 32,   "weight": 0.20},
        "soil_moisture": {"optimal_min": 50, "optimal_max": 80,   "weight": 0.25},
        "light":         {"optimal_min": 600,"optimal_max": 1200, "weight": 0.15},
        "leaf_wetness":  {"optimal_min": 40, "optimal_max": 70,   "weight": 0.10},
        "leaf_color":    {"optimal_min": 65, "optimal_max": 95,   "weight": 0.10},
    },
    "Tomato": {
        "humidity":      {"optimal_min": 65, "optimal_max": 85,   "weight": 0.20},
        "temperature":   {"optimal_min": 18, "optimal_max": 28,   "weight": 0.25},
        "soil_moisture": {"optimal_min": 45, "optimal_max": 75,   "weight": 0.20},
        "light":         {"optimal_min":1000,"optimal_max": 1800, "weight": 0.15},
        "leaf_wetness":  {"optimal_min": 35, "optimal_max": 65,   "weight": 0.10},
        "leaf_color":    {"optimal_min": 70, "optimal_max": 100,  "weight": 0.10},
    },
}

 
# REALISTIC DEMO PROFILES
# Each sensor follows natural patterns:
#   - temperature: sinusoidal diurnal cycle (cool nights, warm days)
#   - humidity: inverse of temperature
#   - light: bell curve (zero at night, peak midday)
#   - soil_moisture: slow drain curve (decreases over time unless reset)
#   - leaf_wetness: high at night/morning (dew), drops midday
#   - leaf_color: slow seasonal drift, mostly stable
 

DEMO_PROFILES = {
    "Mild": {
        "label": "🟢 Mild",
        "description": "Readings mostly within optimal ranges",
        # Base mid-values (around which patterns oscillate)
        "temperature":   {"base": 25,   "amplitude": 6,    "noise": 0.8,  "drift":  0.00},
        "humidity":      {"base": 70,   "amplitude": 12,   "noise": 1.5,  "drift":  0.00},
        "soil_moisture": {"base": 60,   "amplitude": 3,    "noise": 1.0,  "drift": -0.03},
        "light":         {"base": 1100, "amplitude": 950,  "noise": 40,   "drift":  0.00},
        "leaf_wetness":  {"base": 45,   "amplitude": 25,   "noise": 2.0,  "drift":  0.00},
        "leaf_color":    {"base": 82,   "amplitude": 4,    "noise": 1.0,  "drift": -0.01},
        "noise_factor":  0.5,
    },
    "Moderate": {
        "label": "🟡 Moderate",
        "description": "1–2 sensors drifting, health ~55–65%",
        "temperature":   {"base": 33,   "amplitude": 8,    "noise": 2.0,  "drift":  0.05},
        "humidity":      {"base": 52,   "amplitude": 14,   "noise": 3.0,  "drift": -0.04},
        "soil_moisture": {"base": 35,   "amplitude": 4,    "noise": 2.0,  "drift": -0.08},
        "light":         {"base": 700,  "amplitude": 600,  "noise": 60,   "drift": -0.03},
        "leaf_wetness":  {"base": 65,   "amplitude": 20,   "noise": 3.0,  "drift":  0.04},
        "leaf_color":    {"base": 62,   "amplitude": 6,    "noise": 2.5,  "drift": -0.04},
        "noise_factor":  1.2,
    },
    "Severe": {
        "label": "🔴 Severe",
        "description": "Multiple sensors critical, health <40%",
        "temperature":   {"base": 39,   "amplitude": 8,    "noise": 3.0,  "drift":  0.10},
        "humidity":      {"base": 32,   "amplitude": 16,   "noise": 5.0,  "drift": -0.10},
        "soil_moisture": {"base": 18,   "amplitude": 4,    "noise": 2.5,  "drift": -0.12},
        "light":         {"base": 350,  "amplitude": 300,  "noise": 80,   "drift": -0.06},
        "leaf_wetness":  {"base": 82,   "amplitude": 12,   "noise": 4.0,  "drift":  0.08},
        "leaf_color":    {"base": 30,   "amplitude": 6,    "noise": 3.5,  "drift": -0.10},
        "noise_factor":  2.5,
    },
}

def _diurnal_factor(hour_of_day: float) -> float:
    """Returns a -1 to +1 sinusoidal value peaking at 14:00 (2 PM)."""
    return np.sin(2 * np.pi * (hour_of_day - 8) / 24)

def _light_factor(hour_of_day: float) -> float:
    """Bell-shaped light: 0 at night, peak ~12:00."""
    x = (hour_of_day - 12) / 6.0
    val = np.exp(-0.5 * x * x)
    return val if 6 <= hour_of_day <= 20 else 0.0

def _dew_factor(hour_of_day: float) -> float:
    """Leaf wetness: high around 4–8 AM (dew), low midday."""
    x = (hour_of_day - 6) / 4.0
    return np.exp(-0.5 * x * x)

def generate_sample_data(profile_name="Mild", periods=200):
    """Generate realistic sensor data with diurnal cycles, slow drift, and correlated sensors."""
    profile = DEMO_PROFILES[profile_name]
    nf      = profile["noise_factor"]
    dates   = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq='h')
    data    = {'timestamp': dates}

    hours    = np.array([d.hour + d.minute / 60.0 for d in dates])
    t_index  = np.arange(periods)

    # Temperature: diurnal cycle + drift + noise
    p_temp  = profile["temperature"]
    drift_t = np.linspace(0, p_temp["drift"] * periods, periods)
    temp    = (p_temp["base"]
               + p_temp["amplitude"] * _diurnal_factor(hours)
               + drift_t
               + np.random.normal(0, p_temp["noise"] * nf, periods))
    temp    = np.clip(temp, SENSORS["temperature"]["min"], SENSORS["temperature"]["max"])

    # Humidity: roughly inverse of temperature + own drift + noise
    p_hum   = profile["humidity"]
    drift_h = np.linspace(0, p_hum["drift"] * periods, periods)
    # Normalize temp to [0,1] and invert for humidity correlation
    temp_norm = (temp - temp.min()) / (temp.ptp() + 1e-9)
    hum     = (p_hum["base"]
               - p_hum["amplitude"] * 0.5 * temp_norm
               + p_hum["amplitude"] * 0.5 * (1 - temp_norm)
               + drift_h
               + np.random.normal(0, p_hum["noise"] * nf, periods))
    hum     = np.clip(hum, SENSORS["humidity"]["min"], SENSORS["humidity"]["max"])

    # Soil moisture: slow downward drift (drying), small diurnal ripple + noise
    p_sm    = profile["soil_moisture"]
    drift_s = np.linspace(0, p_sm["drift"] * periods * 0.4, periods)
    sm      = (p_sm["base"]
               + p_sm["amplitude"] * np.sin(2 * np.pi * t_index / 48)  # slow oscillation
               + drift_s
               + np.random.normal(0, p_sm["noise"] * nf, periods))
    sm      = np.clip(sm, SENSORS["soil_moisture"]["min"], SENSORS["soil_moisture"]["max"])

    # Light: bell curve each day + noise
    p_lux   = profile["light"]
    drift_l = np.linspace(0, p_lux["drift"] * periods * 20, periods)
    lux_raw = np.array([_light_factor(h) for h in hours])
    lux     = (p_lux["base"] * lux_raw
               + drift_l
               + np.random.normal(0, p_lux["noise"] * nf, periods))
    lux     = np.clip(lux, SENSORS["light"]["min"], SENSORS["light"]["max"])

    # Leaf wetness: dew in morning, dry midday, somewhat tracks humidity
    p_lw    = profile["leaf_wetness"]
    drift_w = np.linspace(0, p_lw["drift"] * periods * 0.3, periods)
    dew_raw = np.array([_dew_factor(h) for h in hours])
    lw      = (p_lw["base"]
               + p_lw["amplitude"] * dew_raw
               - p_lw["amplitude"] * 0.3 * lux_raw  # dries in sunlight
               + drift_w
               + np.random.normal(0, p_lw["noise"] * nf, periods))
    lw      = np.clip(lw, SENSORS["leaf_wetness"]["min"], SENSORS["leaf_wetness"]["max"])

    # Leaf color: slow seasonal drift, very low noise (color changes slowly)
    p_lc    = profile["leaf_color"]
    drift_c = np.linspace(0, p_lc["drift"] * periods * 0.5, periods)
    lc      = (p_lc["base"]
               + p_lc["amplitude"] * np.sin(2 * np.pi * t_index / 168)  # weekly oscillation
               + drift_c
               + np.random.normal(0, p_lc["noise"] * nf, periods))
    lc      = np.clip(lc, SENSORS["leaf_color"]["min"], SENSORS["leaf_color"]["max"])

    data["humidity"]      = hum
    data["temperature"]   = temp
    data["soil_moisture"] = sm
    data["light"]         = lux
    data["leaf_wetness"]  = lw
    data["leaf_color"]    = lc

    return pd.DataFrame(data)

def generate_new_reading(profile_name: str, last_row: dict, step: int) -> dict:
    """Generate a single new reading that continues from the last known values
    with realistic small step changes (not fully random)."""
    profile  = DEMO_PROFILES[profile_name]
    nf       = profile["noise_factor"]
    now      = datetime.datetime.now()
    hour     = now.hour + now.minute / 60.0
    reading  = {"timestamp": now}

    prev_temp = last_row.get("temperature", profile["temperature"]["base"])
    target_temp = (profile["temperature"]["base"]
                   + profile["temperature"]["amplitude"] * _diurnal_factor(hour)
                   + np.random.normal(0, profile["temperature"]["noise"] * nf))
    # Smoothly move toward target (EMA-like)
    new_temp = prev_temp + 0.15 * (target_temp - prev_temp)
    new_temp = float(np.clip(new_temp, SENSORS["temperature"]["min"], SENSORS["temperature"]["max"]))
    reading["temperature"] = new_temp

    prev_hum = last_row.get("humidity", profile["humidity"]["base"])
    temp_norm = (new_temp - 5) / 45.0
    target_hum = (profile["humidity"]["base"]
                  - profile["humidity"]["amplitude"] * 0.4 * temp_norm
                  + profile["humidity"]["amplitude"] * 0.4 * (1 - temp_norm)
                  + np.random.normal(0, profile["humidity"]["noise"] * nf))
    new_hum = prev_hum + 0.12 * (target_hum - prev_hum)
    reading["humidity"] = float(np.clip(new_hum, SENSORS["humidity"]["min"], SENSORS["humidity"]["max"]))

    prev_sm = last_row.get("soil_moisture", profile["soil_moisture"]["base"])
    drift_per_step = profile["soil_moisture"]["drift"] * 0.4
    target_sm = (prev_sm + drift_per_step
                 + np.random.normal(0, profile["soil_moisture"]["noise"] * nf * 0.3))
    new_sm = prev_sm + 0.20 * (target_sm - prev_sm)
    reading["soil_moisture"] = float(np.clip(new_sm, SENSORS["soil_moisture"]["min"], SENSORS["soil_moisture"]["max"]))

    lux_factor = _light_factor(hour)
    target_lux = (profile["light"]["base"] * lux_factor
                  + np.random.normal(0, profile["light"]["noise"] * nf))
    prev_lux = last_row.get("light", profile["light"]["base"] * lux_factor)
    new_lux = prev_lux + 0.25 * (target_lux - prev_lux)
    reading["light"] = float(np.clip(new_lux, SENSORS["light"]["min"], SENSORS["light"]["max"]))

    dew_factor = _dew_factor(hour)
    target_lw = (profile["leaf_wetness"]["base"]
                 + profile["leaf_wetness"]["amplitude"] * dew_factor
                 - profile["leaf_wetness"]["amplitude"] * 0.3 * lux_factor
                 + np.random.normal(0, profile["leaf_wetness"]["noise"] * nf))
    prev_lw = last_row.get("leaf_wetness", profile["leaf_wetness"]["base"])
    new_lw = prev_lw + 0.18 * (target_lw - prev_lw)
    reading["leaf_wetness"] = float(np.clip(new_lw, SENSORS["leaf_wetness"]["min"], SENSORS["leaf_wetness"]["max"]))

    drift_per_step_lc = profile["leaf_color"]["drift"] * 0.5
    prev_lc = last_row.get("leaf_color", profile["leaf_color"]["base"])
    target_lc = (prev_lc + drift_per_step_lc
                 + np.random.normal(0, profile["leaf_color"]["noise"] * nf * 0.5))
    new_lc = prev_lc + 0.08 * (target_lc - prev_lc)
    reading["leaf_color"] = float(np.clip(new_lc, SENSORS["leaf_color"]["min"], SENSORS["leaf_color"]["max"]))

    return reading

 
# DATA FUNCTIONS
 

def load_plants():
    if PLANTS_FILE.exists():
        with open(PLANTS_FILE, 'r') as f:
            return json.load(f)
    return DEFAULT_PLANTS.copy()

def save_plants(plants):
    with open(PLANTS_FILE, 'w') as f:
        json.dump(plants, f, indent=2)

def load_sensor_data():
    if SENSOR_DATA_FILE.exists():
        try:
            df = pd.read_csv(SENSOR_DATA_FILE)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except:
            return None
    return None

def save_sensor_data(df):
    df.to_csv(SENSOR_DATA_FILE, index=False)

def load_alerts():
    if ALERTS_FILE.exists():
        with open(ALERTS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_alerts(alerts):
    with open(ALERTS_FILE, 'w') as f:
        json.dump(alerts, f, indent=2)

def calculate_health_score(row, plant_config):
    total_weight, weighted_score = 0, 0
    for sensor in SENSORS.keys():
        value   = row[sensor]
        opt_min = plant_config[sensor]["optimal_min"]
        opt_max = plant_config[sensor]["optimal_max"]
        weight  = plant_config[sensor]["weight"]
        max_val = SENSORS[sensor]["max"]
        if opt_min <= value <= opt_max:
            sensor_score = 100
        elif value < opt_min:
            deviation    = (opt_min - value) / opt_min
            sensor_score = max(0, 100 * (1 - deviation ** 1.8))
        else:
            deviation    = (value - opt_max) / (max_val - opt_max + 1e-9)
            sensor_score = max(0, 100 * (1 - deviation ** 1.4))
        weighted_score += sensor_score * weight
        total_weight   += weight
    return weighted_score / total_weight if total_weight > 0 else 50

def generate_alert(sensor, value, opt_min, opt_max, plant_name):
    now    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    config = SENSORS[sensor]
    if value < opt_min:
        gap = opt_min - value
        pct = (gap / opt_min) * 100 if opt_min > 0 else 0
        if pct > 30:
            severity = "critical"
            msg = f"{config['icon']} {config['name']} critically LOW ({value:.1f}{config['unit']}) — {pct:.0f}% below optimal for {plant_name}"
        else:
            severity = "warning"
            msg = f"{config['icon']} {config['name']} below optimal ({value:.1f}{config['unit']}) for {plant_name}"
    else:
        gap = value - opt_max
        pct = (gap / opt_max) * 100 if opt_max > 0 else 0
        if pct > 30:
            severity = "critical"
            msg = f"{config['icon']} {config['name']} critically HIGH ({value:.1f}{config['unit']}) — {pct:.0f}% above optimal for {plant_name}"
        else:
            severity = "warning"
            msg = f"{config['icon']} {config['name']} above optimal ({value:.1f}{config['unit']}) for {plant_name}"
    return {"time": now, "sensor": sensor, "value": value,
            "severity": severity, "message": msg, "plant": plant_name}

def check_and_generate_alerts(row, plant_config, plant_name, existing_alerts):
    new_alerts = []
    for sensor in SENSORS.keys():
        value   = row[sensor]
        opt_min = plant_config[sensor]["optimal_min"]
        opt_max = plant_config[sensor]["optimal_max"]
        if not (opt_min <= value <= opt_max):
            recent = [a for a in existing_alerts[-20:] if a["sensor"] == sensor]
            if not recent:
                new_alerts.append(generate_alert(sensor, value, opt_min, opt_max, plant_name))
    return new_alerts

 
# SESSION STATE INIT
 

def init_session():
    if 'plants' not in st.session_state:
        st.session_state.plants = load_plants()
    if 'sensor_data' not in st.session_state:
        data = load_sensor_data()
        if data is None:
            data = generate_sample_data("Mild")
        st.session_state.sensor_data = data
    if 'selected_plant' not in st.session_state:
        st.session_state.selected_plant = "Corn"
    if 'demo_mode' not in st.session_state:
        st.session_state.demo_mode = True
    if 'demo_profile' not in st.session_state:
        st.session_state.demo_profile = "Mild"
    if 'refresh_rate' not in st.session_state:
        st.session_state.refresh_rate = 5
    if 'alerts' not in st.session_state:
        st.session_state.alerts = load_alerts()
    if 'readings_per_hour' not in st.session_state:
        st.session_state.readings_per_hour = 1
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.datetime.now()
    if 'step_counter' not in st.session_state:
        st.session_state.step_counter = 0

init_session()

def update_all_health_scores():
    df = st.session_state.sensor_data.copy()
    for plant_name, plant_config in st.session_state.plants.items():
        df[f'health_{plant_name}'] = df.apply(
            lambda row: calculate_health_score(row, plant_config), axis=1
        )
    st.session_state.sensor_data = df

update_all_health_scores()

 
# CUSTOM CSS
 

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
* { font-family: 'DM Sans', sans-serif; }
code, .mono { font-family: 'DM Mono', monospace; }
.stApp { background-color: #f0f4f8; }
[data-testid="stSidebar"] {
    background: linear-gradient(170deg, #0f1923 0%, #162032 60%, #1a2740 100%);
    border-right: 1px solid #253347;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }
[data-testid="stSidebar"] .stSelectbox > div > div { background: #1e2d40 !important; border-color: #2d4257 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: #1e3a5f !important; border: 1px solid #2e5280 !important;
    color: #93c5fd !important; border-radius: 8px; font-weight: 500; transition: all 0.2s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #2563eb !important; border-color: #3b82f6 !important; color: #ffffff !important;
}
[data-testid="stSidebar"] .stToggle span { background: #2563eb !important; }
[data-testid="stSidebar"] .stSlider > div > div > div { background: #2563eb !important; }
[data-testid="stSidebar"] hr { border-color: #253347 !important; }
[data-testid="stSidebar"] .stTabs [data-baseweb="tab-list"] {
    background: #1a2740 !important; border-radius: 10px; gap: 4px; padding: 4px;
}
[data-testid="stSidebar"] .stTabs [data-baseweb="tab"] {
    color: #94a3b8 !important; background: transparent !important;
    border-radius: 7px; padding: 6px 12px; font-size: 12px;
}
[data-testid="stSidebar"] .stTabs [aria-selected="true"] {
    background: #2563eb !important; color: #ffffff !important;
}
.main-header {
    background: linear-gradient(135deg, #0f1923 0%, #1e3a5f 50%, #1a4a7a 100%);
    border-radius: 20px; padding: 28px 36px; margin-bottom: 24px;
    border: 1px solid #2d5280; display: flex;
    justify-content: space-between; align-items: center;
}
.main-title { color: #ffffff; font-size: 26px; font-weight: 700; letter-spacing: -0.5px; margin: 0; }
.main-subtitle { color: #93c5fd; font-size: 13px; margin-top: 4px; font-weight: 400; }
.header-badge {
    background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.4);
    color: #34d399; border-radius: 24px; padding: 6px 16px;
    font-size: 12px; font-weight: 600; letter-spacing: 0.5px;
}
.header-time { color: #64748b; font-size: 12px; font-family: 'DM Mono', monospace; margin-top: 6px; }
.metric-card {
    background: #ffffff; border-radius: 16px; padding: 22px 24px;
    border: 1px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s, transform 0.2s; height: 100%;
}
.metric-card:hover { box-shadow: 0 6px 20px rgba(0,0,0,0.08); transform: translateY(-2px); }
.metric-label { color: #64748b; font-size: 12px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 10px; }
.metric-value { color: #0f172a; font-size: 34px; font-weight: 700; line-height: 1; }
.metric-unit  { color: #94a3b8; font-size: 15px; margin-left: 3px; }
.metric-sub   { color: #94a3b8; font-size: 12px; margin-top: 8px; }
.metric-trend-up   { color: #10b981; font-size: 12px; font-weight: 600; }
.metric-trend-down { color: #ef4444; font-size: 12px; font-weight: 600; }
.section-header {
    color: #0f172a; font-size: 16px; font-weight: 600;
    margin: 28px 0 16px 0; padding-left: 14px;
    border-left: 3px solid #2563eb; letter-spacing: -0.2px;
}
.chart-card {
    background: #ffffff; border-radius: 16px; padding: 8px 4px 4px 4px;
    border: 1px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.04); margin-bottom: 4px;
}
/* Style Plotly chart containers directly — avoids wrapper divs that cause flicker */
[data-testid="stPlotlyChart"] {
    background: #ffffff; border-radius: 16px; padding: 4px 2px 2px 2px;
    border: 1px solid #e2e8f0; box-shadow: 0 2px 8px rgba(0,0,0,0.04); margin-bottom: 4px;
}
.demo-mild     { background: #d1fae5; color: #065f46; border-radius: 8px; padding: 4px 10px; font-size: 12px; font-weight: 600; }
.demo-moderate { background: #fef3c7; color: #92400e; border-radius: 8px; padding: 4px 10px; font-size: 12px; font-weight: 600; }
.demo-severe   { background: #fee2e2; color: #991b1b; border-radius: 8px; padding: 4px 10px; font-size: 12px; font-weight: 600; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

 
# SIDEBAR
 

with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 20px 0;'>
        <div style='font-size: 20px; font-weight: 700; color: #f1f5f9; letter-spacing: -0.5px;'>🌾 CropPulse</div>
        <div style='font-size: 11px; color: #64748b; margin-top: 3px;'>Precision Crop Health Monitor</div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["⚙️ General", "🌾 Plants", "📁 Data"])

    with tab1:
        st.markdown("<p style='font-size:13px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:0.8px;'>Monitoring</p>", unsafe_allow_html=True)

        demo_toggle = st.toggle("Live Demo Mode", value=st.session_state.demo_mode)
        if demo_toggle != st.session_state.demo_mode:
            st.session_state.demo_mode = demo_toggle
            st.rerun()

        if st.session_state.demo_mode:
            profile_options = list(DEMO_PROFILES.keys())
            profile_labels  = [DEMO_PROFILES[p]["label"] for p in profile_options]
            selected_idx    = profile_options.index(st.session_state.demo_profile)
            chosen_label    = st.radio(
                "Stress Level", options=profile_labels, index=selected_idx,
                help="Mild: near-optimal | Moderate: some drift | Severe: critical conditions"
            )
            chosen_profile = profile_options[profile_labels.index(chosen_label)]
            if chosen_profile != st.session_state.demo_profile:
                st.session_state.demo_profile = chosen_profile
                st.session_state.sensor_data  = generate_sample_data(chosen_profile)
                st.session_state.alerts       = []
                st.session_state.step_counter = 0
                update_all_health_scores()
                st.rerun()
            desc = DEMO_PROFILES[st.session_state.demo_profile]["description"]
            st.markdown(f"<div style='font-size:11px;color:#64748b;margin-top:4px;'>{desc}</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("<p style='font-size:13px;font-weight:600;color:#94a3b8;text-transform:uppercase;letter-spacing:0.8px;'>Crop Selection</p>", unsafe_allow_html=True)
        selected_plant = st.selectbox(
            "Active Crop",
            list(st.session_state.plants.keys()),
            index=list(st.session_state.plants.keys()).index(st.session_state.selected_plant)
        )
        st.session_state.selected_plant = selected_plant

        st.markdown("---")
        new_refresh = st.slider("Refresh Rate (sec)", 1, 30, st.session_state.refresh_rate)
        if new_refresh != st.session_state.refresh_rate:
            st.session_state.refresh_rate = new_refresh

        st.markdown("---")
        st.markdown("""
        <p style='font-size:11px;color:#64748b;line-height:1.6;'>
        <b style='color:#93c5fd;'>Health Score</b> = weighted average of all sensor deviations from optimal range.<br>
        🟢 ≥80% Excellent &nbsp;🟡 60–80% Good &nbsp;🔴 <60% Needs attention
        </p>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<p style='font-size:12px;color:#94a3b8;'>Add or configure plant types</p>", unsafe_allow_html=True)
        new_plant = st.text_input("New Plant Name", placeholder="e.g., Soybean")
        if st.button("➕ Add Plant", width='stretch'):
            if new_plant and new_plant not in st.session_state.plants:
                st.session_state.plants[new_plant] = {
                    s: {"optimal_min": 50, "optimal_max": 80, "weight": round(1/6, 4)}
                    for s in SENSORS.keys()
                }
                save_plants(st.session_state.plants)
                update_all_health_scores()
                st.rerun()

        plant_to_edit = st.selectbox("Edit Plant Config", list(st.session_state.plants.keys()))
        if plant_to_edit:
            st.markdown(f"<small style='color:#93c5fd;font-weight:600;'>Editing: {plant_to_edit}</small>", unsafe_allow_html=True)
            for sensor, config in SENSORS.items():
                st.markdown(f"<small style='color:#94a3b8;'>{config['icon']} {config['name']}</small>", unsafe_allow_html=True)
                c1, c2, c3 = st.columns(3)
                cur = st.session_state.plants[plant_to_edit][sensor]
                with c1:
                    opt_min = st.number_input("Min", value=float(cur["optimal_min"]), key=f"{plant_to_edit}_{sensor}_min", step=1.0, label_visibility="collapsed")
                with c2:
                    opt_max = st.number_input("Max", value=float(cur["optimal_max"]), key=f"{plant_to_edit}_{sensor}_max", step=1.0, label_visibility="collapsed")
                with c3:
                    weight  = st.number_input("Wt",  value=float(cur["weight"]),      key=f"{plant_to_edit}_{sensor}_wt",  step=0.01, format="%.2f", label_visibility="collapsed")
                st.session_state.plants[plant_to_edit][sensor] = {"optimal_min": opt_min, "optimal_max": opt_max, "weight": weight}
            if st.button("💾 Save Config", width='stretch'):
                save_plants(st.session_state.plants)
                update_all_health_scores()
                st.success("Saved!")
                st.rerun()

    with tab3:
        st.markdown("<p style='font-size:12px;color:#94a3b8;'>Import / Export sensor data</p>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Import CSV", type=['csv'])
        if uploaded:
            df_up = pd.read_csv(uploaded)
            if 'timestamp' in df_up.columns:
                df_up['timestamp'] = pd.to_datetime(df_up['timestamp'])
                for plant in st.session_state.plants.keys():
                    df_up[f'health_{plant}'] = df_up.apply(
                        lambda r: calculate_health_score(r, st.session_state.plants[plant]), axis=1)
                st.session_state.sensor_data = pd.concat([st.session_state.sensor_data, df_up]).tail(2000)
                save_sensor_data(st.session_state.sensor_data)
                st.success(f"Imported {len(df_up)} records")
                st.rerun()

        st.markdown("---")
        st.markdown("<p style='font-size:11px;color:#94a3b8;font-weight:600;'>EXPORT</p>", unsafe_allow_html=True)
        if st.button("📥 Export Raw CSV", width='stretch'):
            csv = st.session_state.sensor_data.to_csv(index=False)
            st.download_button("Download Raw CSV", csv, "sensor_data.csv", "text/csv")

        st.markdown("---")
        st.markdown("<p style='font-size:11px;color:#94a3b8;font-weight:600;'>🤖 RF TRAINING DATA</p>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:10px;color:#64748b;'>Generates windowed feature CSV (30 features + 6 targets) ready for Random Forest training.</p>", unsafe_allow_html=True)

        rph = st.slider("Readings per hour", min_value=1, max_value=12,
                        value=st.session_state.readings_per_hour,
                        help="How many sensor readings per hour. Affects how many rows = 6 hrs ahead.")
        st.session_state.readings_per_hour = rph

        n_rows = len(st.session_state.sensor_data)
        needed = WINDOW + HORIZON * rph
        if n_rows >= needed:
            st.markdown(f"<p style='font-size:10px;color:#10b981;'>✅ {n_rows} rows available → ~{n_rows - needed + 1} training samples</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p style='font-size:10px;color:#ef4444;'>⚠️ Need ≥{needed} rows, have {n_rows}. Collect more data first.</p>", unsafe_allow_html=True)

        if st.button("🧠 Generate Training CSV", width='stretch'):
            if n_rows >= needed:
                with st.spinner("Building training dataset..."):
                    train_df = generate_training_csv(
                        st.session_state.sensor_data[SENSOR_COLS].reset_index(drop=True),
                        horizon=HORIZON, readings_per_hour=rph
                    )
                csv_train = train_df.to_csv(index=False)
                st.download_button("⬇️ Download Training CSV", csv_train,
                                   "rf_training_data.csv", "text/csv", width='stretch')
                st.success(f"Generated {len(train_df)} training samples · 30 features + 6 targets")
            else:
                st.error(f"Not enough data. Need {needed} rows minimum.")

        st.markdown("---")
        if st.button("🗑️ Reset Data", width='stretch'):
            st.session_state.sensor_data  = generate_sample_data(st.session_state.demo_profile)
            st.session_state.alerts       = []
            st.session_state.step_counter = 0
            update_all_health_scores()
            save_sensor_data(st.session_state.sensor_data)
            st.rerun()
        if st.button("🔕 Clear Alerts", width='stretch'):
            st.session_state.alerts = []
            save_alerts([])
            st.rerun()

 
# CHART STYLE CONSTANTS
 

CHART_BG    = "#ffffff"
CHART_PAPER = "#ffffff"
CHART_FONT  = dict(family="DM Sans", color="#1e293b", size=12)
GRID_COLOR  = "#e2e8f0"

 
# LIVE DASHBOARD FRAGMENT — partial rerun via @st.fragment
# This eliminates full-page flicker: only this function
# re-executes on the timer, the sidebar and CSS stay intact.
 

@st.fragment(run_every=datetime.timedelta(seconds=st.session_state.refresh_rate))
def render_live_dashboard():
    #   Demo mode tick: add one new reading  
    if st.session_state.demo_mode:
        current_time = datetime.datetime.now()
        time_diff    = (current_time - st.session_state.last_update).total_seconds()

        if time_diff >= st.session_state.refresh_rate:
            last_row  = st.session_state.sensor_data.iloc[-1].to_dict()
            step      = st.session_state.step_counter
            new_reading = generate_new_reading(st.session_state.demo_profile, last_row, step)

            for plant, config in st.session_state.plants.items():
                new_reading[f'health_{plant}'] = calculate_health_score(new_reading, config)

            new_alerts = check_and_generate_alerts(
                new_reading,
                st.session_state.plants[st.session_state.selected_plant],
                st.session_state.selected_plant,
                st.session_state.alerts,
            )
            if new_alerts:
                st.session_state.alerts = (st.session_state.alerts + new_alerts)[-100:]
                save_alerts(st.session_state.alerts)

            new_df = pd.DataFrame([new_reading])
            st.session_state.sensor_data = pd.concat(
                [st.session_state.sensor_data, new_df], ignore_index=True
            ).tail(500)
            save_sensor_data(st.session_state.sensor_data)
            st.session_state.last_update  = current_time
            st.session_state.step_counter = step + 1

     
    # MAIN DASHBOARD — derive latest values
     

    latest    = st.session_state.sensor_data.iloc[-1]
    df_all    = st.session_state.sensor_data
    df_recent = df_all.tail(100)
    plant_cfg = st.session_state.plants[st.session_state.selected_plant]
    health    = latest[f'health_{st.session_state.selected_plant}']

    if health >= 80:
        health_color, health_label, health_badge = "#10b981", "Excellent", "🟢 Excellent"
    elif health >= 60:
        health_color, health_label, health_badge = "#f59e0b", "Good",      "🟡 Good"
    elif health >= 40:
        health_color, health_label, health_badge = "#f97316", "Fair",      "🟠 Fair"
    else:
        health_color, health_label, health_badge = "#ef4444", "Critical",  "🔴 Critical"

    profile_name = st.session_state.demo_profile if st.session_state.demo_mode else "Off"
    profile_css  = {"Mild":"demo-mild","Moderate":"demo-moderate","Severe":"demo-severe"}.get(profile_name,"demo-mild")
    demo_pill    = f"<span class='{profile_css}'>Demo: {DEMO_PROFILES.get(profile_name,{}).get('label','Off')}</span>" \
                   if st.session_state.demo_mode else ""

    st.markdown(f"""
    <div class='main-header'>
        <div>
            <div class='main-title'>🌾 CropPulse — Health Monitor</div>
            <div class='main-subtitle'>Real-time precision agriculture monitoring</div>
            <div class='header-time'>{datetime.datetime.now().strftime('%A, %d %B %Y  •  %I:%M:%S %p')}</div>
        </div>
        <div style='text-align:right;'>
            <div class='header-badge'>{health_badge}</div>
            <div style='margin-top:8px;'>{demo_pill}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    #   TOP METRIC CARDS  
    c1, c2, c3, c4, c5 = st.columns(5)

    prev_health = df_all.iloc[-5][f'health_{st.session_state.selected_plant}'] if len(df_all) >= 5 else health
    delta       = health - prev_health
    trend_html  = f"<div class='metric-trend-up'>▲ +{delta:.1f}%</div>" if delta >= 0 \
                  else f"<div class='metric-trend-down'>▼ {delta:.1f}%</div>"

    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Overall Health</div>
            <div class='metric-value' style='color:{health_color};'>{health:.1f}<span class='metric-unit'>%</span></div>
            {trend_html}
            <div class='metric-sub'>{st.session_state.selected_plant} • {health_label}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        temp = latest['temperature']
        tc   = "#0f172a" if plant_cfg['temperature']['optimal_min'] <= temp <= plant_cfg['temperature']['optimal_max'] else "#ef4444"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>🌡️ Temperature</div>
            <div class='metric-value' style='color:{tc};'>{temp:.1f}<span class='metric-unit'>°C</span></div>
            <div class='metric-sub'>Optimal: {plant_cfg["temperature"]["optimal_min"]}–{plant_cfg["temperature"]["optimal_max"]}°C</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        hum = latest['humidity']
        hc  = "#0f172a" if plant_cfg['humidity']['optimal_min'] <= hum <= plant_cfg['humidity']['optimal_max'] else "#ef4444"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>💧 Humidity</div>
            <div class='metric-value' style='color:{hc};'>{hum:.1f}<span class='metric-unit'>%</span></div>
            <div class='metric-sub'>Optimal: {plant_cfg["humidity"]["optimal_min"]}–{plant_cfg["humidity"]["optimal_max"]}%</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        sm = latest['soil_moisture']
        sc = "#0f172a" if plant_cfg['soil_moisture']['optimal_min'] <= sm <= plant_cfg['soil_moisture']['optimal_max'] else "#ef4444"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>🌱 Soil Moisture</div>
            <div class='metric-value' style='color:{sc};'>{sm:.1f}<span class='metric-unit'>%</span></div>
            <div class='metric-sub'>Optimal: {plant_cfg["soil_moisture"]["optimal_min"]}–{plant_cfg["soil_moisture"]["optimal_max"]}%</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        alerts_count   = len(st.session_state.alerts)
        critical_count = sum(1 for a in st.session_state.alerts if a.get('severity') == 'critical')
        ac = "#ef4444" if critical_count > 0 else "#f59e0b" if alerts_count > 0 else "#10b981"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>🔔 Active Alerts</div>
            <div class='metric-value' style='color:{ac};'>{alerts_count}</div>
            <div class='metric-sub'>{critical_count} critical · {alerts_count - critical_count} warning</div>
        </div>""", unsafe_allow_html=True)

    #   ROW 2: Gauges + Radar  
    st.markdown("<div class='section-header'>🎛️ Live Sensor Dashboard</div>", unsafe_allow_html=True)

    row2_left, row2_right = st.columns([3, 2])

    with row2_left:
        gauge_sensors = list(SENSORS.keys())
        fig_gauges    = make_subplots(
            rows=2, cols=3,
            specs=[[{"type":"indicator"}]*3]*2,
            subplot_titles=[f"{SENSORS[s]['icon']} {SENSORS[s]['name']}" for s in gauge_sensors],
            vertical_spacing=0.15, horizontal_spacing=0.05
        )
        for i, sensor in enumerate(gauge_sensors):
            row_idx = i // 3 + 1
            col_idx = i % 3  + 1
            val     = latest[sensor]
            cfg     = SENSORS[sensor]
            opt_min = plant_cfg[sensor]["optimal_min"]
            opt_max = plant_cfg[sensor]["optimal_max"]
            if opt_min <= val <= opt_max:
                bar_color = "#10b981"
            elif abs(val - opt_min) < (opt_max - opt_min) * 0.25 or abs(val - opt_max) < (opt_max - opt_min) * 0.25:
                bar_color = "#f59e0b"
            else:
                bar_color = "#ef4444"
            fig_gauges.add_trace(go.Indicator(
                mode="gauge+number",
                value=val,
                number={"suffix": cfg["unit"], "font": {"size": 18, "color": "#0f172a", "family": "DM Sans"}},
                gauge={
                    "axis":  {"range": [cfg["min"], cfg["max"]], "tickwidth": 1, "tickcolor": "#94a3b8", "tickfont": {"size": 9}},
                    "bar":   {"color": bar_color, "thickness": 0.55},
                    "bgcolor": "#f8fafc", "borderwidth": 0,
                    "steps": [
                        {"range": [cfg["min"], opt_min], "color": "#fee2e2"},
                        {"range": [opt_min, opt_max],    "color": "#d1fae5"},
                        {"range": [opt_max, cfg["max"]], "color": "#fee2e2"},
                    ],
                    "threshold": {"line": {"color": "#2563eb", "width": 2}, "thickness": 0.75, "value": val},
                }
            ), row=row_idx, col=col_idx)
        fig_gauges.update_layout(
            height=380, paper_bgcolor=CHART_PAPER, font=CHART_FONT,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        for ann in fig_gauges.layout.annotations:
            ann.font = dict(size=11, color="#1e293b", family="DM Sans")
        st.plotly_chart(fig_gauges, width='stretch', key="gauges")

    with row2_right:
        def normalize(val, mn, mx):
            return (val - mn) / (mx - mn) * 100 if mx > mn else 50

        radar_sensors = list(SENSORS.keys())
        sensor_labels = [f"{SENSORS[s]['icon']} {SENSORS[s]['name']}" for s in radar_sensors]
        actual_vals   = [normalize(latest[s], SENSORS[s]["min"], SENSORS[s]["max"]) for s in radar_sensors]
        optimal_vals  = [normalize((plant_cfg[s]["optimal_min"] + plant_cfg[s]["optimal_max"]) / 2,
                                   SENSORS[s]["min"], SENSORS[s]["max"]) for s in radar_sensors]
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=optimal_vals + [optimal_vals[0]], theta=sensor_labels + [sensor_labels[0]],
            fill='toself', fillcolor='rgba(37,99,235,0.10)',
            line=dict(color='#2563eb', width=2, dash='dot'), name='Optimal Range'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=actual_vals + [actual_vals[0]], theta=sensor_labels + [sensor_labels[0]],
            fill='toself', fillcolor='rgba(16,185,129,0.18)',
            line=dict(color='#10b981', width=2.5), name='Current Readings',
            marker=dict(size=6, color='#10b981')
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='#f8fafc',
                radialaxis=dict(visible=True, range=[0,100], tickfont=dict(size=9,color='#94a3b8'),
                                gridcolor='#e2e8f0', linecolor='#e2e8f0'),
                angularaxis=dict(tickfont=dict(size=11,color='#1e293b',family="DM Sans"),
                                 gridcolor='#e2e8f0', linecolor='#cbd5e1')
            ),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.12, xanchor="center", x=0.5, font=dict(size=11)),
            height=380, paper_bgcolor=CHART_PAPER, font=CHART_FONT,
            margin=dict(l=40, r=40, t=40, b=40),
            title=dict(text=f"Sensor vs Optimal — {st.session_state.selected_plant}",
                       font=dict(size=13, color="#0f172a"), x=0.5)
        )
        st.plotly_chart(fig_radar, width='stretch', key="radar")

     
    # ROW 3: SENSOR TIMELINE — 6 individual charts
     
    st.markdown("<div class='section-header'>📈 Sensor Timeline</div>", unsafe_allow_html=True)

    sensor_list = list(SENSORS.keys())

    # Row A: first 3 sensors
    tl_cols_a = st.columns(3)
    for col_idx, sensor in enumerate(sensor_list[:3]):
        cfg     = SENSORS[sensor]
        opt_min = plant_cfg[sensor]["optimal_min"]
        opt_max = plant_cfg[sensor]["optimal_max"]
        vals    = df_recent[sensor].values
        ts      = df_recent['timestamp']

        with tl_cols_a[col_idx]:
            fig = go.Figure()
            # Optimal band shading
            fig.add_hrect(y0=opt_min, y1=opt_max,
                          fillcolor="rgba(16,185,129,0.10)", line_width=0,
                          annotation_text="Optimal", annotation_position="top right",
                          annotation_font=dict(size=9, color="#10b981"))
            fig.add_trace(go.Scatter(
                x=ts, y=vals,
                mode='lines',
                line=dict(color=cfg['color'], width=2.5),
                fill='tozeroy',
                fillcolor=f"rgba({int(cfg['color'][1:3],16)},{int(cfg['color'][3:5],16)},{int(cfg['color'][5:7],16)},0.08)",
                name=cfg['name'],
                hovertemplate=f"{cfg['name']}: %{{y:.1f}}{cfg['unit']}<br>%{{x}}<extra></extra>"
            ))
            # Current value marker
            fig.add_trace(go.Scatter(
                x=[ts.iloc[-1]], y=[vals[-1]],
                mode='markers',
                marker=dict(size=8, color=cfg['color'], line=dict(color='white', width=2)),
                showlegend=False,
                hovertemplate=f"Latest: {vals[-1]:.1f}{cfg['unit']}<extra></extra>"
            ))
            fig.update_layout(
                height=220,
                plot_bgcolor=CHART_BG, paper_bgcolor=CHART_PAPER, font=CHART_FONT,
                showlegend=False,
                margin=dict(l=40, r=16, t=36, b=36),
                title=dict(text=f"{cfg['icon']} {cfg['name']}  <b style='color:{cfg['color']}'>{vals[-1]:.1f}{cfg['unit']}</b>",
                           font=dict(size=12, color="#0f172a"), x=0),
                xaxis=dict(gridcolor=GRID_COLOR, showgrid=True, tickfont=dict(color="#94a3b8", size=9),
                           showticklabels=True, tickangle=30),
                yaxis=dict(gridcolor=GRID_COLOR, showgrid=True,
                           tickfont=dict(color="#1e293b", size=10),
                           title=dict(text=cfg['unit'], font=dict(size=10, color="#94a3b8")),
                           range=[max(SENSORS[sensor]["min"], vals.min() * 0.9 - 1),
                                  min(SENSORS[sensor]["max"], vals.max() * 1.1 + 1)]),
            )
            st.plotly_chart(fig, width='stretch', key=f"timeline_{sensor}")

    # Row B: last 3 sensors
    tl_cols_b = st.columns(3)
    for col_idx, sensor in enumerate(sensor_list[3:]):
        cfg     = SENSORS[sensor]
        opt_min = plant_cfg[sensor]["optimal_min"]
        opt_max = plant_cfg[sensor]["optimal_max"]
        vals    = df_recent[sensor].values
        ts      = df_recent['timestamp']

        with tl_cols_b[col_idx]:
            fig = go.Figure()
            fig.add_hrect(y0=opt_min, y1=opt_max,
                          fillcolor="rgba(16,185,129,0.10)", line_width=0,
                          annotation_text="Optimal", annotation_position="top right",
                          annotation_font=dict(size=9, color="#10b981"))
            fig.add_trace(go.Scatter(
                x=ts, y=vals,
                mode='lines',
                line=dict(color=cfg['color'], width=2.5),
                fill='tozeroy',
                fillcolor=f"rgba({int(cfg['color'][1:3],16)},{int(cfg['color'][3:5],16)},{int(cfg['color'][5:7],16)},0.08)",
                name=cfg['name'],
                hovertemplate=f"{cfg['name']}: %{{y:.1f}}{cfg['unit']}<br>%{{x}}<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=[ts.iloc[-1]], y=[vals[-1]],
                mode='markers',
                marker=dict(size=8, color=cfg['color'], line=dict(color='white', width=2)),
                showlegend=False,
                hovertemplate=f"Latest: {vals[-1]:.1f}{cfg['unit']}<extra></extra>"
            ))
            fig.update_layout(
                height=220,
                plot_bgcolor=CHART_BG, paper_bgcolor=CHART_PAPER, font=CHART_FONT,
                showlegend=False,
                margin=dict(l=40, r=16, t=36, b=36),
                title=dict(text=f"{cfg['icon']} {cfg['name']}  <b style='color:{cfg['color']}'>{vals[-1]:.1f}{cfg['unit']}</b>",
                           font=dict(size=12, color="#0f172a"), x=0),
                xaxis=dict(gridcolor=GRID_COLOR, showgrid=True, tickfont=dict(color="#94a3b8", size=9),
                           showticklabels=True, tickangle=30),
                yaxis=dict(gridcolor=GRID_COLOR, showgrid=True,
                           tickfont=dict(color="#1e293b", size=10),
                           title=dict(text=cfg['unit'], font=dict(size=10, color="#94a3b8")),
                           range=[max(SENSORS[sensor]["min"], vals.min() * 0.9 - 1),
                                  min(SENSORS[sensor]["max"], vals.max() * 1.1 + 1)]),
            )
            st.plotly_chart(fig, width='stretch', key=f"timeline_{sensor}")

    #   ALERTS (below timeline)  
    st.markdown("<div class='section-header'>🔔 Alert Log</div>", unsafe_allow_html=True)

    alerts     = st.session_state.alerts[::-1]
    critical_n = sum(1 for a in alerts if a.get('severity') == 'critical')
    items_html = ""
    if alerts:
        for a in alerts[:20]:
            is_crit  = a.get('severity') == 'critical'
            icon     = "🔴" if is_crit else "🟡"
            left_col = "#ef4444" if is_crit else "#f59e0b"
            items_html += f"""
            <div style="padding:12px 18px;border-bottom:1px solid #f1f5f9;
                        border-left:3px solid {left_col};background:#ffffff;
                        font-family:'DM Sans',sans-serif;">
                <div style="color:#1e293b;font-size:12.5px;line-height:1.5;">{icon} {a.get('message','')}</div>
                <div style="color:#94a3b8;font-size:11px;margin-top:3px;font-family:'DM Mono',monospace;">{a.get('time','')}</div>
            </div>"""
    else:
        items_html = """<div style="padding:32px;text-align:center;color:#94a3b8;
                        font-size:13px;font-family:'DM Sans',sans-serif;">
                        ✅ No alerts — all sensors within range</div>"""

    count_bg   = "#ef4444" if critical_n > 0 else "#f59e0b" if alerts else "#10b981"
    alert_html = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;background:transparent;">
    <div style="background:#ffffff;border-radius:16px;border:1px solid #e2e8f0;
                box-shadow:0 2px 8px rgba(0,0,0,0.04);overflow:hidden;
                max-height:260px;overflow-y:auto;font-family:'DM Sans',sans-serif;">
        <div style="background:#0f1923;color:#f1f5f9;padding:14px 18px;font-size:14px;
                    font-weight:600;position:sticky;top:0;z-index:10;
                    display:flex;justify-content:space-between;align-items:center;">
            <span>🔔 Alert Log</span>
            <span style="background:{count_bg};color:white;border-radius:12px;
                         padding:2px 8px;font-size:11px;font-weight:700;">{len(alerts)}</span>
        </div>
        {items_html}
    </div></body></html>"""
    st.html(alert_html)


    #   ROW 4: Health Trends + Pie + Performance  
    st.markdown("<div class='section-header'>📊 Health Trends & Distribution</div>", unsafe_allow_html=True)

    health_col, pie_col, perf_col = st.columns([3, 2, 2])

    with health_col:
        fig_health = go.Figure()
        colors6    = ['#2563eb','#ef4444','#10b981','#f59e0b','#8b5cf6','#ec4899']
        for idx, plant in enumerate(st.session_state.plants.keys()):
            col_key = f'health_{plant}'
            if col_key in df_recent.columns:
                fig_health.add_trace(go.Scatter(
                    x=df_recent['timestamp'], y=df_recent[col_key],
                    name=plant, line=dict(color=colors6[idx % 6], width=2), mode='lines',
                    hovertemplate=f"{plant}: %{{y:.1f}}%<extra></extra>"
                ))
        fig_health.add_hrect(y0=80, y1=100, fillcolor="rgba(16,185,129,0.06)", line_width=0)
        fig_health.add_hrect(y0=60, y1=80,  fillcolor="rgba(245,158,11,0.06)", line_width=0)
        fig_health.add_hrect(y0=0,  y1=60,  fillcolor="rgba(239,68,68,0.06)",  line_width=0)
        fig_health.add_hline(y=80, line_dash="dash", line_color="#10b981", line_width=1,
                             annotation_text="Excellent", annotation_font_size=10, annotation_font_color="#10b981")
        fig_health.add_hline(y=60, line_dash="dash", line_color="#f59e0b", line_width=1,
                             annotation_text="Good",      annotation_font_size=10, annotation_font_color="#f59e0b")
        fig_health.update_layout(
            height=300, plot_bgcolor=CHART_BG, paper_bgcolor=CHART_PAPER, font=CHART_FONT,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10, color="#1e293b")),
            hovermode='x unified', margin=dict(l=40, r=20, t=40, b=40),
            xaxis=dict(title=dict(text="Time", font=dict(color="#1e293b", size=12)),
                       gridcolor=GRID_COLOR, showgrid=True, tickfont=dict(color="#1e293b", size=11)),
            yaxis=dict(title=dict(text="Health Score (%)", font=dict(color="#1e293b", size=12)),
                       range=[0,100], gridcolor=GRID_COLOR, showgrid=True, tickfont=dict(color="#1e293b", size=11)),
        )
        st.plotly_chart(fig_health, width='stretch', key="health_trend")

    with pie_col:
        health_series = df_all.tail(200)[f'health_{st.session_state.selected_plant}']
        cats = []
        for v in health_series:
            if v >= 80:   cats.append("Excellent")
            elif v >= 60: cats.append("Good")
            elif v >= 40: cats.append("Fair")
            else:         cats.append("Critical")
        cat_counts = pd.Series(cats).value_counts()
        pie_colors = {"Excellent":"#10b981","Good":"#3b82f6","Fair":"#f59e0b","Critical":"#ef4444"}
        fig_pie = go.Figure(data=[go.Pie(
            labels=cat_counts.index, values=cat_counts.values,
            marker_colors=[pie_colors.get(l,"#94a3b8") for l in cat_counts.index],
            hole=0.55, textinfo='percent', textfont=dict(size=11, color='#1e293b'),
            hovertemplate="%{label}: %{value} readings (%{percent})<extra></extra>"
        )])
        fig_pie.add_annotation(
            text=f"{health:.0f}%<br><span style='font-size:10px'>Now</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=18, color=health_color, family="DM Sans")
        )
        fig_pie.update_layout(
            height=300, paper_bgcolor=CHART_PAPER, font=CHART_FONT,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5, font=dict(size=10, color="#1e293b")),
            margin=dict(l=10, r=10, t=40, b=10),
            title=dict(text="Health Distribution", font=dict(size=13, color="#0f172a"), x=0.5)
        )
        st.plotly_chart(fig_pie, width='stretch', key="pie")

    with perf_col:
        perfs, p_labels, p_colors = [], [], []
        for sensor, cfg in SENSORS.items():
            val     = latest[sensor]
            opt_min = plant_cfg[sensor]["optimal_min"]
            opt_max = plant_cfg[sensor]["optimal_max"]
            if opt_min <= val <= opt_max:
                perf, col = 100, "#10b981"
            elif val < opt_min:
                perf, col = max(0, (val / opt_min) * 100), "#f59e0b"
            else:
                perf, col = max(0, 100*(1 - (val-opt_max)/(SENSORS[sensor]["max"]-opt_max+1e-9))), "#ef4444"
            perfs.append(perf)
            p_labels.append(f"{cfg['icon']} {cfg['name']}")
            p_colors.append(col)
        fig_perf = go.Figure(go.Bar(
            x=perfs, y=p_labels, orientation='h', marker_color=p_colors,
            text=[f"{p:.0f}%" for p in perfs], textposition='outside',
            textfont=dict(size=10), hovertemplate="%{y}: %{x:.1f}%<extra></extra>"
        ))
        fig_perf.update_layout(
            height=300, plot_bgcolor=CHART_BG, paper_bgcolor=CHART_PAPER, font=CHART_FONT,
            xaxis=dict(range=[0,115], title=dict(text="Performance (%)", font=dict(color="#1e293b", size=12)),
                       gridcolor=GRID_COLOR, showgrid=True, tickfont=dict(color="#1e293b", size=11)),
            yaxis=dict(title="", tickfont=dict(color="#1e293b", size=11)),
            margin=dict(l=10, r=40, t=40, b=40),
            title=dict(text="Sensor Performance", font=dict(size=13, color="#0f172a"), x=0.5)
        )
        st.plotly_chart(fig_perf, width='stretch', key="perf")


     
    # ROW 5: FUTURE PREDICTION — 6 individual per-sensor charts
     
    st.markdown("<div class='section-header'>🔮 Future Prediction — 6 Hours Ahead</div>", unsafe_allow_html=True)

    rf_model    = load_model()
    using_mock  = rf_model is None
    model_label = "🟡 Mock (trend-based)" if using_mock else "🟢 RF Model loaded"
    model_color = "#f59e0b" if using_mock else "#10b981"

    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
                padding:12px 20px;margin-bottom:16px;display:flex;
                justify-content:space-between;align-items:center;
                box-shadow:0 1px 4px rgba(0,0,0,0.04);">
        <div style="font-size:13px;color:#1e293b;font-weight:600;">
            Prediction Engine &nbsp;
            <span style="background:{model_color}22;color:{model_color};
                         border-radius:8px;padding:3px 10px;font-size:11px;font-weight:700;">
                {model_label}
            </span>
        </div>
        <div style="font-size:11px;color:#94a3b8;">
            Features: last {WINDOW} readings · Horizon: {HORIZON} hrs ·
            {"Place crop_rf_model.pkl alongside app.py to use real model" if using_mock else "crop_rf_model.pkl detected ✓"}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if len(df_all) >= WINDOW:
        pred_vals = mock_predict_future(df_all) if using_mock else predict_future(df_all, rf_model)

        if pred_vals:
            pred_health  = calculate_health_score(pred_vals, plant_cfg)
            delta_health = pred_health - health

            if pred_health >= 80:   ph_color = "#10b981"
            elif pred_health >= 60: ph_color = "#f59e0b"
            elif pred_health >= 40: ph_color = "#f97316"
            else:                   ph_color = "#ef4444"

            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                dh_arrow = "▲" if delta_health >= 0 else "▼"
                dh_col   = "#10b981" if delta_health >= 0 else "#ef4444"
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>🔮 Predicted Health (6hr)</div>
                    <div class='metric-value' style='color:{ph_color};'>{pred_health:.1f}<span class='metric-unit'>%</span></div>
                    <div style='color:{dh_col};font-size:12px;font-weight:600;margin-top:6px;'>
                        {dh_arrow} {abs(delta_health):.1f}% vs now
                    </div>
                </div>""", unsafe_allow_html=True)

            with pc2:
                worst_sensor = min(pred_vals, key=lambda s: pred_vals[s] / SENSORS[s]["max"])
                ws_cfg       = SENSORS[worst_sensor]
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>⚠️ Most Stressed Sensor</div>
                    <div class='metric-value' style='font-size:22px;color:#ef4444;'>{ws_cfg['icon']} {ws_cfg['name']}</div>
                    <div class='metric-sub'>Predicted: {pred_vals[worst_sensor]:.1f}{ws_cfg['unit']}</div>
                </div>""", unsafe_allow_html=True)

            with pc3:
                in_range = sum(1 for s in SENSOR_COLS
                               if plant_cfg[s]["optimal_min"] <= pred_vals[s] <= plant_cfg[s]["optimal_max"])
                ir_color = "#10b981" if in_range >= 5 else "#f59e0b" if in_range >= 3 else "#ef4444"
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-label'>✅ Sensors in Optimal Range</div>
                    <div class='metric-value' style='color:{ir_color};'>{in_range}<span class='metric-unit'>/6</span></div>
                    <div class='metric-sub'>Predicted 6 hrs from now</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

            #   6 individual forecast charts (3 per row)  
            # FIX: Convert pd.Timestamp to string for Plotly compatibility
            last_ts  = df_recent['timestamp'].iloc[-1]
            last_ts_str = str(last_ts)
            hist_ts  = [str(t) for t in df_recent['timestamp'].tolist()]
            # Build future timestamps as strings for X axis labels
            future_ts     = [(last_ts + datetime.timedelta(hours=i+1)) for i in range(HORIZON)]
            future_ts_str = [str(t) for t in future_ts]

            pred_row_a = st.columns(3)
            pred_row_b = st.columns(3)

            for s_idx, sensor in enumerate(SENSOR_COLS):
                cfg       = SENSORS[sensor]
                opt_min   = plant_cfg[sensor]["optimal_min"]
                opt_max   = plant_cfg[sensor]["optimal_max"]
                hist_vals = df_recent[sensor].tolist()
                curr_val  = float(hist_vals[-1])
                pred_val  = pred_vals[sensor]

                # Linear interpolation from now → predicted value at +6 hrs
                interp_ts   = [last_ts_str] + future_ts_str
                interp_vals = [curr_val + (pred_val - curr_val) * (i / HORIZON) for i in range(HORIZON + 1)]

                # Determine forecast endpoint color
                if opt_min <= pred_val <= opt_max:
                    dot_color = "#10b981"
                elif abs(pred_val - opt_min) < (opt_max - opt_min) * 0.25 or \
                     abs(pred_val - opt_max) < (opt_max - opt_min) * 0.25:
                    dot_color = "#f59e0b"
                else:
                    dot_color = "#ef4444"

                # Dynamic Y range — show both history and forecast comfortably
                all_vals  = hist_vals + interp_vals
                y_min     = max(SENSORS[sensor]["min"], min(all_vals) - (max(all_vals) - min(all_vals)) * 0.15 - 1)
                y_max     = min(SENSORS[sensor]["max"], max(all_vals) + (max(all_vals) - min(all_vals)) * 0.15 + 1)

                fig = go.Figure()

                # Optimal band
                fig.add_hrect(y0=opt_min, y1=opt_max,
                              fillcolor="rgba(16,185,129,0.10)", line_width=0)

                # Historical line (solid)
                fig.add_trace(go.Scatter(
                    x=hist_ts, y=hist_vals,
                    mode='lines',
                    line=dict(color=cfg['color'], width=2.5),
                    name='History',
                    hovertemplate=f"{cfg['name']}: %{{y:.1f}}{cfg['unit']}<extra></extra>"
                ))

                # Forecast line (dotted) — connects from last historical point
                fig.add_trace(go.Scatter(
                    x=interp_ts, y=interp_vals,
                    mode='lines+markers',
                    line=dict(color=cfg['color'], width=2, dash='dot'),
                    marker=dict(
                        size=[0] * HORIZON + [10],
                        color=[cfg['color']] * HORIZON + [dot_color],
                        line=dict(color='white', width=2),
                        symbol='circle'
                    ),
                    name='Forecast',
                    hovertemplate=f"Forecast: %{{y:.1f}}{cfg['unit']}<extra></extra>"
                ))

                # Vertical "Now" line — use add_shape to avoid Plotly's internal _mean() on mixed types
                fig.add_shape(
                    type="line", x0=last_ts_str, x1=last_ts_str,
                    y0=0, y1=1, yref="paper",
                    line=dict(dash="dash", color="#64748b", width=1),
                )
                fig.add_annotation(
                    x=last_ts_str, y=1.0, yref="paper",
                    text="Now", showarrow=False,
                    font=dict(size=10, color="#64748b"),
                    xanchor="center", yanchor="bottom"
                )

                # Predicted value annotation
                delta_str = f"+{pred_val - curr_val:.1f}" if pred_val >= curr_val else f"{pred_val - curr_val:.1f}"
                fig.add_annotation(
                    x=future_ts_str[-1], y=pred_val,
                    text=f"  {pred_val:.1f}{cfg['unit']} ({delta_str})",
                    showarrow=False,
                    font=dict(size=10, color=dot_color),
                    xanchor="left"
                )

                fig.update_layout(
                    height=260,
                    plot_bgcolor=CHART_BG, paper_bgcolor=CHART_PAPER, font=CHART_FONT,
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                xanchor="right", x=1, font=dict(size=9, color="#64748b")),
                    margin=dict(l=44, r=60, t=38, b=38),
                    title=dict(text=f"{cfg['icon']} {cfg['name']}",
                               font=dict(size=12, color="#0f172a"), x=0),
                    xaxis=dict(gridcolor=GRID_COLOR, showgrid=True,
                               tickfont=dict(color="#94a3b8", size=9), tickangle=30),
                    yaxis=dict(gridcolor=GRID_COLOR, showgrid=True,
                               tickfont=dict(color="#1e293b", size=10),
                               title=dict(text=cfg['unit'], font=dict(size=10, color="#94a3b8")),
                               range=[y_min, y_max]),
                    hovermode='x unified'
                )

                target_col = pred_row_a[s_idx] if s_idx < 3 else pred_row_b[s_idx - 3]
                with target_col:
                    st.plotly_chart(fig, width='stretch', key=f"pred_{sensor}")

            #   Compact sensor breakdown table  
            st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
            breakdown_cols = st.columns(6)
            for s_idx, sensor in enumerate(SENSOR_COLS):
                cfg     = SENSORS[sensor]
                curr    = float(df_all[sensor].iloc[-1])
                pred    = pred_vals[sensor]
                opt_min = plant_cfg[sensor]["optimal_min"]
                opt_max = plant_cfg[sensor]["optimal_max"]
                in_opt  = opt_min <= pred <= opt_max
                arrow   = "▲" if pred > curr else "▼" if pred < curr else "→"
                a_color = "#10b981" if pred > curr else "#ef4444" if pred < curr else "#94a3b8"
                badge_bg   = "#d1fae5" if in_opt else "#fee2e2"
                badge_col  = "#065f46" if in_opt else "#991b1b"
                badge_text = "✓ Optimal" if in_opt else "✗ Out of range"
                with breakdown_cols[s_idx]:
                    st.markdown(f"""
                    <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;
                                padding:12px 14px;text-align:center;">
                        <div style="font-size:18px;">{cfg['icon']}</div>
                        <div style="font-size:11px;font-weight:600;color:#1e293b;margin:4px 0;">{cfg['name']}</div>
                        <div style="font-size:10px;color:#64748b;">Now: <b>{curr:.1f}{cfg['unit']}</b></div>
                        <div style="font-size:13px;font-weight:700;color:{a_color};margin:4px 0;">
                            {arrow} {pred:.1f}{cfg['unit']}
                        </div>
                        <div style="background:{badge_bg};color:{badge_col};border-radius:6px;
                                    padding:2px 6px;font-size:9px;font-weight:600;display:inline-block;">
                            {badge_text}
                        </div>
                    </div>""", unsafe_allow_html=True)

    else:
        st.info(f"Need at least {WINDOW} data points for prediction. Currently have {len(df_all)}. Keep demo mode running to collect more data.")

    #   FOOTER  
    st.markdown("---")
    st.markdown(f"""
    <div style='display:flex;justify-content:space-between;align-items:center;padding:4px 0;'>
        <div style='color:#94a3b8;font-size:11px;'>🌾 CropPulse v5.0 — Precision Crop Health Monitor</div>
        <div style='color:#94a3b8;font-size:11px;'>
            {len(df_all)} data points · {st.session_state.selected_plant} ·
            Last update: {latest['timestamp'].strftime('%H:%M:%S') if hasattr(latest['timestamp'],'strftime') else str(latest['timestamp'])[:19]}
            · 🔄 Live updates: {st.session_state.refresh_rate}s
        </div>
    </div>
    """, unsafe_allow_html=True)


#   Call the fragment function to render the live dashboard  
render_live_dashboard()
