# train_model.py  (run this separately)
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import pickle

df      = pd.read_csv("rf_training_data.csv")
feat_cols  = [c for c in df.columns if not c.startswith("target_")]
label_cols = [c for c in df.columns if c.startswith("target_")]

X, y = df[feat_cols].values, df[label_cols].values

model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X, y)

with open("crop_rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved!")