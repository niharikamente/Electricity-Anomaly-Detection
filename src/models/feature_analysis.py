import csv
import joblib
import numpy as np
from datetime import datetime
from math import cos, pi, sin
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

project = Path(__file__).resolve().parents[2]
source = project / "data" / "processed" / "MT_196_features.csv"
model_file = project / "models" / "MT_196_isolation_forest.joblib"
output_file = project / "results" / "MT_196_feature_importance.csv"

# 1. Load data
features = []
with source.open(encoding="utf-8", newline="") as input_file:
    for row in csv.DictReader(input_file):
        z_text = row["previous_24h_z_score"]
        if not z_text:
            continue

        timestamp = datetime.strptime(
            row["timestamp"], "%Y-%m-%d %H:%M:%S"
        )
        minute_of_day = timestamp.hour * 60 + timestamp.minute
        time_angle = 2 * pi * minute_of_day / 1440
        week_angle = 2 * pi * timestamp.weekday() / 7

        features.append([
            float(row["consumption_kw"]),
            float(z_text),
            sin(time_angle),
            cos(time_angle),
            sin(week_angle),
            cos(week_angle),
        ])

X = np.asarray(features, dtype=float)

# 2. Load the isolation forest model and get anomaly scores
iso_model = joblib.load(model_file)
# Higher scores mean more unusual
anomaly_scores = -iso_model.score_samples(X)

# 3. Train a surrogate Random Forest model to explain the anomaly scores
print("Training surrogate model to calculate feature importances...")
rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
rf.fit(X, anomaly_scores)

# 4. Extract and save feature importances
feature_names = [
    "consumption_kw",
    "previous_24h_z_score",
    "time_angle_sin",
    "time_angle_cos",
    "week_angle_sin",
    "week_angle_cos"
]
importances = rf.feature_importances_

output_file.parent.mkdir(parents=True, exist_ok=True)
with output_file.open("w", encoding="utf-8", newline="") as out_file:
    writer = csv.writer(out_file)
    writer.writerow(["feature", "importance"])
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        writer.writerow([name, imp])
        print(f"{name}: {imp:.4f}")

print(f"Feature importances saved to {output_file}")
