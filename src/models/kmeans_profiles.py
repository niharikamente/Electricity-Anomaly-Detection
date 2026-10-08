import csv
from datetime import datetime
from math import cos, pi, sin
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data" / "processed" / "MT_196_features.csv"
OUTPUT = ROOT / "results" / "MT_196_kmeans_scores.csv"
MODEL_FILE = ROOT / "models" / "MT_196_kmeans_profiles.joblib"
ALERT_RATE = 0.00115

records = []
features = []

with FEATURES.open(encoding="utf-8", newline="") as file:
    for row in csv.DictReader(file):
        if not row["previous_24h_z_score"]:
            continue

        timestamp = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
        time_angle = 2 * pi * (timestamp.hour * 60 + timestamp.minute) / 1440
        week_angle = 2 * pi * timestamp.weekday() / 7

        features.append([
            float(row["consumption_kw"]),
            float(row["previous_24h_z_score"]),
            sin(time_angle),
            cos(time_angle),
            sin(week_angle),
            cos(week_angle),
        ])
        records.append((row["timestamp"], row["consumption_kw"]))

X = np.asarray(features, dtype=float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=3, n_init=10, random_state=42)
clusters = model.fit_predict(X_scaled)

distances = np.linalg.norm(
    X_scaled - model.cluster_centers_[clusters],
    axis=1,
)
threshold = float(np.quantile(distances, 1 - ALERT_RATE))
flags = distances >= threshold

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT.open("w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "timestamp", "client", "consumption_kw",
        "cluster", "distance_score", "flag",
    ])
    for record, cluster, distance, flag in zip(
        records, clusters, distances, flags
    ):
        writer.writerow([
            record[0], "MT_196", record[1],
            int(cluster), float(distance), bool(flag),
        ])

joblib.dump(
    {"scaler": scaler, "kmeans": model, "distance_threshold": threshold},
    MODEL_FILE,
)

print(f"Scored readings: {len(records):,}")
print(f"Distance flags: {int(np.sum(flags)):,}")
print(f"Saved scores to {OUTPUT}")
print(f"Saved clustering model to {MODEL_FILE}")