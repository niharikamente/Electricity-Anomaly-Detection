import csv
from datetime import datetime
from math import cos, pi, sin
from pathlib import Path

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data" / "processed" / "MT_196_features.csv"
IF_SCORES = ROOT / "results" / "MT_196_isolation_forest_scores.csv"
BASELINE = ROOT / "results" / "MT_196_baseline_anomalies.csv"
OUTPUT = ROOT / "results" / "MT_196_lof_sample_scores.csv"

SAMPLE_SIZE = 10_000
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
        records.append((
            row["timestamp"],
            row["consumption_kw"],
            row["previous_24h_z_score"],
        ))

X = np.asarray(features, dtype=float)
rng = np.random.default_rng(42)
indices = np.sort(
    rng.choice(len(records), size=min(SAMPLE_SIZE, len(records)), replace=False)
)

# Scaling matters for LOF because it compares distances between readings.
X_sample = StandardScaler().fit_transform(X[indices])

model = LocalOutlierFactor(
    n_neighbors=35,
    contamination=ALERT_RATE,
    n_jobs=-1,
)
labels = model.fit_predict(X_sample)
scores = -model.negative_outlier_factor_  # Higher means more unusual.

with IF_SCORES.open(encoding="utf-8", newline="") as file:
    if_flags = {
        row["timestamp"]: row["isolation_forest_flag"].lower() == "true"
        for row in csv.DictReader(file)
    }

with BASELINE.open(encoding="utf-8", newline="") as file:
    baseline_flags = {
        row["timestamp"] for row in csv.DictReader(file)
    }

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "timestamp",
        "client",
        "consumption_kw",
        "rolling_z_score",
        "rolling_z_flag",
        "isolation_forest_flag",
        "lof_score",
        "lof_flag",
    ])

    for index, score, label in zip(indices, scores, labels):
        timestamp, reading, z_score = records[index]
        writer.writerow([
            timestamp,
            "MT_196",
            reading,
            z_score,
            timestamp in baseline_flags,
            if_flags.get(timestamp, False),
            score,
            label == -1,
        ])

print(f"LOF sample size: {len(indices):,}")
print(f"LOF flags in sample: {int(np.sum(labels == -1))}")
print(f"Saved comparison sample to {OUTPUT}")