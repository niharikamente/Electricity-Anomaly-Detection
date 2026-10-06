import csv
import joblib
import numpy as np
from datetime import datetime
from math import cos, pi, sin
from pathlib import Path
from sklearn.ensemble import IsolationForest

project = Path(__file__).resolve().parents[2]
source = project / "data" / "processed" / "MT_196_features.csv"
scores_file = project / "results" / "MT_196_isolation_forest_scores.csv"
model_file = project / "models" / "MT_196_isolation_forest.joblib"

features = []
records = []

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

        records.append([
            row["timestamp"],
            row["consumption_kw"],
            z_text,
        ])

X = np.asarray(features, dtype=float)

model = IsolationForest(
    n_estimators=200,
    contamination=0.00115,
    random_state=42,
    n_jobs=-1,
)
model.fit(X)

# Higher scores mean more unusual.
anomaly_scores = -model.score_samples(X)
predictions = model.predict(X)

scores_file.parent.mkdir(parents=True, exist_ok=True)
model_file.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(model, model_file)

flagged = 0
with scores_file.open("w", encoding="utf-8", newline="") as output_file:
    writer = csv.writer(output_file)
    writer.writerow([
        "timestamp",
        "client",
        "consumption_kw",
        "previous_24h_z_score",
        "isolation_forest_score",
        "isolation_forest_flag",
    ])

    for record, score, prediction in zip(records, anomaly_scores, predictions):
        is_anomaly = prediction == -1
        flagged += is_anomaly
        writer.writerow([
            record[0],
            "MT_196",
            record[1],
            record[2],
            score,
            is_anomaly,
        ])

print(f"Scored readings: {len(records):,}")
print(f"Flagged for review: {flagged:,}")
print(f"Saved scores to {scores_file}")
print(f"Saved model to {model_file}")