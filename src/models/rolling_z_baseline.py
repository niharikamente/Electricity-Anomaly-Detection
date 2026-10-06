import csv
from pathlib import Path

project = Path(__file__).resolve().parents[2]
source = project / "data" / "processed" / "MT_196_features.csv"
destination = project / "results" / "MT_196_baseline_anomalies.csv"

threshold = 3.0
scored = 0
flagged = 0

with source.open(encoding="utf-8", newline="") as input_file:
    reader = csv.DictReader(input_file)

    with destination.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            "timestamp",
            "client",
            "consumption_kw",
            "previous_24h_mean_kw",
            "deviation_kw",
            "anomaly_score",
        ])

        for row in reader:
            z_text = row["previous_24h_z_score"]
            if not z_text:
                continue

            scored += 1
            score = abs(float(z_text))

            if score > threshold:
                flagged += 1
                writer.writerow([
                    row["timestamp"],
                    "MT_196",
                    row["consumption_kw"],
                    row["previous_24h_mean_kw"],
                    row["deviation_from_previous_24h_kw"],
                    score,
                ])

print(f"Scored readings: {scored:,}")
print(f"Flagged for review: {flagged:,} (threshold: |z| > {threshold})")
print(f"Saved flagged readings to {destination}")