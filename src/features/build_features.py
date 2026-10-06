import csv
from collections import deque
from datetime import datetime
from math import sqrt
from pathlib import Path

project = Path(__file__).resolve().parents[2]
source = project / "data" / "processed" / "LD2011_2014_clean.csv"
destination = project / "data" / "processed" / "MT_196_features.csv"
client = "MT_196"
window_size = 96  # 96 readings × 15 minutes = 24 hours

window = deque()
window_sum = 0.0
window_sum_squares = 0.0
rows = 0

with source.open(encoding="utf-8", newline="") as input_file:
    reader = csv.DictReader(input_file)

    with destination.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            "timestamp",
            "consumption_kw",
            "hour",
            "day_of_week",
            "is_weekend",
            "previous_reading_kw",
            "previous_24h_mean_kw",
            "deviation_from_previous_24h_kw",
            "previous_24h_z_score",
        ])

        for row in reader:
            timestamp_text = row["timestamp"]
            timestamp = datetime.strptime(
                timestamp_text, "%Y-%m-%d %H:%M:%S"
            )
            current = float(row[client])

            previous_reading = window[-1] if window else ""
            if len(window) == window_size:
                baseline = window_sum / window_size
                variance = max(
                    window_sum_squares / window_size - baseline**2,
                    0.0,
                )
                std_dev = sqrt(variance)
                deviation = current - baseline
                z_score = deviation / std_dev if std_dev > 0 else ""
            else:
                baseline = deviation = z_score = ""

            writer.writerow([
                timestamp_text,
                current,
                timestamp.hour,
                timestamp.weekday(),
                int(timestamp.weekday() >= 5),
                previous_reading,
                baseline,
                deviation,
                z_score,
            ])

            if len(window) == window_size:
                oldest = window.popleft()
                window_sum -= oldest
                window_sum_squares -= oldest**2

            window.append(current)
            window_sum += current
            window_sum_squares += current**2
            rows += 1

print(f"Saved {rows:,} feature rows to {destination}")
print("The first 96 rows have no previous-day baseline yet.")