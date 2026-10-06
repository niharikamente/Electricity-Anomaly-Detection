import csv
from pathlib import Path

import matplotlib.pyplot as plt

project = Path(__file__).resolve().parents[2]
data_file = project / "data" / "processed" / "LD2011_2014_clean.csv"
chart_file = project / "results" / "average_daily_profile_MT_196.png"
client = "MT_196"

hour_totals = [0.0] * 24
hour_counts = [0] * 24

with data_file.open(encoding="utf-8", newline="") as file:
    reader = csv.reader(file)
    header = next(reader)
    client_index = header.index(client)

    for row in reader:
        hour = int(row[0][11:13])
        hour_totals[hour] += float(row[client_index])
        hour_counts[hour] += 1

hour_averages = [
    hour_totals[hour] / hour_counts[hour]
    for hour in range(24)
]

chart_file.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 4))
plt.plot(range(24), hour_averages, marker="o")
plt.title(f"Average daily electricity profile: {client}")
plt.xlabel("Hour of day")
plt.ylabel("Average consumption (kW)")
plt.xticks(range(0, 24, 2))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(chart_file, dpi=150)

print(f"Saved chart to {chart_file}")