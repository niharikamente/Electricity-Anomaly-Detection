import csv
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt

project = Path(__file__).resolve().parents[2]
data_file = project / "data" / "processed" / "LD2011_2014_clean.csv"
chart_file = project / "results" / "first_week_MT_001.png"

times = []
readings = []
first_time = None
client = "MT_001"

with data_file.open(encoding="utf-8", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        time = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")

        if first_time is None:
            first_time = time

        if time >= first_time + timedelta(days=7):
            break

        times.append(time)
        readings.append(float(row[client]))

chart_file.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(11, 4))
plt.plot(times, readings, linewidth=0.8)
plt.title(f"{client} electricity use during the first week")
plt.xlabel("Time")
plt.ylabel("Consumption (kW)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(chart_file, dpi=150)

print(f"Saved chart to {chart_file}")