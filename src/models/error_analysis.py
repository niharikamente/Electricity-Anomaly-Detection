import csv
from pathlib import Path

project = Path(__file__).resolve().parents[2]
scores_file = project / "results" / "MT_196_isolation_forest_scores.csv"
report_file = project / "reports" / "error_analysis.md"

records = []
with scores_file.open(encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        records.append({
            "timestamp": row["timestamp"],
            "consumption_kw": float(row["consumption_kw"]),
            "z_score": float(row["previous_24h_z_score"]),
            "score": float(row["isolation_forest_score"]),
            "is_flagged": row["isolation_forest_flag"].lower() == "true",
        })

# Sort by anomaly score descending
records.sort(key=lambda x: x["score"], reverse=True)

# Find characteristics of flagged anomalies
flagged = [r for r in records if r["is_flagged"]]
zero_consumption = [r for r in flagged if r["consumption_kw"] <= 0.1]
high_z_score = [r for r in flagged if r["z_score"] > 5.0]

report_content = f"""# Error Analysis and Model Limitations

## Summary of Findings

Out of {len(records):,} total readings, the Isolation Forest model flagged {len(flagged):,} anomalies.

### Common Failure Modes & True Positives
1. **Sensor Dropouts (Zero Consumption):**
   - We observed that {len(zero_consumption)} out of {len(flagged)} flagged anomalies have near-zero consumption (<= 0.1 kW).
   - *Limitation:* The model correctly identifies these as mathematically anomalous, but without external context, we cannot tell if this is a true power outage or just a sensor reporting failure.
2. **Sudden Spikes (High Z-Score):**
   - {len(high_z_score)} flagged anomalies have a z-score > 5.0, indicating a massive deviation from the previous 24 hours.
   - *Limitation:* While these are likely genuine unusual events (e.g. equipment turned on during off-hours), the model lacks information about public holidays or special events which might naturally explain the spike.

## Conclusion and Model Limitations
- **Contextual Blindness:** The model operates purely on time and historical consumption. It does not know if a business is closed for a holiday or maintenance.
- **Unsupervised Nature:** Because there are no manual labels, we are strictly detecting mathematical outliers. Some "anomalies" may actually be the new normal (concept drift).
- **False Positives:** Gradual seasonal changes might occasionally trigger false positives if the 24-hour rolling average lags behind a sudden weather change.

## Top 5 Most Anomalous Readings
"""

for i, r in enumerate(records[:5], 1):
    report_content += f"{i}. **{r['timestamp']}**: {r['consumption_kw']:.2f} kW (Z-Score: {r['z_score']:.2f}, Anomaly Score: {r['score']:.4f})\n"

report_file.parent.mkdir(parents=True, exist_ok=True)
with report_file.open("w", encoding="utf-8") as f:
    f.write(report_content)

print(f"Error analysis report generated at {report_file}")
