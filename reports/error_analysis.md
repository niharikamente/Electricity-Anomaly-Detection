# Error Analysis and Model Limitations

## Summary of Findings

Out of 140,160 total readings, the Isolation Forest model flagged 162 anomalies.

### Common Failure Modes & True Positives
1. **Sensor Dropouts (Zero Consumption):**
   - We observed that 0 out of 162 flagged anomalies have near-zero consumption (<= 0.1 kW).
   - *Limitation:* The model correctly identifies these as mathematically anomalous, but without external context, we cannot tell if this is a true power outage or just a sensor reporting failure.
2. **Sudden Spikes (High Z-Score):**
   - 1 flagged anomalies have a z-score > 5.0, indicating a massive deviation from the previous 24 hours.
   - *Limitation:* While these are likely genuine unusual events (e.g. equipment turned on during off-hours), the model lacks information about public holidays or special events which might naturally explain the spike.

## Conclusion and Model Limitations
- **Contextual Blindness:** The model operates purely on time and historical consumption. It does not know if a business is closed for a holiday or maintenance.
- **Unsupervised Nature:** Because there are no manual labels, we are strictly detecting mathematical outliers. Some "anomalies" may actually be the new normal (concept drift).
- **False Positives:** Gradual seasonal changes might occasionally trigger false positives if the 24-hour rolling average lags behind a sudden weather change.

## Top 5 Most Anomalous Readings
1. **2011-08-08 13:30:00**: 43583.33 kW (Z-Score: 2.46, Anomaly Score: 0.6404)
2. **2011-09-26 13:30:00**: 41833.33 kW (Z-Score: 2.36, Anomaly Score: 0.6402)
3. **2011-10-03 13:30:00**: 42333.33 kW (Z-Score: 2.09, Anomaly Score: 0.6350)
4. **2011-07-25 13:00:00**: 44041.67 kW (Z-Score: 2.03, Anomaly Score: 0.6349)
5. **2014-08-19 16:45:00**: 78541.67 kW (Z-Score: 4.35, Anomaly Score: 0.6343)
