# Electricity Consumption Anomaly Detection Using Unsupervised Machine Learning

## Team 7

A machine learning project for detecting anomalies in electricity consumption using historical consumption and time-related features.

## Project Objective

The objective is to identify unusual or anomalous electricity consumption patterns for specific clients (e.g., `MT_196`) using historical energy consumption data and time-based features. The project uses an Isolation Forest model to score and flag potential anomalies.

## Dataset

**UCI ElectricityLoadDiagrams20112014**

Dataset:
https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014

The dataset contains electricity consumption measurements recorded at 15-minute intervals for multiple meters (clients).

## Features


The project extracts the following features for anomaly detection:

- consumption (kW)
- z_score (based on rolling mean and standard deviation)
- sin(time_angle) (time of day encoded as sine)
- cos(time_angle) (time of day encoded as cosine)
- sin(week_angle) (day of week encoded as sine)
- cos(week_angle) (day of week encoded as cosine)

These features help capture cyclical patterns in energy consumption.

## Machine Learning Models

The following anomaly detection model was used:

1. Isolation Forest
   - **Algorithm**: Tree-based ensemble method that isolates observations by randomly selecting a feature and splitting value. Because anomalies require fewer splits to isolate, they appear noticeably closer to the root of the trees.
   - **Configuration**:
     - `n_estimators`: 200 trees
     - `contamination`: 0.00115 (~0.115% expected anomaly rate)
     - `random_state`: 42

## Model Evaluation

The model was evaluated by calculating the anomaly score for consumption readings. Readings with a high anomaly score (Isolation Forest prediction of `-1`) are flagged for further review as potential faults or abnormal usage.

- **Total Readings Evaluated**: 140,160
- **Flagged Anomalies**: 162
- **Scoring**: Higher anomaly score indicates more severe deviation from normal consumption behaviors.

## Note on Anomaly Detection

This tool acts as a screening mechanism to highlight unusual behavior. Flagged readings should undergo further review and are not definitively confirmed faults.

## Project Structure

```text
ML-MINI-PROJECT/
│
├── README.md
├── requirements.txt
│
├── data/
│
├── models/
│   └── MT_196_isolation_forest.joblib
│
├── notebooks/
│
├── outputs/
│
├── reports/
│   └── error_analysis.md
│
└── src/
    ├── data/
    │   └── clean_data.py
    ├── features/
    │   └── build_features.py
    ├── models/
    │   ├── error_analysis.py
    │   ├── feature_analysis.py
    │   └── isolation_forest.py
    ├── score_reading.py
    └── visualization/
```
