import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "data" / "processed" / "MT_196_features.csv"
CLUSTERS = ROOT / "results" / "MT_196_kmeans_scores.csv"
OUTPUT = ROOT / "results" / "MT_196_kmeans_pca.png"
MAX_POINTS = 10_000

records = []
features = []

with FEATURES.open(encoding="utf-8", newline="") as file:
    for row in csv.DictReader(file):
        if not row["previous_24h_z_score"]:
            continue

        hour_angle = 2 * np.pi * (
            int(row["hour"]) + int(row["timestamp"][14:16]) / 60
        ) / 24
        weekday_angle = 2 * np.pi * int(row["day_of_week"]) / 7

        features.append([
            float(row["consumption_kw"]),
            float(row["previous_24h_z_score"]),
            np.sin(hour_angle),
            np.cos(hour_angle),
            np.sin(weekday_angle),
            np.cos(weekday_angle),
        ])
        records.append(row["timestamp"])

with CLUSTERS.open(encoding="utf-8", newline="") as file:
    cluster_rows = list(csv.DictReader(file))

rng = np.random.default_rng(42)
indices = np.sort(
    rng.choice(len(records), size=min(MAX_POINTS, len(records)), replace=False)
)

X = np.asarray(features, dtype=float)[indices]
scaled = StandardScaler().fit_transform(X)
projection = PCA(n_components=2, random_state=42).fit_transform(scaled)

labels = np.asarray([
    int(cluster_rows[index]["cluster"]) for index in indices
])
flags = np.asarray([
    cluster_rows[index]["flag"].lower() == "true" for index in indices
])

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(9, 6))
points = plt.scatter(
    projection[:, 0], projection[:, 1],
    c=labels, cmap="tab10", s=8, alpha=0.55,
)
if flags.any():
    plt.scatter(
        projection[flags, 0], projection[flags, 1],
        facecolors="none", edgecolors="black", s=28,
        label="Distance flagged",
    )

plt.title("MT_196 usage patterns: PCA view of K-means clusters")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.colorbar(points, label="K-means cluster")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig(OUTPUT, dpi=160)
print(f"Saved PCA cluster chart to {OUTPUT}")