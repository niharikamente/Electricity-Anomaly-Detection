import csv
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOF_FILE = ROOT / "results" / "MT_196_lof_sample_scores.csv"
KMEANS_FILE = ROOT / "results" / "MT_196_kmeans_scores.csv"
OUTPUT = ROOT / "results" / "MT_196_method_comparison.csv"

with LOF_FILE.open(encoding="utf-8", newline="") as file:
    sample_rows = list(csv.DictReader(file))

with KMEANS_FILE.open(encoding="utf-8", newline="") as file:
    kmeans_flags = {
        row["timestamp"]: row["flag"].lower() == "true"
        for row in csv.DictReader(file)
    }

methods = {
    "Rolling z-score": {
        row["timestamp"] for row in sample_rows
        if row["rolling_z_flag"].lower() == "true"
    },
    "Isolation Forest": {
        row["timestamp"] for row in sample_rows
        if row["isolation_forest_flag"].lower() == "true"
    },
    "Local Outlier Factor": {
        row["timestamp"] for row in sample_rows
        if row["lof_flag"].lower() == "true"
    },
    "K-means distance": {
        row["timestamp"] for row in sample_rows
        if kmeans_flags.get(row["timestamp"], False)
    },
}

sample_size = len(sample_rows)
output_rows = []

for name, flagged in methods.items():
    output_rows.append([
        "method",
        name,
        sample_size,
        len(flagged),
        len(flagged) / sample_size,
        "",
        "",
        "",
    ])

for first, second in combinations(methods, 2):
    a, b = methods[first], methods[second]
    union = a | b
    shared = a & b
    jaccard = len(shared) / len(union) if union else 0.0
    output_rows.append([
        "pairwise",
        "",
        sample_size,
        "",
        "",
        first,
        second,
        f"{len(shared)} shared; Jaccard={jaccard:.4f}",
    ])

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT.open("w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([
        "row_type", "method", "sample_size", "flagged",
        "flag_rate", "method_a", "method_b", "overlap",
    ])
    writer.writerows(output_rows)

print(f"Compared {len(methods)} methods over {sample_size:,} shared timestamps.")
for name, flagged in methods.items():
    print(f"{name}: {len(flagged)} flags ({len(flagged) / sample_size:.3%})")
print(f"Saved comparison to {OUTPUT}")