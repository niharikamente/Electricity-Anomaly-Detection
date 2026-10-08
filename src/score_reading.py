import argparse
from datetime import datetime
from math import cos, pi, sin
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "MT_196_isolation_forest.joblib"

parser = argparse.ArgumentParser(
    description="Score one new MT_196 reading with Isolation Forest."
)
parser.add_argument("--timestamp", required=True)
parser.add_argument("--consumption", required=True, type=float, help="Reading in kW")
parser.add_argument("--previous-mean", required=True, type=float)
parser.add_argument("--previous-std", required=True, type=float)
args = parser.parse_args()

if args.previous_std <= 0:
    parser.error("--previous-std must be greater than zero")
if not MODEL.exists():
    parser.error(f"Model not found at {MODEL}; run src/models/isolation_forest.py first.")

timestamp = datetime.strptime(args.timestamp, "%Y-%m-%d %H:%M:%S")
z_score = (args.consumption - args.previous_mean) / args.previous_std

time_angle = 2 * pi * (timestamp.hour * 60 + timestamp.minute) / 1440
week_angle = 2 * pi * timestamp.weekday() / 7

features = [[
    args.consumption,
    z_score,
    sin(time_angle),
    cos(time_angle),
    sin(week_angle),
    cos(week_angle),
]]

model = joblib.load(MODEL)
prediction = int(model.predict(features)[0])
score = float(-model.score_samples(features)[0])

print("Client: MT_196")
print(f"Anomaly score (higher is more unusual): {score:.5f}")
print("Flag for review" if prediction == -1 else "Not flagged by this model")
print("This is a screening result, not a confirmed fault.")