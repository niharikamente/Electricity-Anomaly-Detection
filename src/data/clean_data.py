import csv
from datetime import datetime
from pathlib import Path

project = Path(__file__).resolve().parents[2]
source = project / "data" / "raw" / "LD2011_2014.txt"
destination = project / "data" / "processed" / "LD2011_2014_clean.csv"

with source.open(encoding="utf-8-sig", newline="") as input_file:
    reader = csv.reader(input_file, delimiter=";")
    header = next(reader)
    header[0] = "timestamp"

    with destination.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(header)

        rows = 0
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"Unexpected number of columns at data row {rows + 1}")

            datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")

            # Convert decimal commas to decimal points; keep zero readings.
            for i in range(1, len(row)):
                number = float(row[i].replace(",", "."))
                row[i] = str(number)

            writer.writerow(row)
            rows += 1

print(f"Saved {rows:,} rows to {destination}")
print(f"Client columns: {len(header) - 1}")