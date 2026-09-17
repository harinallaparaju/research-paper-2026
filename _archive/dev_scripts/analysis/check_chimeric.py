"""Check chimeric mapping for duplicate iris subjects."""
import json, re
from collections import Counter

with open("chimeric_db/chimeric_fvc2002_1_min2.json") as f:
    data = json.load(f)

# Show first 3 entries
for entry in data["mapping"][:3]:
    cid = entry["chimeric_id"]
    for p in entry["pairs"]:
        print(f"  CID={cid} imp={p['impression']}: iris={p['iris_file'][-50:]}")
    print()

# Extract iris subject IDs
iris_subjects = []
for entry in data["mapping"]:
    iris_file = entry["pairs"][0]["iris_file"]
    # Path format: .../CASIA-Iris-Interval/S1001/L/S1001L01.jpg
    parts = iris_file.replace("\\", "/").split("/")
    for part in parts:
        if re.match(r"^S\d{4}$", part):
            iris_subjects.append(part)
            break

print(f"Total FP subjects: {len(data['mapping'])}")
print(f"Extracted iris IDs: {len(iris_subjects)}")
print(f"Unique iris subjects: {len(set(iris_subjects))}")

dupes = {k: v for k, v in Counter(iris_subjects).items() if v > 1}
if dupes:
    print(f"DUPLICATES: {dupes}")
else:
    print("All iris subjects unique")

# Print first 5 iris subject IDs
print(f"\nFirst 5 iris IDs: {iris_subjects[:5]}")
