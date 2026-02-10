import csv
from math import sqrt
from utils import subtext, success, warning

RAW_CSV = "sounds_raw.csv"
SORTED_CSV = "sounds_sorted.csv"
SIMILARITY_THRESHOLD = 0.15  # lower = stricter match

# ----------------------------
# HELPER: Euclidean distance
# ----------------------------
def distance(a, b):
    # Compare numerical features: peak, duration, energy, sharpness
    return sqrt(
        (float(a["peak"]) - float(b["peak"]))**2 +
        (float(a["duration"]) - float(b["duration"]))**2 +
        (float(a["energy"]) - float(b["energy"]))**2 +
        (float(a["sharpness"]) - float(b["sharpness"]))**2
    )

# ----------------------------
# LOAD SORTED CSV (known sounds)
# ----------------------------
known_sounds = []
try:
    with open(SORTED_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("label") and row["label"].lower() != "unknown":
                known_sounds.append(row)
except FileNotFoundError:
    subtext(f"No existing {SORTED_CSV} found. It will be created.")

# ----------------------------
# LOAD RAW CSV
# ----------------------------
raw_sounds = []
with open(RAW_CSV, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        raw_sounds.append(row)

if not raw_sounds:
    warning(f"No sounds in {RAW_CSV}. Exiting.")
    exit(0)

# ----------------------------
# AUTO LABELING
# ----------------------------
labeled_sounds = []

for row in raw_sounds:
    if row.get("label") and row["label"].lower() != "unknown":
        labeled_sounds.append(row)
        continue

    # Attempt to match with known sounds
    assigned_label = "unknown"
    for known in known_sounds:
        dist = distance(row, known)
        # Normalize distance by peak to account for scale
        if dist / (float(known["peak"]) + 1) < SIMILARITY_THRESHOLD:
            assigned_label = known["label"]
            break

    row["label"] = assigned_label
    labeled_sounds.append(row)

# ----------------------------
# APPEND to known_sounds for future auto-labeling
# ----------------------------
known_sounds.extend([s for s in labeled_sounds if s["label"] != "unknown"])

# ----------------------------
# SAVE SORTED CSV
# ----------------------------
with open(SORTED_CSV, "w", newline="") as f:
    fieldnames = ["time", "peak", "duration", "energy", "sharpness", "label"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in labeled_sounds:
        writer.writerow(row)

subtext(f"Auto-labeled sounds saved to {SORTED_CSV}")

# ----------------------------
# SHOW UNKNOWN FOR MANUAL LABELING
# ----------------------------
for row in labeled_sounds:
    if row["label"] == "unknown":
        print(f"\nSound at {float(row['time']):.3f}s | peak={row['peak']}, energy={row['energy']}, sharpness={row['sharpness']}")
        label = input("Enter label for this new sound: ").strip()
        if label:
            row["label"] = label
            success(f"Labeled as: {label}")

# ----------------------------
# SAVE UPDATED CSV
# ----------------------------
with open(SORTED_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in labeled_sounds:
        writer.writerow(row)

subtext(f"All sounds updated and saved to {SORTED_CSV}")

