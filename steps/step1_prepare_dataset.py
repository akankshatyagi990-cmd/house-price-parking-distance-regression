import pandas as pd
from pathlib import Path


# ============================================================
# STEP 1 - PREPARE DATASET
# Features:
#   X1 = parking_spaces
#   X2 = distance_city_km
# Target:
#   y = price_usd
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "house_price_linear_regression_practice.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "house_price_parking_distance.csv"
)


# ------------------------------------------------------------
# 2. LOAD ORIGINAL DATASET
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)


print("\n" + "=" * 70)
print("ORIGINAL DATASET LOADED")
print("=" * 70)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ------------------------------------------------------------
# 3. DEFINE REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "house_id",
    "parking_spaces",
    "distance_city_km",
    "price_usd",
]


# ------------------------------------------------------------
# 4. CHECK THAT REQUIRED COLUMNS EXIST
# ------------------------------------------------------------

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


print("\nRequired columns found successfully.")


# ------------------------------------------------------------
# 5. SELECT ONLY THE COLUMNS WE NEED
# ------------------------------------------------------------

prepared_df = df[
    required_columns
].copy()


# ------------------------------------------------------------
# 6. DISPLAY FIRST FEW ROWS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)

print(
    prepared_df.head()
)


# ------------------------------------------------------------
# 7. CHECK DATA TYPES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(
    prepared_df.dtypes
)


# ------------------------------------------------------------
# 8. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(
    prepared_df.isnull().sum()
)


# ------------------------------------------------------------
# 9. CHECK DUPLICATE ROWS
# ------------------------------------------------------------

duplicate_count = prepared_df.duplicated().sum()


print("\n" + "=" * 70)
print("DUPLICATE ROWS")
print("=" * 70)

print(
    f"Duplicate rows: {duplicate_count}"
)


# ------------------------------------------------------------
# 10. SAVE PREPARED DATASET
# ------------------------------------------------------------

prepared_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("\n" + "=" * 70)
print("PREPARED DATASET SAVED")
print("=" * 70)

print(
    OUTPUT_FILE
)


# ------------------------------------------------------------
# 11. SHOW FINAL DATASET SHAPE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print(
    f"Rows    : {prepared_df.shape[0]}"
)

print(
    f"Columns : {prepared_df.shape[1]}"
)

print(
    "\nSelected columns:"
)

for column in prepared_df.columns:
    print(f"- {column}")


print("\nStep 1 completed successfully.")