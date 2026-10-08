import pandas as pd
from pathlib import Path


# ============================================================
# STEP 2 - EXPLORE DATASET
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "house_price_parking_distance.csv"
)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)


# ------------------------------------------------------------
# 3. FIRST 5 ROWS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)

print(
    df.head()
)


# ------------------------------------------------------------
# 4. LAST 5 ROWS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("LAST 5 ROWS")
print("=" * 70)

print(
    df.tail()
)


# ------------------------------------------------------------
# 5. DATASET SHAPE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET SHAPE")
print("=" * 70)

print(
    f"Rows    : {df.shape[0]}"
)

print(
    f"Columns : {df.shape[1]}"
)


# ------------------------------------------------------------
# 6. COLUMN NAMES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("COLUMN NAMES")
print("=" * 70)

for column in df.columns:
    print(column)


# ------------------------------------------------------------
# 7. DATA TYPES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATA TYPES")
print("=" * 70)

print(
    df.dtypes
)


# ------------------------------------------------------------
# 8. MISSING VALUES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

print(
    df.isnull().sum()
)


# ------------------------------------------------------------
# 9. DUPLICATE ROWS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATE ROWS")
print("=" * 70)

print(
    df.duplicated().sum()
)


# ------------------------------------------------------------
# 10. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS")
print("=" * 70)

print(
    df.describe()
)


# ------------------------------------------------------------
# 11. PARKING SPACE DISTRIBUTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PARKING SPACE DISTRIBUTION")
print("=" * 70)

print(
    df["parking_spaces"]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# 12. DISTANCE STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DISTANCE FROM CITY STATISTICS")
print("=" * 70)

print(
    f"Minimum distance : {df['distance_city_km'].min():.2f} km"
)

print(
    f"Maximum distance : {df['distance_city_km'].max():.2f} km"
)

print(
    f"Average distance : {df['distance_city_km'].mean():.2f} km"
)


# ------------------------------------------------------------
# 13. PRICE STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("HOUSE PRICE STATISTICS")
print("=" * 70)

print(
    f"Minimum price : ${df['price_usd'].min():,.2f}"
)

print(
    f"Maximum price : ${df['price_usd'].max():,.2f}"
)

print(
    f"Average price : ${df['price_usd'].mean():,.2f}"
)


# ------------------------------------------------------------
# 14. DISPLAY RELEVANT COLUMNS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FIRST 10 RELEVANT ROWS")
print("=" * 70)

print(
    df[
        [
            "parking_spaces",
            "distance_city_km",
            "price_usd",
        ]
    ].head(10)
)


# ------------------------------------------------------------
# 15. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 2 COMPLETED SUCCESSFULLY")
print("=" * 70)