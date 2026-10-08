import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# STEP 7 - MAKE MANUAL PREDICTIONS
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "house_price_parking_distance.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step7_manual_predictions.csv"
)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)


# ------------------------------------------------------------
# 3. PREPARE X AND y
# ------------------------------------------------------------

X = df[
    [
        "parking_spaces",
        "distance_city_km",
    ]
].to_numpy(dtype=float)

y = df[
    "price_usd"
].to_numpy(dtype=float)


# ------------------------------------------------------------
# 4. ADD INTERCEPT COLUMN
# ------------------------------------------------------------

ones = np.ones(
    (X.shape[0], 1)
)

X_design = np.hstack(
    (
        ones,
        X,
    )
)


# ------------------------------------------------------------
# 5. CALCULATE COEFFICIENTS
# ------------------------------------------------------------

beta = np.linalg.lstsq(
    X_design,
    y,
    rcond=None,
)[0]


intercept = beta[0]

parking_coefficient = beta[1]

distance_coefficient = beta[2]


# ------------------------------------------------------------
# 6. DISPLAY COEFFICIENTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MODEL COEFFICIENTS")
print("=" * 70)


print(
    f"Intercept               : ${intercept:,.2f}"
)

print(
    f"Parking coefficient     : ${parking_coefficient:,.2f}"
)

print(
    f"Distance coefficient    : ${distance_coefficient:,.2f}"
)


# ------------------------------------------------------------
# 7. MANUALLY CALCULATE PREDICTIONS
# ------------------------------------------------------------

predicted_prices = (
    intercept
    + parking_coefficient * df["parking_spaces"]
    + distance_coefficient * df["distance_city_km"]
)


# ------------------------------------------------------------
# 8. CREATE RESULTS DATAFRAME
# ------------------------------------------------------------

results = pd.DataFrame(
    {
        "house_id": df["house_id"],
        "parking_spaces": df["parking_spaces"],
        "distance_city_km": df["distance_city_km"],
        "actual_price_usd": df["price_usd"],
        "predicted_price_usd": predicted_prices,
    }
)


# ------------------------------------------------------------
# 9. DISPLAY FIRST 10 PREDICTIONS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FIRST 10 PREDICTIONS")
print("=" * 70)


print(
    results.head(10)
)


# ------------------------------------------------------------
# 10. SAVE PREDICTIONS
# ------------------------------------------------------------

results.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("\n" + "=" * 70)
print("PREDICTIONS SAVED")
print("=" * 70)


print(
    OUTPUT_FILE
)


# ------------------------------------------------------------
# 11. CUSTOM HOUSE PREDICTIONS
# ------------------------------------------------------------

custom_houses = [
    (0, 5),
    (1, 5),
    (1, 15),
    (2, 10),
    (2, 25),
    (3, 10),
]


print("\n" + "=" * 70)
print("CUSTOM HOUSE PREDICTIONS")
print("=" * 70)


for parking, distance in custom_houses:

    prediction = (
        intercept
        + parking_coefficient * parking
        + distance_coefficient * distance
    )

    print(
        f"Parking = {parking}, "
        f"Distance = {distance} km "
        f"-> Predicted Price = ${prediction:,.2f}"
    )


# ------------------------------------------------------------
# 12. EXPLAIN WHAT WE DID
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("WHAT HAPPENED?")
print("=" * 70)


print(
    """
For every house, we used the equation:

Predicted Price
=
Intercept
+ (Parking Coefficient × Parking Spaces)
+ (Distance Coefficient × Distance from City)

This gives us one predicted house price for every row.
"""
)


# ------------------------------------------------------------
# 13. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 7 COMPLETED SUCCESSFULLY")
print("=" * 70)
