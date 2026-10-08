import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# STEP 6 - CALCULATE ACTUAL REGRESSION COEFFICIENTS
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
# 3. CREATE X AND y
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


print("\n" + "=" * 70)
print("DATA PREPARED")
print("=" * 70)

print(
    f"X shape: {X.shape}"
)

print(
    f"y shape: {y.shape}"
)


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


print("\n" + "=" * 70)
print("DESIGN MATRIX")
print("=" * 70)

print(
    "First 5 rows:"
)

print(
    X_design[:5]
)


# ------------------------------------------------------------
# 5. CALCULATE COEFFICIENTS
# ------------------------------------------------------------

beta = np.linalg.lstsq(
    X_design,
    y,
    rcond=None,
)[0]


# ------------------------------------------------------------
# 6. EXTRACT COEFFICIENTS
# ------------------------------------------------------------

intercept = beta[0]

parking_coefficient = beta[1]

distance_coefficient = beta[2]


# ------------------------------------------------------------
# 7. DISPLAY COEFFICIENTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CALCULATED COEFFICIENTS")
print("=" * 70)


print(
    f"Intercept:"
    f" ${intercept:,.2f}"
)

print(
    f"Parking coefficient:"
    f" ${parking_coefficient:,.2f}"
)

print(
    f"Distance coefficient:"
    f" ${distance_coefficient:,.2f}"
)


# ------------------------------------------------------------
# 8. DISPLAY FINAL EQUATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL REGRESSION EQUATION")
print("=" * 70)


print(
    f"""
Predicted Price
=
{intercept:,.2f}
+ ({parking_coefficient:,.2f} × Parking Spaces)
+ ({distance_coefficient:,.2f} × Distance from City)
"""
)


# ------------------------------------------------------------
# 9. INTERPRET PARKING COEFFICIENT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PARKING COEFFICIENT INTERPRETATION")
print("=" * 70)


if parking_coefficient > 0:

    print(
        f"For every additional parking space, "
        f"the predicted house price increases by "
        f"approximately ${parking_coefficient:,.2f}, "
        f"while distance is held constant."
    )

else:

    print(
        f"For every additional parking space, "
        f"the predicted house price decreases by "
        f"approximately ${abs(parking_coefficient):,.2f}, "
        f"while distance is held constant."
    )


# ------------------------------------------------------------
# 10. INTERPRET DISTANCE COEFFICIENT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DISTANCE COEFFICIENT INTERPRETATION")
print("=" * 70)


if distance_coefficient > 0:

    print(
        f"For every additional kilometer away from the city, "
        f"the predicted house price increases by "
        f"approximately ${distance_coefficient:,.2f}, "
        f"while parking spaces are held constant."
    )

else:

    print(
        f"For every additional kilometer away from the city, "
        f"the predicted house price decreases by "
        f"approximately ${abs(distance_coefficient):,.2f}, "
        f"while parking spaces are held constant."
    )


# ------------------------------------------------------------
# 11. EXAMPLE PREDICTION
# ------------------------------------------------------------

example_parking = 2

example_distance = 10


example_prediction = (
    intercept
    + parking_coefficient * example_parking
    + distance_coefficient * example_distance
)


print("\n" + "=" * 70)
print("EXAMPLE PREDICTION")
print("=" * 70)


print(
    f"Parking Spaces     : {example_parking}"
)

print(
    f"Distance from City : {example_distance} km"
)

print(
    f"Predicted Price    : ${example_prediction:,.2f}"
)


# ------------------------------------------------------------
# 12. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 6 COMPLETED SUCCESSFULLY")
print("=" * 70)