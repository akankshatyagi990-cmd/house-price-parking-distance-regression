import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# STEP 9 - TRAIN / TEST SPLIT
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

TRAIN_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "step9_train_data.csv"
)

TEST_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "step9_test_data.csv"
)


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)


print("\n" + "=" * 70)
print("DATASET LOADED")
print("=" * 70)

print(
    f"Total rows: {len(df)}"
)


# ------------------------------------------------------------
# 3. PREPARE FEATURES AND TARGET
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
# 4. CREATE RANDOM INDICES
# ------------------------------------------------------------

np.random.seed(42)

indices = np.arange(
    len(df)
)

np.random.shuffle(
    indices
)


# ------------------------------------------------------------
# 5. CREATE 80% TRAIN / 20% TEST SPLIT
# ------------------------------------------------------------

train_ratio = 0.80

train_size = int(
    len(df) * train_ratio
)

train_indices = indices[
    :train_size
]

test_indices = indices[
    train_size:
]


# ------------------------------------------------------------
# 6. CREATE TRAINING DATA
# ------------------------------------------------------------

X_train = X[
    train_indices
]

y_train = y[
    train_indices
]


# ------------------------------------------------------------
# 7. CREATE TEST DATA
# ------------------------------------------------------------

X_test = X[
    test_indices
]

y_test = y[
    test_indices
]


# ------------------------------------------------------------
# 8. DISPLAY SPLIT INFORMATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(
    f"Training rows : {len(X_train)}"
)

print(
    f"Testing rows  : {len(X_test)}"
)


# ------------------------------------------------------------
# 9. ADD INTERCEPT TO TRAINING DATA
# ------------------------------------------------------------

train_ones = np.ones(
    (X_train.shape[0], 1)
)

X_train_design = np.hstack(
    (
        train_ones,
        X_train,
    )
)


# ------------------------------------------------------------
# 10. TRAIN MODEL USING ONLY TRAINING DATA
# ------------------------------------------------------------

beta = np.linalg.lstsq(
    X_train_design,
    y_train,
    rcond=None,
)[0]


intercept = beta[0]

parking_coefficient = beta[1]

distance_coefficient = beta[2]


# ------------------------------------------------------------
# 11. DISPLAY TRAINED COEFFICIENTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINED MODEL COEFFICIENTS")
print("=" * 70)

print(
    f"Intercept            : ${intercept:,.2f}"
)

print(
    f"Parking coefficient  : ${parking_coefficient:,.2f}"
)

print(
    f"Distance coefficient : ${distance_coefficient:,.2f}"
)


# ------------------------------------------------------------
# 12. MAKE PREDICTIONS ON TEST DATA
# ------------------------------------------------------------

test_predictions = (
    intercept
    + parking_coefficient * X_test[:, 0]
    + distance_coefficient * X_test[:, 1]
)


# ------------------------------------------------------------
# 13. CREATE TRAIN DATAFRAME
# ------------------------------------------------------------

train_df = pd.DataFrame(
    {
        "parking_spaces": X_train[:, 0],
        "distance_city_km": X_train[:, 1],
        "price_usd": y_train,
    }
)


# ------------------------------------------------------------
# 14. CREATE TEST DATAFRAME
# ------------------------------------------------------------

test_df = pd.DataFrame(
    {
        "parking_spaces": X_test[:, 0],
        "distance_city_km": X_test[:, 1],
        "actual_price_usd": y_test,
        "predicted_price_usd": test_predictions,
    }
)


# ------------------------------------------------------------
# 15. DISPLAY FIRST TEST PREDICTIONS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FIRST 10 TEST PREDICTIONS")
print("=" * 70)

print(
    test_df.head(10)
)


# ------------------------------------------------------------
# 16. SAVE TRAINING DATA
# ------------------------------------------------------------

train_df.to_csv(
    TRAIN_OUTPUT,
    index=False,
)


# ------------------------------------------------------------
# 17. SAVE TEST DATA
# ------------------------------------------------------------

test_df.to_csv(
    TEST_OUTPUT,
    index=False,
)


print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(
    TRAIN_OUTPUT
)

print(
    TEST_OUTPUT
)


# ------------------------------------------------------------
# 18. IMPORTANT EXPLANATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("WHY DO WE SPLIT THE DATA?")
print("=" * 70)

print(
    """
Training data:
Used to calculate the regression coefficients.

Testing data:
Not used while training the model.

The testing data helps us check how well the model
performs on houses it did not use during training.

This gives us a more realistic evaluation.
"""
)


# ------------------------------------------------------------
# 19. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 9 COMPLETED SUCCESSFULLY")
print("=" * 70)