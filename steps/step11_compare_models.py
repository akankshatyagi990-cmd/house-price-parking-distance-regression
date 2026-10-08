import pandas as pd
from pathlib import Path


# ============================================================
# STEP 11 - COMPARE ALL REGRESSION MODELS
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

CURRENT_PROJECT = Path(__file__).resolve().parents[1]

DESKTOP = CURRENT_PROJECT.parent


# Model 1:
# Area + Bedrooms

AREA_BEDROOMS_FILE = (
    DESKTOP
    / "house-price-multiple-linear-regression"
    / "outputs"
    / "step10_metrics.csv"
)


# Model 2:
# Bathrooms + Age

BATHROOMS_AGE_FILE = (
    DESKTOP
    / "house-price-bathrooms-age-regression"
    / "outputs"
    / "step10_metrics.csv"
)


# Model 3:
# Parking + Distance

PARKING_DISTANCE_FILE = (
    CURRENT_PROJECT
    / "outputs"
    / "step10_metrics.csv"
)


OUTPUT_FILE = (
    CURRENT_PROJECT
    / "outputs"
    / "step11_model_comparison.csv"
)


# ------------------------------------------------------------
# 2. LOAD METRICS FILES
# ------------------------------------------------------------

area_bedrooms_df = pd.read_csv(
    AREA_BEDROOMS_FILE
)

bathrooms_age_df = pd.read_csv(
    BATHROOMS_AGE_FILE
)

parking_distance_df = pd.read_csv(
    PARKING_DISTANCE_FILE
)


# ------------------------------------------------------------
# 3. CONVERT EACH FILE TO DICTIONARY
# ------------------------------------------------------------

area_bedrooms = dict(
    zip(
        area_bedrooms_df["metric"],
        area_bedrooms_df["value"],
    )
)


bathrooms_age = dict(
    zip(
        bathrooms_age_df["metric"],
        bathrooms_age_df["value"],
    )
)


parking_distance = dict(
    zip(
        parking_distance_df["metric"],
        parking_distance_df["value"],
    )
)


# ------------------------------------------------------------
# 4. CREATE COMPARISON TABLE
# ------------------------------------------------------------

comparison = pd.DataFrame(
    {
        "metric": [
            "MAE",
            "RMSE",
            "R_squared",
        ],

        "area_plus_bedrooms": [
            area_bedrooms["MAE"],
            area_bedrooms["RMSE"],
            area_bedrooms["R_squared"],
        ],

        "bathrooms_plus_age": [
            bathrooms_age["MAE"],
            bathrooms_age["RMSE"],
            bathrooms_age["R_squared"],
        ],

        "parking_plus_distance": [
            parking_distance["MAE"],
            parking_distance["RMSE"],
            parking_distance["R_squared"],
        ],
    }
)


# ------------------------------------------------------------
# 5. DISPLAY COMPARISON TABLE
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("MODEL COMPARISON TABLE")
print("=" * 90)

print(
    comparison
)


# ------------------------------------------------------------
# 6. GET INDIVIDUAL METRICS
# ------------------------------------------------------------

model1_mae = area_bedrooms["MAE"]
model1_rmse = area_bedrooms["RMSE"]
model1_r2 = area_bedrooms["R_squared"]


model2_mae = bathrooms_age["MAE"]
model2_rmse = bathrooms_age["RMSE"]
model2_r2 = bathrooms_age["R_squared"]


model3_mae = parking_distance["MAE"]
model3_rmse = parking_distance["RMSE"]
model3_r2 = parking_distance["R_squared"]


# ------------------------------------------------------------
# 7. DISPLAY EACH MODEL
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("MODEL 1 - AREA + BEDROOMS")
print("=" * 90)

print(
    f"MAE  = ${model1_mae:,.2f}"
)

print(
    f"RMSE = ${model1_rmse:,.2f}"
)

print(
    f"R²   = {model1_r2:.6f}"
)


print("\n" + "=" * 90)
print("MODEL 2 - BATHROOMS + AGE")
print("=" * 90)

print(
    f"MAE  = ${model2_mae:,.2f}"
)

print(
    f"RMSE = ${model2_rmse:,.2f}"
)

print(
    f"R²   = {model2_r2:.6f}"
)


print("\n" + "=" * 90)
print("MODEL 3 - PARKING + DISTANCE")
print("=" * 90)

print(
    f"MAE  = ${model3_mae:,.2f}"
)

print(
    f"RMSE = ${model3_rmse:,.2f}"
)

print(
    f"R²   = {model3_r2:.6f}"
)


# ------------------------------------------------------------
# 8. FIND BEST MAE
# ------------------------------------------------------------

mae_scores = {
    "Area + Bedrooms": model1_mae,
    "Bathrooms + Age": model2_mae,
    "Parking + Distance": model3_mae,
}

best_mae_model = min(
    mae_scores,
    key=mae_scores.get,
)


print("\n" + "=" * 90)
print("BEST MAE")
print("=" * 90)

print(
    f"{best_mae_model} has the lowest MAE."
)


# ------------------------------------------------------------
# 9. FIND BEST RMSE
# ------------------------------------------------------------

rmse_scores = {
    "Area + Bedrooms": model1_rmse,
    "Bathrooms + Age": model2_rmse,
    "Parking + Distance": model3_rmse,
}

best_rmse_model = min(
    rmse_scores,
    key=rmse_scores.get,
)


print("\n" + "=" * 90)
print("BEST RMSE")
print("=" * 90)

print(
    f"{best_rmse_model} has the lowest RMSE."
)


# ------------------------------------------------------------
# 10. FIND BEST R-SQUARED
# ------------------------------------------------------------

r2_scores = {
    "Area + Bedrooms": model1_r2,
    "Bathrooms + Age": model2_r2,
    "Parking + Distance": model3_r2,
}

best_r2_model = max(
    r2_scores,
    key=r2_scores.get,
)


print("\n" + "=" * 90)
print("BEST R-SQUARED")
print("=" * 90)

print(
    f"{best_r2_model} has the highest R²."
)


# ------------------------------------------------------------
# 11. EXPLAIN NEGATIVE R-SQUARED
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("NEGATIVE R-SQUARED CHECK")
print("=" * 90)


for model_name, score in r2_scores.items():

    if score < 0:

        print(
            f"{model_name} has a negative R² of {score:.6f}."
        )

        print(
            "This means it performs worse than predicting "
            "the average house price for every observation."
        )


# ------------------------------------------------------------
# 12. FINAL RANKING BY R-SQUARED
# ------------------------------------------------------------

ranking = sorted(
    r2_scores.items(),
    key=lambda item: item[1],
    reverse=True,
)


print("\n" + "=" * 90)
print("FINAL RANKING")
print("=" * 90)


for position, (model_name, score) in enumerate(
    ranking,
    start=1,
):

    print(
        f"{position}. {model_name} -> R² = {score:.6f}"
    )


# ------------------------------------------------------------
# 13. SAVE COMPARISON TABLE
# ------------------------------------------------------------

comparison.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("\n" + "=" * 90)
print("COMPARISON FILE SAVED")
print("=" * 90)

print(
    OUTPUT_FILE
)


# ------------------------------------------------------------
# 14. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("STEP 11 COMPLETED SUCCESSFULLY")
print("=" * 90)

