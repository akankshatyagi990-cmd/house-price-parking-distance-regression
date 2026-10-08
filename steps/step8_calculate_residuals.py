import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# STEP 8 - CALCULATE RESIDUALS
# ============================================================


# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step7_manual_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "step8_residuals.csv"
)

PLOT_FILE = (
    PROJECT_ROOT
    / "plots"
    / "step8_residual_plot.png"
)


# ------------------------------------------------------------
# 2. LOAD PREDICTION DATA
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)


print("\n" + "=" * 70)
print("PREDICTION DATA LOADED")
print("=" * 70)

print(
    df.head()
)


# ------------------------------------------------------------
# 3. CALCULATE RESIDUALS
# ------------------------------------------------------------

df["residual"] = (
    df["actual_price_usd"]
    - df["predicted_price_usd"]
)


# ------------------------------------------------------------
# 4. DISPLAY FIRST 10 RESIDUALS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FIRST 10 RESIDUALS")
print("=" * 70)

print(
    df[
        [
            "actual_price_usd",
            "predicted_price_usd",
            "residual",
        ]
    ].head(10)
)


# ------------------------------------------------------------
# 5. EXPLAIN RESIDUAL SIGN
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("HOW TO READ RESIDUALS")
print("=" * 70)

print(
    """
Residual = Actual Price - Predicted Price

Positive residual:
Actual price is HIGHER than predicted price.

Negative residual:
Actual price is LOWER than predicted price.

Residual close to 0:
Prediction is close to the actual price.
"""
)


# ------------------------------------------------------------
# 6. RESIDUAL STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RESIDUAL STATISTICS")
print("=" * 70)

print(
    f"Minimum residual : ${df['residual'].min():,.2f}"
)

print(
    f"Maximum residual : ${df['residual'].max():,.2f}"
)

print(
    f"Average residual : ${df['residual'].mean():,.2f}"
)


# ------------------------------------------------------------
# 7. SAVE RESIDUAL DATA
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


print("\n" + "=" * 70)
print("RESIDUAL DATA SAVED")
print("=" * 70)

print(
    OUTPUT_FILE
)


# ------------------------------------------------------------
# 8. CREATE RESIDUAL PLOT
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 5)
)

plt.scatter(
    df["predicted_price_usd"],
    df["residual"],
    alpha=0.7,
)

plt.axhline(
    y=0,
    linestyle="--",
)

plt.title(
    "Residual Plot"
)

plt.xlabel(
    "Predicted Price (USD)"
)

plt.ylabel(
    "Residual"
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    PLOT_FILE
)

plt.show()


# ------------------------------------------------------------
# 9. DISPLAY PLOT LOCATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("RESIDUAL PLOT SAVED")
print("=" * 70)

print(
    PLOT_FILE
)


# ------------------------------------------------------------
# 10. WHAT TO LOOK FOR
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("WHAT TO LOOK FOR")
print("=" * 70)

print(
    """
A good residual plot usually has points scattered
randomly around the zero line.

If you see a strong pattern, curve, or structure,
it may mean that a simple linear relationship
does not fully explain the data.
"""
)


# ------------------------------------------------------------
# 11. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 8 COMPLETED SUCCESSFULLY")
print("=" * 70)