import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# STEP 4 - VISUALIZE FEATURES
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

PLOTS_FOLDER = PROJECT_ROOT / "plots"


# ------------------------------------------------------------
# 2. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)


# ------------------------------------------------------------
# 3. PARKING SPACES VS PRICE
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["parking_spaces"],
    df["price_usd"],
    alpha=0.7,
)

plt.title(
    "Parking Spaces vs House Price"
)

plt.xlabel(
    "Parking Spaces"
)

plt.ylabel(
    "Price (USD)"
)

plt.grid(
    alpha=0.3
)

parking_plot = (
    PLOTS_FOLDER
    / "step4_parking_vs_price.png"
)

plt.tight_layout()

plt.savefig(
    parking_plot
)

plt.show()


# ------------------------------------------------------------
# 4. DISTANCE FROM CITY VS PRICE
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["distance_city_km"],
    df["price_usd"],
    alpha=0.7,
)

plt.title(
    "Distance from City vs House Price"
)

plt.xlabel(
    "Distance from City (km)"
)

plt.ylabel(
    "Price (USD)"
)

plt.grid(
    alpha=0.3
)

distance_plot = (
    PLOTS_FOLDER
    / "step4_distance_vs_price.png"
)

plt.tight_layout()

plt.savefig(
    distance_plot
)

plt.show()


# ------------------------------------------------------------
# 5. PRINT OUTPUT LOCATIONS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PLOTS SAVED")
print("=" * 70)

print(
    parking_plot
)

print(
    distance_plot
)


# ------------------------------------------------------------
# 6. SIMPLE INTERPRETATION GUIDE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("WHAT TO LOOK FOR")
print("=" * 70)

print(
    """
Parking Spaces vs Price:
- Look for whether prices generally increase as parking spaces increase.

Distance vs Price:
- Look for whether prices generally decrease as distance from the city increases.

Remember:
A scatter plot helps us visually inspect relationships before training the model.
"""
)


# ------------------------------------------------------------
# 7. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 4 COMPLETED SUCCESSFULLY")
print("=" * 70)
