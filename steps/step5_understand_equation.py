# ============================================================
# STEP 5 - UNDERSTAND THE REGRESSION EQUATION
# ============================================================


# ------------------------------------------------------------
# 1. EXPLAIN THE MODEL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MULTIPLE LINEAR REGRESSION EQUATION")
print("=" * 70)


print(
    """
Our model uses:

X1 = parking_spaces
X2 = distance_city_km
y  = price_usd

The equation is:

Predicted Price
=
b0
+ (b1 × Parking Spaces)
+ (b2 × Distance from City)

Where:

b0 = Intercept
b1 = Parking Spaces Coefficient
b2 = Distance Coefficient
"""
)


# ------------------------------------------------------------
# 2. USE EXAMPLE COEFFICIENTS
# ------------------------------------------------------------

intercept = 300000

parking_coefficient = 25000

distance_coefficient = -7000


print("\n" + "=" * 70)
print("EXAMPLE COEFFICIENTS")
print("=" * 70)


print(
    f"Intercept               = {intercept}"
)

print(
    f"Parking coefficient     = {parking_coefficient}"
)

print(
    f"Distance coefficient    = {distance_coefficient}"
)


# ------------------------------------------------------------
# 3. EXAMPLE HOUSE
# ------------------------------------------------------------

parking_spaces = 2

distance_city_km = 10


predicted_price = (
    intercept
    + parking_coefficient * parking_spaces
    + distance_coefficient * distance_city_km
)


print("\n" + "=" * 70)
print("EXAMPLE PREDICTION")
print("=" * 70)


print(
    f"Parking Spaces      : {parking_spaces}"
)

print(
    f"Distance from City  : {distance_city_km} km"
)

print(
    f"Predicted Price     : ${predicted_price:,.2f}"
)


# ------------------------------------------------------------
# 4. EXPLAIN EACH COEFFICIENT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("COEFFICIENT INTERPRETATION")
print("=" * 70)


print(
    """
Intercept:
The predicted house price when:

Parking Spaces = 0
Distance from City = 0

Parking Coefficient:
If the coefficient is positive, then adding one parking space
increases the predicted price, while keeping distance constant.

Distance Coefficient:
If the coefficient is negative, then moving one kilometer farther
from the city decreases the predicted price, while keeping parking
spaces constant.
"""
)


# ------------------------------------------------------------
# 5. IMPORTANT NOTE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("IMPORTANT")
print("=" * 70)


print(
    """
The coefficients used in this step are only examples.

They are NOT learned from the dataset yet.

In the next step, we will calculate the real coefficients
using the actual house-price data.
"""
)


# ------------------------------------------------------------
# 6. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 5 COMPLETED SUCCESSFULLY")
print("=" * 70)