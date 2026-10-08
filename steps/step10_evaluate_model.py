from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
X_TRAIN_PATH = PROJECT_ROOT / "outputs" / "X_train.csv"
X_TEST_PATH = PROJECT_ROOT / "outputs" / "X_test.csv"
Y_TRAIN_PATH = PROJECT_ROOT / "outputs" / "y_train.csv"
Y_TEST_PATH = PROJECT_ROOT / "outputs" / "y_test.csv"
COEFFICIENTS_PATH = PROJECT_ROOT / "outputs" / "coefficients.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def load_split_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict[str, float], float]:
    if not X_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training features not found at {X_TRAIN_PATH}. Run step9_train_test_split.py first.")
    if not X_TEST_PATH.exists():
        raise FileNotFoundError(f"Testing features not found at {X_TEST_PATH}. Run step9_train_test_split.py first.")
    if not Y_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Training target not found at {Y_TRAIN_PATH}. Run step9_train_test_split.py first.")
    if not Y_TEST_PATH.exists():
        raise FileNotFoundError(f"Testing target not found at {Y_TEST_PATH}. Run step9_train_test_split.py first.")
    if not COEFFICIENTS_PATH.exists():
        raise FileNotFoundError(
            f"Coefficient file not found at {COEFFICIENTS_PATH}. Run step6_calculate_coefficients.py first."
        )

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).iloc[:, 0]
    y_test = pd.read_csv(Y_TEST_PATH).iloc[:, 0]

    coefficients_df = pd.read_csv(COEFFICIENTS_PATH)
    intercept = float(coefficients_df[coefficients_df["feature"] == "intercept"].iloc[0]["coefficient"])
    slopes = {
        row["feature"]: float(row["coefficient"])
        for _, row in coefficients_df[coefficients_df["feature"] != "intercept"].iterrows()
    }

    return X_train, X_test, y_train, y_test, slopes, intercept


def predict_with_coefficients(X: pd.DataFrame, slopes: dict[str, float], intercept: float) -> pd.Series:
    predictions = pd.Series(intercept, index=X.index, dtype=float)
    for column, coefficient in slopes.items():
        if column not in X.columns:
            raise ValueError(f"Coefficient for '{column}' does not match a feature column in X.")
        predictions = predictions + X[column] * coefficient
    return predictions


def evaluate_model(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    residuals = y_true - y_pred
    mae = float((residuals.abs()).mean())
    rmse = float((residuals ** 2).mean() ** 0.5)
    r2 = float(1 - (residuals ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum())

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


def save_evaluation_summary(test_metrics: dict[str, float]) -> None:
    summary_df = pd.DataFrame(
        {
            "metric": ["mae", "rmse", "r2"],
            "value": [test_metrics["mae"], test_metrics["rmse"], test_metrics["r2"]],
        }
    )
    summary_path = OUTPUTS_DIR / "model_evaluation.csv"
    summary_df.to_csv(summary_path, index=False)
    print(summary_df.to_string(index=False))
    print(f"\nSaved model evaluation to: {summary_path}")


def main() -> None:
    X_train, X_test, y_train, y_test, slopes, intercept = load_split_data()

    y_pred = predict_with_coefficients(X_test, slopes, intercept)
    test_metrics = evaluate_model(y_test.reset_index(drop=True), y_pred.reset_index(drop=True))

    print("Train set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    print("\nTest evaluation metrics:")
    save_evaluation_summary(test_metrics)
    print("\nModel evaluation complete.")


if __name__ == "__main__":
    main()
