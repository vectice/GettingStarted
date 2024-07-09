# import the Vectice provided test
from vectice.models.test_library.regression_test import (
    plot_residuals,
    r2_score,
    explainability,
    feature_importance,
    target_drift,
    prediction_drift,
)

# Map the tests to be used
REGRESSION_FULL_SUITE_MAP_TEST = {
    "roc": plot_residuals,
    "cm": r2_score,
    "explainability": explainability,
    "feature_importance": feature_importance,
    "drift": [target_drift, prediction_drift],
    "binary_full_suite": [
        plot_residuals,
        r2_score,
        explainability,
        feature_importance,
        target_drift,
        prediction_drift,
    ],
}
