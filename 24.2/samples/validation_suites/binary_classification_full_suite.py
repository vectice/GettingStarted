# import the Vectice provided binary classification tests
from vectice.models.test_library.binary_classification_test import (
    plot_roc_curve,
    conf_matrix,
    explainability,
    feature_importance,
    label_drift,
    prediction_drift,
)

# Map the tests to be used
BINARY_CLASSIFICATION_FULL_SUITE_MAP_TEST = {
    "roc": plot_roc_curve,
    "cm": conf_matrix,
    "explainability": explainability,
    "feature_importance": feature_importance,
    "drift": [label_drift, prediction_drift],
    "binary_full_suite": [
        plot_roc_curve,
        conf_matrix,
        explainability,
        feature_importance,
        label_drift,
        prediction_drift,
    ],
}
