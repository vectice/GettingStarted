# import the Vectice provided probability of default tests
from vectice.models.test_library.probability_of_default_test import (
    plot_roc_curve,
    conf_matrix,
    explainability,
    feature_importance,
    label_drift,
    prediction_drift,
)

# import the Vectice provided regression tests
from vectice.models.test_library.regression_test import (
    plot_residuals,
    r2_score,
    explainability,
    feature_importance,
    target_drift,
    prediction_drift,
)

# import the Vectice provided time series tests
from vectice.models.test_library.time_series_test import (
    trend_analysis,
    seasonality_check,
    autocorrelation_test,
    stationarity_test,
    missing_value_analysis,
)


# custom data quality tests
from data_quality_tests import (
    test_dataset_split,
    iqr_and_outliers,
)


# Map the tests to be used for regression
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

# Map the tests to be used for time series
TIME_SERIES_FULL_SUITE_MAP_TEST = {
    "trend": trend_analysis,
    "seasonality": seasonality_check,
    "autocorrelation": autocorrelation_test,
    "stationarity": stationarity_test,
    "missing_value": missing_value_analysis,
    "time_series_full_suite": [
        trend_analysis,
        seasonality_check,
        autocorrelation_test,
        stationarity_test,
        missing_value_analysis,
    ],
}


# The master test suite file is used to map all tests which can be run.
# The tests can be provided by Vectice or custom functions from your test suite modules.
# Vectice uses this configuration to simply identify available tests, when you run
# your validations in your notebook.

# Accumulation and mapping of all tests to be run
MASTER_SUITE_MAP_TEST = {
    "probability_of_default_validation": [
        plot_roc_curve,
        conf_matrix,
        explainability,
        feature_importance,
        label_drift,
        prediction_drift,
    ],
    "data_quality": [
        test_dataset_split,
        iqr_and_outliers,
    ],
}
