# import the Vectice provided probability of default validation tests
from vectice.models.test_library.binary_classification_test import (
    plot_roc_curve,
    conf_matrix,
    explainability,
    feature_importance,
    label_drift,
    prediction_drift,
)

# import the Vectice provided regression validation tests
from vectice.models.test_library.regression_test import (
    plot_residuals,
    r2_score,
    explainability,
    feature_importance,
    target_drift,
    prediction_drift,
)

# import the Vectice provided time series validation tests
from vectice.models.test_library.time_series_test import (
    trend_analysis,
    seasonality_check,
    autocorrelation_test,
    stationarity_test,
    missing_value_analysis,
)


# custom data quality validation tests
from data_quality_tests import (
    test_dataset_split,
    iqr_and_outliers,
)

# custom data privacy validation tests
from data_privacy_tests import (
    sensitive_data_check,
    sensitive_data_type_check,
    pii_check,
)

from custom_tests import (
    plot_correlation_matrix
)

# Map the tests to be used for regression validation
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

# Map the tests to be used for time series validation
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

# Map the tests to be used for data quality
DATA_QUALITY_SUITE_MAP_TEST = {
    "dataset_split": test_dataset_split,
    "iqr_and_outliers": iqr_and_outliers,
    "full_dataset_validation": [
        test_dataset_split,
        iqr_and_outliers,
    ],
}

# Map the tests to be used for data privacy validation
DATA_PRIVACY_SUITE_MAP_TEST = {
    "sensitive_data_check": sensitive_data_check,
    "pii_check": pii_check,
    "sensitive_data_type_check": sensitive_data_type_check,
    "data_privacy_full_suite": [
        sensitive_data_check,
        pii_check,
        sensitive_data_type_check,
    ],
}

# The master test suite file is used to map all tests which can be run.
# The tests can be provided by Vectice or custom functions from your test suite modules.
# Vectice uses this configuration to simply identify available tests, when you run
# your validations in your notebook.

# Accumulation and mapping of all validation tests to be run
CUSTOM_TEST_PD_MODEL = {
    "binary_suite": [
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
    "custom":[
        plot_correlation_matrix,
    ]
}
