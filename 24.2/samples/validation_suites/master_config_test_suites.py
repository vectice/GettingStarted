# import the Vectice provided probability of default validation tests
from vectice.models.test_library.binary_classification_test import (
    plot_roc_curve,
    conf_matrix,
    explainability,
    feature_importance,
    label_drift,
    prediction_drift,
)


# custom data quality validation tests
from data_quality_modules import (
    test_dataset_split,
    iqr_and_outliers,
)

# custom data privacy validation tests
from data_privacy_modules import (
    sensitive_data_check,
    sensitive_data_type_check,
    pii_check,
)

from correlation_matrix_module import (
    plot_correlation_matrix
)


# The master test suite file is used to map all suite of test which can be run.
# The tests can be provided by Vectice or custom functions from your modules.
# Vectice uses this configuration to simply identify and bundle available tests into suite, when you run
# your validations in your notebook.

# Accumulation and mapping of all validation tests to be run for the PD model suite
PD_model_suite= {
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

# Map the tests to be used for data privacy validation
Robustness_suite = {
    "sensitive_data_check": sensitive_data_check,
    "pii_check": pii_check,
    "sensitive_data_type_check": sensitive_data_type_check,
    "data_privacy_full_suite": [
        sensitive_data_check,
        pii_check,
        sensitive_data_type_check,
    ],
}
