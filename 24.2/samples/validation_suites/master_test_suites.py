# Vectice provided model validation tests
from binary_classification_full_suite import BINARY_CLASSIFICATION_FULL_SUITE_MAP_TEST

# custom data quality tests
from data_quality_full_suite import (
    test_dataset_split,
    iqr_and_outliers,
)


# The master test suite file is used to map all tests which can be run.
# The tests can be provided by Vectice or custom functions from your test suite modules.
# Vectice uses this configuration to simply identify available tests, when you run
# your validations in your notebook.

# Accumulation and mapping of all tests to be run
MASTER_FULL_SUITE_MAP_TEST = {
    "binary_full_suite": BINARY_CLASSIFICATION_FULL_SUITE_MAP_TEST["binary_full_suite"],
    "full_dataset_validation": [
        test_dataset_split,
        iqr_and_outliers,
    ],
}
