# Write custom tests which can be used to validate your datasets security
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from pandas import DataFrame

    from vectice.models.validation_dataset import TestSuiteReturnType


def sensitive_data_check(
    dataset: DataFrame | None = None,
    training_df: DataFrame | None = None,
    testing_df: DataFrame | None = None,
    feature_columns: ArrayLike | list | None = None,
    target_column: ArrayLike | str | None = None,
    sensitive_keywords: list | None = None,
) -> TestSuiteReturnType | None:
    from vectice import Table
    from vectice.models.validation_dataset import TestSuiteReturnType

    if dataset is None or sensitive_keywords is None:
        return None

    # Initialize a dictionary to hold counts of sensitive data
    sensitive_counts = {keyword: 0 for keyword in sensitive_keywords}

    # Check each cell in the DataFrame for sensitive keywords
    for keyword in sensitive_keywords:
        sensitive_counts[keyword] = dataset.apply(
            lambda x: x.astype(str).str.contains(keyword, case=False).sum()
        ).sum()

    # Create a DataFrame with the results
    sensitive_counts_df = pd.DataFrame(
        {
            "Sensitive Keyword": list(sensitive_counts.keys()),
            "Count": list(sensitive_counts.values()),
        }
    )

    table = Table(sensitive_counts_df)

    return TestSuiteReturnType(
        properties={},
        tables=[table],
        attachments=[],
    )


def pii_check(
    dataset: DataFrame | None = None,
    training_df: DataFrame | None = None,
    testing_df: DataFrame | None = None,
    feature_columns: ArrayLike | list | None = None,
    target_column: ArrayLike | str | None = None,
) -> TestSuiteReturnType | None:
    from vectice import Table
    from vectice.models.validation_dataset import TestSuiteReturnType

    if dataset is None:
        return None

    # Define common PII patterns
    pii_patterns = {
        "name": r"\b[A-Z][a-z]*\b",
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",
        "phone": r"\b(\+?[\d]{1,3}[-.\s]?[\d]{1,4}[-.\s]?[\d]{1,4}[-.\s]?[\d]{1,9})\b",
    }

    # Initialize a dictionary to hold counts of PII matches
    pii_counts = {key: 0 for key in pii_patterns.keys()}

    # Check each column in the DataFrame for PII patterns
    for column in dataset.columns:
        for key, pattern in pii_patterns.items():
            pii_counts[key] += (
                dataset[column]
                .astype(str)
                .str.contains(pattern, case=False, regex=True)
                .sum()
            )

    # Create a DataFrame with the results
    pii_counts_df = pd.DataFrame(
        {"PII Type": list(pii_counts.keys()), "Count": list(pii_counts.values())}
    )

    table = Table(pii_counts_df)

    return TestSuiteReturnType(
        properties={},
        tables=[table],
        attachments=[],
    )


def sensitive_data_type_check(
    dataset: DataFrame | None = None,
    training_df: DataFrame | None = None,
    testing_df: DataFrame | None = None,
    feature_columns: ArrayLike | list | None = None,
    target_column: ArrayLike | str | None = None,
) -> TestSuiteReturnType | None:
    from vectice import Table
    from vectice.models.validation_dataset import TestSuiteReturnType

    if dataset is None:
        return None

    # Define patterns for sensitive data types
    sensitive_data_patterns = {
        "credit_card": r"\b(?:\d[ -]*?){13,16}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    }

    # Initialize a dictionary to hold counts of sensitive data type matches
    sensitive_data_counts = {key: 0 for key in sensitive_data_patterns.keys()}

    # Check each column in the DataFrame for sensitive data type patterns
    for column in dataset.columns:
        for key, pattern in sensitive_data_patterns.items():
            sensitive_data_counts[key] += (
                dataset[column]
                .astype(str)
                .str.contains(pattern, case=False, regex=True)
                .sum()
            )

    # Create a DataFrame with the results
    sensitive_data_counts_df = pd.DataFrame(
        {
            "Sensitive Data Type": list(sensitive_data_counts.keys()),
            "Count": list(sensitive_data_counts.values()),
        }
    )

    table = Table(sensitive_data_counts_df)

    return TestSuiteReturnType(
        properties={},
        tables=[table],
        attachments=[],
    )


# Map the tests to be used
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
