# Write custom tests which can be used to validate your datasets quality
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from vectice.models.validation_dataset import TestSuiteReturnType


# custom test which can be used for dataset validation
def test_dataset_split(
    dataset: DataFrame | None,
    training_df: DataFrame,
    testing_df: DataFrame,
    target_column: str,
    feature_columns: list | None = None,
    threshold: float | None = None,
) -> TestSuiteReturnType:
    from vectice import Table
    from vectice.models.validation_dataset import TestSuiteReturnType

    if dataset is None:
        return None

    total_df = len(training_df) + len(testing_df)

    # Create a DataFrame with the results
    datasplit_df = pd.DataFrame(
        {
            "Dataset": ["Train", "Test", "Total"],
            "Size": [len(training_df), len(testing_df), total_df],
            "Percentage": [
                (len(training_df) / total_df * 100),
                (len(testing_df) / total_df * 100),
                100,
            ],
        }
    )

    table = Table(datasplit_df)

    return TestSuiteReturnType(
        properties={},
        tables=[table],
        attachments=[],
    )


# custom test which can be used for dataset validation
def iqr_and_outliers(
    dataset: DataFrame | None = None,
    training_df: DataFrame | None = None,
    testing_df: DataFrame | None = None,
    feature_columns: list | None = None,
    target_column: str | None = None,
    threshold: float | None = None,
) -> TestSuiteReturnType | None:
    from vectice.models.validation_dataset import TestSuiteReturnType

    if dataset is None:
        return None

    files = []
    # disable plots showing
    plt.ioff()
    for column in dataset.select_dtypes(include=[np.number]).columns:
        file_name = f"iqr_and_outliers_{column}.png"

        temp_file_path = file_name

        Q1 = dataset[column].quantile(0.25)
        Q3 = dataset[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        plt.figure(figsize=(10, 6))
        plt.hist(dataset[column], bins=20, edgecolor="k", alpha=0.7)
        plt.axvline(
            Q1, color="r", linestyle="--", label=f"Q1 (25th percentile): {Q1:.2f}"
        )
        plt.axvline(
            Q3, color="b", linestyle="--", label=f"Q3 (75th percentile): {Q3:.2f}"
        )
        plt.axvline(
            dataset[column].median(),
            color="g",
            linestyle="-",
            label=f"Median: {dataset[column].median():.2f}",
        )
        plt.fill_betweenx(
            [0, plt.ylim()[1]], Q1, Q3, color="gray", alpha=0.3, label=f"IQR: {IQR:.2f}"
        )

        # Highlight outliers
        outliers = dataset[
            (dataset[column] < lower_bound) | (dataset[column] > upper_bound)
        ][column]
        plt.scatter(
            outliers, [0] * len(outliers), color="red", label="Outliers", zorder=5
        )

        plt.title(f"Histogram with IQR and Outliers for {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(temp_file_path, bbox_inches="tight")
        files.append(temp_file_path)

    plt.ion()
    return TestSuiteReturnType(
        properties={},
        tables=[],
        attachments=files,
    )
