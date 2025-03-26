from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


if TYPE_CHECKING:
    from matplotlib.container import BarContainer
    from numpy import ndarray
    from numpy.typing import ArrayLike
    from pandas import DataFrame

    from vectice.models.validation import TestSuiteReturnType

_logger = logging.getLogger(__name__)

def plot_correlation_matrix(
    training_df: DataFrame,
    testing_df: DataFrame,
    target_column: str,
    predictor: Any,
    predict_proba_train: ArrayLike | None,
    predict_proba_test: ArrayLike | None,
    internal_parameters: Dict[str, Any] = {"subset_columns": None, "cmap": "Blues"},
) -> TestSuiteReturnType:
    from vectice.models.validation import TestSuiteReturnType
    
    subset_columns = internal_parameters.get("subset_columns", [target_column] + [col for col in training_df.columns[:10] if col != "TARGET"])
    cmap = internal_parameters.get("cmap", "Blues")

    # Select subset of columns
    training_df = training_df[subset_columns]

    # Calculate the correlation matrix
    corr_matrix = training_df.corr()

    # Plot the correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, fmt=".2f", annot_kws={"fontsize": 12}, cbar=True)
    plt.title("Correlation Matrix")

    # Save the plot
    file_path = "Correlation_matrix_plot.png"
    plt.savefig(file_path)
    plt.close()

    # RETURN IN THE VECTICE EXPECTED FORMART
    return TestSuiteReturnType(
        metrics={},
        properties={},
        tables=[],
        attachments=[file_path],
    )