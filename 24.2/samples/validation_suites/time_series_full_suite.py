# import the Vectice provided time series tests
from vectice.models.test_library.time_series_test import (
    trend_analysis,
    seasonality_check,
    autocorrelation_test,
    stationarity_test,
    missing_value_analysis,
)

# Map the tests to be used
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
