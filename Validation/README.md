## List of validation tests provided by Vectice (source code from PiML)
| **Category**                 | **Test Name**                    | **Function**                         |
|------------------------------|----------------------------------|--------------------------------------|
| **Classification Tests**     | ROC Curve                        | `plot_roc_curve`                     |
|                              | Confusion Matrix                 | `conf_matrix`                        |
|                              | Explainability                   | `explainability`                     |
|                              | Feature Importance               | `feature_importance`                 |
|                              | Label Drift                      | `label_drift`                        |
|                              | Prediction Drift                 | `prediction_drift`                   |
|                              | Recall by class                  | `recall_by_class `                   |
|                              | Precision by class               | `precision_by_class `                   |
|                              | **Binary Classification suite** | `plot_roc_curve`, `conf_matrix`, `explainability`, `feature_importance`, `label_drift`, `prediction_drift` |
|                              | **Multiclass Classification suite** | `plot_roc_curve`, `conf_matrix`, `explainability`, `feature_importance`, `label_drift`, `prediction_drift`, `recall_by_class `, `precision_by_class ` |
| **Data Privacy Tests**       | Sensitive Data Check             | `sensitive_data_check`               |
|                              | PII Check                        | `pii_check`                          |
|                              | Sensitive Data Type Check        | `sensitive_data_type_check`          |
| **Data Quality Tests**       | Dataset Split Validation         | `test_dataset_split`                 |
|                              | IQR and Outliers                 | `iqr_and_outliers`                   |
|                              | **Dataset Quality suite**    | `test_dataset_split`, `iqr_and_outliers` |
| **Regression Tests**         | Residuals Plot                   | `plot_residuals`                     |
|                              | R² Score                         | `r2_score`                           |
|                              | Explainability                   | `explainability`                     |
|                              | Feature Importance               | `feature_importance`                 |
|                              | Target Drift                     | `target_drift`                       |
|                              | Prediction Drift                 | `prediction_drift`                   |
|                              | **Regression suite**         | `plot_residuals`, `r2_score`, `explainability`, `feature_importance`, `target_drift`, `prediction_drift` |


