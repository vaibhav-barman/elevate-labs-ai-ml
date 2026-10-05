# Task 3 - Linear Regression

## Overview

This project applies simple and multiple linear regression to the supplied `Housing 2.csv` dataset. The executed notebook documents the complete workflow, from inspection and preprocessing through model evaluation, coefficient interpretation, multicollinearity checks, and residual diagnostics.

## Objective

Implement and compare simple and multiple linear regression models for predicting house `price`, then interpret their results responsibly.

## Dataset

- **File:** `data/Housing 2.csv`
- **Observations:** 545
- **Columns:** 13 (12 predictors plus the target)
- **Target variable:** `price`
- **Important features:** `area`, `bathrooms`, `bedrooms`, `stories`, `parking`, and the categorical housing characteristics (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`, and `furnishingstatus`).

The dataset contains no missing values and no duplicate rows. The simple model uses `area`, the numerical feature most strongly correlated with `price` (correlation: 0.536).

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Seaborn
- Statsmodels (VIF calculation)

## Project Workflow

1. Load the copied dataset using a relative path.
2. Inspect data types, descriptive statistics, missing values, duplicates, and categories.
3. Preserve the clean data and one-hot encode categorical predictors.
4. Train a simple linear regression model using `area`.
5. Train a multiple linear regression model using all non-target predictors.
6. Evaluate both models with MAE, MSE, RMSE, and R² on the same 80/20 test split (`random_state=42`).
7. Visualize the simple regression line, correlations, actual-vs-predicted values, and residuals.
8. Interpret the intercept and feature coefficients.
9. Check predictor correlation and VIF for multicollinearity.
10. Discuss regression assumptions and limitations.

## Evaluation Metrics

- **MAE:** Average absolute prediction error in the target's units.
- **MSE:** Average squared prediction error; it penalizes large errors more heavily.
- **RMSE:** Square root of MSE, expressed in the target's units.
- **R²:** Proportion of test-set target variation explained by the model.

## Results

All values below come from the executed notebook's held-out 20% test split.

| Model | MAE | MSE | RMSE | R² |
| --- | ---: | ---: | ---: | ---: |
| Simple Linear Regression (`area`) | 1,474,748.134 | 3,675,286,604,768.185 | 1,917,103.702 | 0.273 |
| Multiple Linear Regression | 970,043.404 | 1,754,318,687,330.665 | 1,324,506.960 | 0.653 |

The multiple model performs better on this split: it has a higher R² and lower MAE, MSE, and RMSE than the area-only model.

## Key Findings

- `area` is the strongest numerical correlate of `price` (0.536), making it the data-informed simple-regression feature.
- Using the remaining housing characteristics alongside area improves test R² from 0.273 to 0.653.
- The multiple model's MAE is about 970 thousand price units on the held-out data; individual predictions can still have meaningful error.
- The largest calculated VIF is 1.674, so no encoded predictor exceeds the common VIF screening threshold of 5 in this analysis.
- Correlation and regression coefficients describe associations in this dataset; they do not establish causal effects.

## Project Structure

```text
task-3-linear-regression/
├── data/
│   └── Housing 2.csv
├── notebooks/
│   └── task_3_linear_regression.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run

From the repository root on macOS:

```bash
cd task-3-linear-regression
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then open `notebooks/task_3_linear_regression.ipynb` and run all cells. The notebook loads `../data/Housing 2.csv` when opened from its `notebooks` directory, with a project-root fallback for convenience.

## Learning Outcomes

This task demonstrates data inspection, encoding categorical variables, reproducible train/test splitting, fitting simple and multiple regression, interpreting coefficients and error metrics, assessing multicollinearity, and using diagnostic plots to discuss model assumptions.
