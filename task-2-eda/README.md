# Task 2 — Exploratory Data Analysis

## Overview

This project is a beginner-friendly exploratory data analysis (EDA) of the Titanic passenger dataset. It uses descriptive statistics and clear visualisations to understand the data before machine-learning modelling.

## Objective

Identify distributions, relationships, unusual values, missing data, and potential preprocessing needs in the Titanic data. The analysis describes patterns; it does not make causal claims.

## Dataset

The included `data/titanic.csv` is the public Titanic Dataset (891 passenger records) published by M Yasser H on Kaggle under CC0. Key columns include `Survived`, `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Cabin`, and `Embarked`.

## Technologies Used

- Python, Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- Jupyter Notebook
- Statsmodels (Variance Inflation Factor)

## Analysis Performed

- Data inspection and data-quality checks
- Missing-value analysis and EDA-level cleaning
- Descriptive statistics (including mean, median, and standard deviation)
- Univariate, bivariate, and multivariate analysis
- Correlation and pairplot analysis
- Outlier detection, skewness, and multicollinearity analysis

## Key Findings

The executed notebook derives its findings from the included data: 38.4% of passengers survived; female passengers had a substantially higher observed survival rate (74.2%) than male passengers (18.9%); first class had the highest observed survival rate (63.0%) and third class the lowest (24.2%). Age has 177 missing values, Cabin has 687, and Embarked has 2. Fare is strongly right-skewed and has potential IQR outliers. These are observations, not evidence of causation.

## Project Structure

```text
task-2-eda/
├── data/titanic.csv
├── notebooks/task_2_eda.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## How to Run

From `task-2-eda/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

Open `notebooks/task_2_eda.ipynb` and run all cells. The notebook uses a relative path and works when launched from the project root.

## Learning Outcomes

This analysis practises descriptive statistics, data-quality assessment, visualisation, correlation interpretation, skewness, potential outlier identification, and multicollinearity checks. It also distinguishes an observed association from a causal conclusion and prepares the data for later modelling.
