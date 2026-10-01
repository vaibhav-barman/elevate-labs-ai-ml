# Task 1 — Data Cleaning and Preprocessing

This project is part of my **Artificial Intelligence and Machine Learning Internship at Elevate Labs**.

The objective is to explore a real-world dataset, identify data quality issues, and apply appropriate preprocessing techniques to prepare the data for machine learning.

## Objective

To build a structured data cleaning and preprocessing workflow using Python and its data science libraries.

## Dataset

**Titanic Dataset — M Yasser H**

Source: [Kaggle Titanic Dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset)

The dataset contains passenger information that can be used to practise missing-value handling, categorical encoding, numerical scaling, and exploratory data analysis.

Download the CSV from Kaggle and place it inside the `data/` directory.

Expected location:

```text
data/
└── Titanic-Dataset.csv
```

The dataset file is excluded from this repository. Refer to Kaggle for the original dataset and its licence.

## Tools and Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## Workflow

The notebook is designed to cover the following stages:

1. **Data loading:** Load the CSV into a Pandas DataFrame.
2. **Initial inspection:** Review dimensions, column names, data types, and summary statistics.
3. **Data quality checks:** Identify missing values and duplicate records.
4. **Missing-value handling:** Apply suitable imputation strategies.
5. **Categorical encoding:** Convert categorical features into numerical representations.
6. **Feature scaling:** Apply standardisation or normalisation where appropriate.
7. **Outlier analysis:** Visualise numerical distributions and investigate potential outliers.
8. **Preprocessing pipeline:** Use Scikit-learn tools to organise transformations where appropriate.

Preprocessing decisions should be based on the data and task requirements. Potential outliers should be investigated rather than automatically removed.

## Project Structure

```text
Task-1-Data-Cleaning-Preprocessing/
├── README.md
├── data/
│   └── Titanic-Dataset.csv  # Download separately
└── data_cleaning.ipynb
```

## How to Run

From the root of the repository:

```bash
source .venv/bin/activate
jupyter notebook
```

Open `Task-1-Data-Cleaning-Preprocessing/data_cleaning.ipynb` and run the notebook cells in order.

If the virtual environment has not been created yet, follow the setup instructions in the root README.

## Learning Outcomes

This task provides practical experience with:

* Data inspection and quality assessment.
* Missing-value treatment.
* Categorical feature encoding.
* Numerical feature scaling.
* Exploratory data analysis and visualisation.
* Reproducible data preparation using Python.

## Author

**Vaibhav Barman**

[GitHub Profile](https://github.com/vaibhav-barman)

## Internship

Artificial Intelligence and Machine Learning Internship — Elevate Labs
