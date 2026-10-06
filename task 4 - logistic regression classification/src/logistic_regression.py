"""
Task 4: Classification with Logistic Regression
Dataset: Breast Cancer Wisconsin Diagnostic Dataset (sklearn)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_and_inspect_dataset():
    """
    Load the Breast Cancer Wisconsin dataset from scikit-learn
    and inspect its characteristics.
    """
    print("=" * 70)
    print("PHASE 1: DATASET INSPECTION")
    print("=" * 70)

    # Load dataset as pandas dataframe
    cancer_data = load_breast_cancer(as_frame=True)
    df = cancer_data.frame
    X = cancer_data.data
    y = cancer_data.target
    feature_names = cancer_data.feature_names
    target_names = cancer_data.target_names

    print(f"Total number of samples : {X.shape[0]}")
    print(f"Total number of features: {X.shape[1]}")
    print(f"\nFeature Names ({len(feature_names)} features):")
    for idx, name in enumerate(feature_names, start=1):
        print(f"  {idx:2d}. {name}")

    print("\nTarget Class Mapping:")
    for class_id, class_name in enumerate(target_names):
        count = (y == class_id).sum()
        pct = (count / len(y)) * 100
        print(f"  Class {class_id} -> {class_name:9s}: {count:3d} samples ({pct:.2f}%)")

    print("\nTarget Explanation:")
    print("  In the Breast Cancer Wisconsin dataset:")
    print("  - Class 0 = malignant (harmful, cancerous tumor)")
    print("  - Class 1 = benign (non-harmful, non-cancerous tumor)")
    print("  In standard binary classification conventions, 1 is the positive class.")

    print("\nSample Data (First 5 rows of selected features):")
    preview_cols = list(feature_names[:5]) + ["target"]
    print(df[preview_cols].head())

    return X, y, feature_names, target_names


if __name__ == "__main__":
    X, y, feature_names, target_names = load_and_inspect_dataset()
