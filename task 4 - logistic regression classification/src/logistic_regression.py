"""
Task 4: Classification with Logistic Regression
Dataset: Breast Cancer Wisconsin Diagnostic Dataset (sklearn)
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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


def prepare_data(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets with stratification,
    and standardize features using StandardScaler.

    DATA LEAKAGE PREVENTION:
    - StandardScaler is fit ONLY on X_train: scaler.fit_transform(X_train)
    - X_test is scaled using ONLY train parameters: scaler.transform(X_test)
    """
    print("\n" + "=" * 70)
    print("PHASE 2: DATA PREPARATION & STANDARDIZATION")
    print("=" * 70)

    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"Train set shape: {X_train.shape} ({(1 - test_size) * 100:.0f}%)")
    print(f"Test set shape : {X_test.shape} ({test_size * 100:.0f}%)")

    # Verify stratified class proportions
    train_dist = y_train.value_counts(normalize=True)
    test_dist = y_test.value_counts(normalize=True)
    print("\nClass distribution verification (Stratification):")
    print(f"  Train: Class 0 (Malignant)={train_dist[0]:.2%}, Class 1 (Benign)={train_dist[1]:.2%}")
    print(f"  Test : Class 0 (Malignant)={test_dist[0]:.2%}, Class 1 (Benign)={test_dist[1]:.2%}")

    # Feature Standardization
    scaler = StandardScaler()

    # Fit ONLY on training data, then transform training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform test data using fitted training parameters (NO FIT ON TEST)
    X_test_scaled = scaler.transform(X_test)

    print("\nFeature Standardization details:")
    print("  Scaler fit strictly on X_train to prevent data leakage.")
    print(f"  Train mean before scaling: {X_train.iloc[:, 0].mean():.2f}, std: {X_train.iloc[:, 0].std():.2f}")
    print(f"  Train mean after scaling : {X_train_scaled[:, 0].mean():.2f}, std: {X_train_scaled[:, 0].std():.2f}")
    print(f"  Test mean after scaling  : {X_test_scaled[:, 0].mean():.2f}, std: {X_test_scaled[:, 0].std():.2f}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    X, y, feature_names, target_names = load_and_inspect_dataset()
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(X, y)

