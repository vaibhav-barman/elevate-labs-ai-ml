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


from sklearn.linear_model import LogisticRegression


def train_logistic_regression(X_train_scaled, y_train, random_state=42):
    """
    Train a Logistic Regression model on the standardized training data.
    Inspect the learned coefficients and intercept.
    """
    print("\n" + "=" * 70)
    print("PHASE 3: LOGISTIC REGRESSION MODEL TRAINING")
    print("=" * 70)

    # Initialize Logistic Regression with reproducible random_state
    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    print("Model successfully trained.")
    print(f"Number of iterations to converge: {model.n_iter_[0]}")
    print(f"Model intercept (b)             : {model.intercept_[0]:.4f}")
    print(f"Number of coefficients (weights): {len(model.coef_[0])}")

    # Inspect top 3 positive and top 3 negative coefficients
    coef_series = pd.Series(model.coef_[0], index=feature_names).sort_values()
    print("\nTop 3 features pushing towards Class 0 (Malignant, negative weights):")
    for feat, val in coef_series.head(3).items():
        print(f"  {feat:25s}: {val:+.4f}")

    print("\nTop 3 features pushing towards Class 1 (Benign, positive weights):")
    for feat, val in coef_series.tail(3).items():
        print(f"  {feat:25s}: {val:+.4f}")

    return model


def make_predictions(model, X_test_scaled):
    """
    Generate discrete class predictions and continuous probability predictions.
    """
    print("\n" + "-" * 70)
    print("GENERATING MODEL PREDICTIONS")
    print("-" * 70)

    # Discrete class predictions (default threshold = 0.5)
    y_pred = model.predict(X_test_scaled)

    # Continuous probability predictions for both classes [P(class 0), P(class 1)]
    y_prob_all = model.predict_proba(X_test_scaled)

    # Probability of positive class (Class 1: Benign)
    y_prob = y_prob_all[:, 1]

    print("Predictions generated for test set:")
    print(f"  Predicted class counts: Class 0={(y_pred == 0).sum()}, Class 1={(y_pred == 1).sum()}")
    print("\nFirst 5 test sample probabilities and assigned class:")
    print("  Index | P(Malignant / 0) | P(Benign / 1) | Predicted Class")
    print("  ----------------------------------------------------------")
    for i in range(5):
        print(f"    {i:2d}  |      {y_prob_all[i, 0]:.4f}      |    {y_prob[i]:.4f}     |      {y_pred[i]} ({target_names[y_pred[i]]})")

    return y_pred, y_prob


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def compute_classification_metrics(y_test, y_pred, y_prob):
    """
    Calculate core evaluation metrics:
    Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
    Break down the confusion matrix into TN, FP, FN, and TP.
    """
    print("\n" + "=" * 70)
    print("PHASE 4: MODEL EVALUATION METRICS")
    print("=" * 70)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("Core Classification Metrics:")
    print(f"  Accuracy : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"  Precision: {prec:.4f} ({prec * 100:.2f}%)")
    print(f"  Recall   : {rec:.4f} ({rec * 100:.2f}%)")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")

    # Confusion Matrix Breakdown
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix:")
    print("                Predicted Malignant (0)  Predicted Benign (1)")
    print(f"Actual Malignant (0)         {tn:3d} (TN)                 {fp:3d} (FP)")
    print(f"Actual Benign (1)            {fn:3d} (FN)                 {tp:3d} (TP)")

    print("\nConfusion Matrix Components Explanation:")
    print(f"  - True Negative  (TN) = {tn:2d}: Correctly diagnosed as Malignant (Class 0).")
    print(f"  - False Positive (FP) = {fp:2d}: Type I error. Actually Malignant, falsely flagged as Benign.")
    print(f"  - False Negative (FN) = {fn:2d}: Type II error. Actually Benign, falsely flagged as Malignant.")
    print(f"  - True Positive  (TP) = {tp:2d}: Correctly diagnosed as Benign (Class 1).")

    print("\nDetailed Scikit-Learn Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Malignant (0)", "Benign (1)"]))

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "cm": cm,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }
    return metrics


import os
import matplotlib.pyplot as plt


def plot_sigmoid_curve(output_dir):
    """
    Explain and visualize the Sigmoid Activation Function:
    sigma(z) = 1 / (1 + e^(-z))
    """
    print("\n" + "=" * 70)
    print("PHASE 5: SIGMOID ACTIVATION FUNCTION")
    print("=" * 70)

    print("Mathematical formulation:")
    print("  sigma(z) = 1 / (1 + exp(-z))")
    print("\nKey concepts:")
    print("  1. What z represents:")
    print("     z = w^T * x + b = w_1*x_1 + w_2*x_2 + ... + w_n*x_n + b")
    print("     z is the linear combination of weighted features plus bias (log-odds).")
    print("  2. S-shaped squashing property:")
    print("     - As z -> +infinity, exp(-z) -> 0, sigma(z) -> 1")
    print("     - As z -> -infinity, exp(-z) -> +infinity, sigma(z) -> 0")
    print("     - When z = 0, sigma(0) = 1 / (1 + 1) = 0.5")
    print("     The output is strictly bounded in the open interval (0, 1).")
    print("  3. Probability interpretation:")
    print("     P(y = 1 | x) = sigma(z)")
    print("     P(y = 0 | x) = 1 - sigma(z)")
    print("  4. Decision rule:")
    print("     y_hat = 1 if P(y = 1 | x) >= threshold (default 0.5, i.e., z >= 0)")
    print("     y_hat = 0 otherwise (i.e., z < 0)")

    # Generate z values
    z = np.linspace(-10, 10, 500)
    sigma_z = 1 / (1 + np.exp(-z))

    # Plot Sigmoid Curve
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "sigmoid_curve.png")

    plt.figure(figsize=(9, 5.5), dpi=150)
    plt.plot(z, sigma_z, color="#1f77b4", linewidth=2.5, label=r"$\sigma(z) = \frac{1}{1 + e^{-z}}$")

    # Key threshold and reference lines
    plt.axhline(0.5, color="#d62728", linestyle="--", alpha=0.8, label="Default Threshold: P = 0.5 (z = 0)")
    plt.axvline(0.0, color="#7f7f7f", linestyle=":", alpha=0.7)
    plt.axhline(1.0, color="#2ca02c", linestyle=":", alpha=0.5, label="Upper Asymptote (P = 1.0)")
    plt.axhline(0.0, color="#9467bd", linestyle=":", alpha=0.5, label="Lower Asymptote (P = 0.0)")

    # Decision regions
    plt.fill_between(z, 0, 1, where=(z >= 0), color="#2ca02c", alpha=0.08, label="Class 1 Region (z >= 0, P >= 0.5)")
    plt.fill_between(z, 0, 1, where=(z < 0), color="#d62728", alpha=0.08, label="Class 0 Region (z < 0, P < 0.5)")

    # Annotate inflection point
    plt.scatter([0], [0.5], color="#d62728", s=80, zorder=5)
    plt.annotate(
        "Inflection Point\n(z=0, P=0.5)",
        xy=(0, 0.5),
        xytext=(1.5, 0.40),
        arrowprops=dict(facecolor="#333333", arrowstyle="->", lw=1.2),
        fontsize=10,
        fontweight="bold"
    )

    plt.title("Sigmoid (Logistic) Activation Function", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Log-Odds: $z = w^T x + b$", fontsize=12)
    plt.ylabel(r"Predicted Probability: $P(y = 1 \mid x)$", fontsize=12)
    plt.xlim(-10, 10)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", fontsize=9.5)
    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
    print(f"Sigmoid curve saved successfully to: {save_path}")


if __name__ == "__main__":
    # Define project results directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "..", "results")

    X, y, feature_names, target_names = load_and_inspect_dataset()
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data(X, y)
    model = train_logistic_regression(X_train_scaled, y_train)
    y_pred, y_prob = make_predictions(model, X_test_scaled)
    metrics = compute_classification_metrics(y_test, y_pred, y_prob)
    plot_sigmoid_curve(results_dir)




