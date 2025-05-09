# evaluate_model.py

import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import json


def evaluate_model(model, X_test, y_test, model_name="model", output_dir="outputs"):
    """
    Evaluates the performance of a classification model using common metrics and plots.

    Parameters:
    - model: Trained classification model (e.g., LogisticRegression, RandomForestClassifier).
    - X_test: Features of the test dataset.
    - y_test: True labels of the test dataset.

    Prints:
    - Classification report with precision, recall, f1-score, and support.
    - Confusion matrix as a heatmap.
    - ROC AUC score.
    - ROC curve plot.

    Saves:
    - Classification report as JSON.
    - Confusion matrix plot as PNG.
    - ROC curve plot as PNG.
    """

    # Create output subfolders
    os.makedirs(f"{output_dir}/confusion_matrices", exist_ok=True)
    os.makedirs(f"{output_dir}/roc_curves", exist_ok=True)
    os.makedirs(f"{output_dir}/reports", exist_ok=True)

    # Predict class labels and probabilities
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    with open(f"{output_dir}/reports/{model_name}_report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"{output_dir}/confusion_matrices/{model_name}_confusion_matrix.png")
    plt.show()

    # ROC AUC Score
    roc_auc = roc_auc_score(y_test, y_proba)
    print("ROC AUC Score:", roc_auc)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.savefig(f"{output_dir}/roc_curves/{model_name}_roc_curve.png")
    plt.show()
