# base_models.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def train_logistic_regression_base(X_train, y_train):
    """
    Train a base Logistic Regression model using the provided training data.

    Parameters:
        X_train (DataFrame or array-like): Features for training.
        y_train (Series or array-like): Target labels for training.

    Returns:
        LogisticRegression: Trained Logistic Regression model.
    """

    # Instantiate model
    lr_base = LogisticRegression()

    # Fit model to training data
    lr_base.fit(X_train, y_train)

    return lr_base


def train_random_forest_base(X_train, y_train):
    """
    Train a base Random Forest Classifier using the provided training data.

    Parameters:
        X_train (DataFrame or array-like): Features for training.
        y_train (Series or array-like): Target labels for training.

    Returns:
        RandomForestClassifier: Trained Random Forest model.
    """

    # Instantiate model with random_state=42 for reproducibility
    rf_base = RandomForestClassifier(random_state=42)

    # Fit model to training data
    rf_base.fit(X_train, y_train)
    return rf_base


def train_xgboost_base(X_train, y_train):
    """
    Train a base XGBoost Classifier using the provided training data.

    Parameters:
        X_train (DataFrame or array-like): Features for training.
        y_train (Series or array-like): Target labels for training.

    Returns:
        XGBClassifier: Trained XGBoost model.
    """

    # Instantiate XGBoost with key parameters
    xgb_base = xgb.XGBClassifier(eval_metric='logloss', random_state=42)

    # Fit model to training data
    xgb_base.fit(X_train, y_train)
    return xgb_base
