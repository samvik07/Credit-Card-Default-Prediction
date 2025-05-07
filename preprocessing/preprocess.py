# preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter


def preprocess_data(df):
    """
    Preprocesses the input DataFrame:
        - Drops the ID and target columns
        - Scales the features
        - Splits into training and test sets
        - Applies SMOTE oversampling to the training set to handle class imbalance

    Parameters:
    ----------
    df : pd.DataFrame
        Input DataFrame containing features and target

    Returns:
    -------
    X_train : np.ndarray
        Scaled training features (before SMOTE)
    y_train : pd.Series
        Training labels (before SMOTE)
    X_train_resampled : np.ndarray
        Resampled and scaled training features
    y_train_resampled : np.ndarray
        Resampled training labels
    X_test : np.ndarray
        Scaled testing features
    y_test : pd.Series
        Testing labels
    """

    # Ensure 'default' is integer
    df['default'] = df['default'].astype(int)

    # Separate features and target, drop ID column
    X = df.drop(['ID', 'default'], axis=1)
    y = df['default']

    # Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Before oversampling:", Counter(y_train))

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    print("After oversampling:", Counter(y_train_resampled))

    return X_train, y_train, X_train_resampled, y_train_resampled, X_test, y_test
