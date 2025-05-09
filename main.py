# main.py

from data.load_data import load_data
from preprocessing.preprocess import preprocess_data
from models.base_models import train_logistic_regression_base, train_random_forest_base, train_xgboost_base
from evaluation.evaluate_model import evaluate_model


def main():
    """
    Main function to load the credit card dataset and initiate the analysis pipeline.
    """

    # Load dataset
    file_path = 'data/default_of_credit_card_clients.csv'
    df = load_data(file_path)

    # Check if the dataframe is not empty and print message accordingly.
    if df is not None:
        print("Dataset loaded successfully.")
        print(df.head())
    else:
        print("Failed to load dataset.")

    # Get the training (unsampled and sampled) and test data upon preprocessing, for EDA.
    X_train, y_train, X_train_resampled, y_train_resampled, X_test, y_test = preprocess_data(df)

    # --------- Train Base Models on Unsampled Data ---------
    lr_base_model = train_logistic_regression_base(X_train, y_train)
    rf_base_model = train_random_forest_base(X_train, y_train)
    xgb_base_model = train_xgboost_base(X_train, y_train)

    # --------- Evaluate Base Models trained on Unsampled Data ---------
    evaluate_model(lr_base_model, X_test, y_test, model_name="logreg_base_unsampled")
    evaluate_model(rf_base_model, X_test, y_test, model_name="random_forest_base_unsampled")
    evaluate_model(xgb_base_model, X_test, y_test, model_name="lxgboost_base_unsampled")

    # --------- Train Base Models on Resampled Data ---------
    lr_base_resampled = train_logistic_regression_base(X_train_resampled, y_train_resampled)
    rf_base_resampled = train_random_forest_base(X_train_resampled, y_train_resampled)
    xgb_base_resampled = train_xgboost_base(X_train_resampled, y_train_resampled)

    # --------- Evaluate Base Models trained on Resampled Data ---------
    evaluate_model(lr_base_resampled, X_test, y_test, model_name="logreg_base_resampled")
    evaluate_model(rf_base_resampled, X_test, y_test, model_name="random_forest_base_resampled")
    evaluate_model(xgb_base_resampled, X_test, y_test, model_name="xgboost_base_resampled")


if __name__ == "__main__":
    main()
