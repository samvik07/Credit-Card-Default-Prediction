# main.py

from data.load_data import load_data
from preprocessing.preprocess import preprocess_data


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


if __name__ == "__main__":
    main()
