# main.py

from data.load_data import load_data


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


if __name__ == "__main__":
    main()
