# load_data.py

import pandas as pd


# Function for loading the dataset
def load_data(file_path):
    """
    Loads the credit card default dataset.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame with renamed target column.
    """

    try:
        # Load the dataset, skipping the first row as it contains header
        df = pd.read_csv(file_path, header=1)

        # Rename target column for clarity
        df.rename(columns={'default payment next month': 'default'}, inplace=True)

        return df

    except FileNotFoundError:
        print(f"File not found at {file_path}. Please check the path and try again.")
        return None

    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return None