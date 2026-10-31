
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(data_path, output_dir):

    data_path = Path(data_path)
    output_dir = Path(output_dir)

    # --------------------------------------------------
    # 1. LOAD DATASET
    # --------------------------------------------------

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}"
        )

    df = pd.read_csv(data_path)

    print("Dataset loaded successfully.")
    print("Original Dataset Shape:", df.shape)

    # --------------------------------------------------
    # 2. VALIDATE TARGET
    # --------------------------------------------------

    target_column = "ProdTaken"

    if target_column not in df.columns:
        raise ValueError(
            "ProdTaken target column is missing."
        )

    if df[target_column].isnull().any():
        raise ValueError(
            "Target column contains missing values."
        )

    # --------------------------------------------------
    # 3. REMOVE DUPLICATES
    # --------------------------------------------------

    initial_records = len(df)

    df.drop_duplicates(inplace=True)

    print(
        "Duplicate Records Removed:",
        initial_records - len(df)
    )

    # --------------------------------------------------
    # 4. REMOVE UNNECESSARY COLUMNS
    # --------------------------------------------------

    unnecessary_columns = [
        "Unnamed: 0",
        "CustomerID"
    ]

    existing_columns = [
        column for column in unnecessary_columns
        if column in df.columns
    ]

    df.drop(
        columns=existing_columns,
        inplace=True
    )

    print(
        "Removed Columns:",
        existing_columns
    )

    # --------------------------------------------------
    # 5. SEPARATE FEATURES AND TARGET
    # --------------------------------------------------

    X = df.drop(columns=[target_column])

    y = df[target_column]

    # --------------------------------------------------
    # 6. TRAIN TEST SPLIT
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Combine features and target
    train_data = X_train.copy()

    train_data[target_column] = y_train

    test_data = X_test.copy()

    test_data[target_column] = y_test

    # --------------------------------------------------
    # 7. SAVE TRAINING AND TESTING DATA
    # --------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    train_data.to_csv(
        output_dir / "train.csv",
        index=False
    )

    test_data.to_csv(
        output_dir / "test.csv",
        index=False
    )

    # --------------------------------------------------
    # 8. DISPLAY SUMMARY
    # --------------------------------------------------

    print("\nDATA PREPARATION SUMMARY")

    print("Cleaned Dataset Shape:", df.shape)

    print("Training Dataset Shape:", train_data.shape)

    print("Testing Dataset Shape:", test_data.shape)

    print("\nTraining Target Distribution:")
    print(y_train.value_counts())

    print("\nTesting Target Distribution:")
    print(y_test.value_counts())

    print("\nTraining and testing datasets saved successfully.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        required=True
    )

    args = parser.parse_args()

    prepare_data(
        args.data,
        args.output_dir
    )
