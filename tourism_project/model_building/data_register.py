
import pandas as pd
from pathlib import Path

# Dataset location
RAW_PATH = Path("tourism_project/data/tourism.csv")

# Expected dataset columns
EXPECTED_COLUMNS = [
    "CustomerID",
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "DurationOfPitch",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "ProductPitched",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome"
]


def register_dataset():

    # Validate dataset existence
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_PATH}"
        )

    # Load dataset
    df = pd.read_csv(RAW_PATH)

    # Validate expected columns
    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}"
        )

    print("=" * 60)
    print("TOURISM DATASET REGISTRATION")
    print("=" * 60)

    print("\nDataset registered successfully.")

    print("\nDataset Shape:", df.shape)

    print("\nDataset Columns:")
    print(df.columns.tolist())

    print("\nDataset Information:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Records:", df.duplicated().sum())

    print("\nTarget Distribution:")
    print(df["ProdTaken"].value_counts())

    print("\nTarget Statistics:")
    print(df["ProdTaken"].describe())

    print("\nDATASET REGISTRATION COMPLETED")


if __name__ == "__main__":
    register_dataset()
