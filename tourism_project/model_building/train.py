
import argparse
import json
import os

from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_model(splits_dir, model_dir):

    splits_dir = Path(splits_dir)

    model_dir = Path(model_dir)

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # 1. LOAD TRAINING AND TESTING DATA
    # --------------------------------------------------

    train_path = splits_dir / "train.csv"

    test_path = splits_dir / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_path}"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Testing file not found: {test_path}"
        )

    train_df = pd.read_csv(train_path)

    test_df = pd.read_csv(test_path)

    target_column = "ProdTaken"

    if target_column not in train_df.columns:
        raise ValueError("ProdTaken missing from training data.")

    if target_column not in test_df.columns:
        raise ValueError("ProdTaken missing from testing data.")

    if train_df.columns.tolist() != test_df.columns.tolist():
        raise ValueError(
            "Training and testing columns do not match."
        )

    X_train = train_df.drop(
        columns=[target_column]
    )

    y_train = train_df[target_column]

    X_test = test_df.drop(
        columns=[target_column]
    )

    y_test = test_df[target_column]

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "Training and testing data must contain both classes."
        )

    print("=" * 60)

    print("TOURISM MODEL TRAINING")

    print("=" * 60)

    print("Training Shape:", X_train.shape)

    print("Testing Shape:", X_test.shape)

    # --------------------------------------------------
    # 2. IDENTIFY FEATURE TYPES
    # --------------------------------------------------

    numerical_features = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X_train.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # --------------------------------------------------
    # 3. DATA PREPROCESSING
    # --------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    # --------------------------------------------------
    # 4. RANDOM FOREST CLASSIFIER
    # --------------------------------------------------

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    random_state=42,
                    n_jobs=1
                )
            )
        ]
    )

    # --------------------------------------------------
    # 5. HYPERPARAMETER GRID
    # --------------------------------------------------

    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2],
        "model__class_weight": [
            None,
            "balanced",
            "balanced_subsample"
        ]
    }

    # --------------------------------------------------
    # 6. MLFLOW EXPERIMENT TRACKING
    # --------------------------------------------------

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "file:./mlruns"
    )

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(
        "Tourism_Package_Prediction"
    )

    with mlflow.start_run(
        run_name="Tourism_RandomForest_Production"
    ):

        mlflow.log_param(
            "model",
            "RandomForestClassifier"
        )

        mlflow.log_param(
            "target_column",
            target_column
        )

        mlflow.log_param(
            "training_rows",
            len(X_train)
        )

        mlflow.log_param(
            "testing_rows",
            len(X_test)
        )

        mlflow.log_param(
            "cross_validation",
            3
        )

        mlflow.log_param(
            "scoring_metric",
            "f1"
        )

        mlflow.log_dict(
            param_grid,
            "hyperparameter_search_space.json"
        )

        # --------------------------------------------------
        # 7. GRID SEARCH
        # --------------------------------------------------

        grid_search = GridSearchCV(
            estimator=model_pipeline,
            param_grid=param_grid,
            cv=3,
            scoring="f1",
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )

        grid_search.fit(
            X_train,
            y_train
        )

        best_model = grid_search.best_estimator_

        best_params = grid_search.best_params_

        best_cv_score = grid_search.best_score_

        print("\nBest Parameters:")

        print(best_params)

        print("\nBest CV F1 Score:", best_cv_score)

        # --------------------------------------------------
        # 8. LOG ALL HYPERPARAMETER COMBINATIONS
        # --------------------------------------------------

        cv_results = grid_search.cv_results_

        for i, params in enumerate(cv_results["params"]):

            with mlflow.start_run(nested=True):

                mlflow.log_params({
                    key: str(value)
                    for key, value in params.items()
                })

                mlflow.log_metric(
                    "mean_cv_f1_score",
                    float(cv_results["mean_test_score"][i])
                )

                mlflow.log_metric(
                    "mean_cv_train_f1_score",
                    float(cv_results["mean_train_score"][i])
                )

        mlflow.log_params({
            key: str(value)
            for key, value in best_params.items()
        })

        mlflow.log_metric(
            "best_cv_f1_score",
            float(best_cv_score)
        )

        # --------------------------------------------------
        # 9. MODEL EVALUATION
        # --------------------------------------------------

        y_pred = best_model.predict(X_test)

        y_probability = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(
                y_test, y_pred, zero_division=0
            ),
            "recall": recall_score(
                y_test, y_pred, zero_division=0
            ),
            "f1_score": f1_score(
                y_test, y_pred, zero_division=0
            ),
            "roc_auc": roc_auc_score(
                y_test, y_probability
            )
        }

        mlflow.log_metrics({
            key: float(value)
            for key, value in metrics.items()
        })

        print("\nMODEL EVALUATION")

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # --------------------------------------------------
        # 10. CLASSIFICATION REPORT
        # --------------------------------------------------

        report = classification_report(
            y_test,
            y_pred,
            zero_division=0
        )

        print("\nClassification Report:")

        print(report)

        report_path = (
            model_dir / "classification_report.txt"
        )

        report_path.write_text(report)

        mlflow.log_artifact(
            str(report_path)
        )

        # --------------------------------------------------
        # 11. CONFUSION MATRIX
        # --------------------------------------------------

        matrix = confusion_matrix(
            y_test,
            y_pred
        )

        matrix_path = (
            model_dir / "confusion_matrix.csv"
        )

        pd.DataFrame(
            matrix,
            index=["Actual_0", "Actual_1"],
            columns=["Predicted_0", "Predicted_1"]
        ).to_csv(matrix_path)

        mlflow.log_artifact(
            str(matrix_path)
        )

        # --------------------------------------------------
        # 12. HYPERPARAMETER RESULTS
        # --------------------------------------------------

        results_path = (
            model_dir / "hyperparameter_results.csv"
        )

        pd.DataFrame(
            cv_results
        ).to_csv(
            results_path,
            index=False
        )

        mlflow.log_artifact(
            str(results_path)
        )

        # --------------------------------------------------
        # 13. SAVE BEST MODEL
        # --------------------------------------------------

        model_path = model_dir / "model.joblib"

        joblib.dump(
            best_model,
            model_path
        )

        mlflow.log_artifact(
            str(model_path)
        )

        mlflow.sklearn.log_model(
            best_model,
            artifact_path="tourism_random_forest_model"
        )

        # --------------------------------------------------
        # 14. SAVE MODEL METADATA
        # --------------------------------------------------

        feature_metadata = {}

        for column in X_train.columns:

            if column in numerical_features:

                feature_metadata[column] = {
                    "type": "numerical",
                    "default": float(
                        X_train[column].median()
                    )
                }

            else:

                feature_metadata[column] = {
                    "type": "categorical",
                    "options": sorted(
                        X_train[column]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                }

        metadata = {
            "model_name": "RandomForestClassifier",
            "target_column": target_column,
            "best_parameters": {
                key: str(value)
                for key, value in best_params.items()
            },
            "cross_validation_f1": float(best_cv_score),
            "evaluation_metrics": {
                key: float(value)
                for key, value in metrics.items()
            },
            "feature_metadata": feature_metadata
        }

        metadata_path = (
            model_dir / "model_metadata.json"
        )

        with open(metadata_path, "w") as file:

            json.dump(
                metadata,
                file,
                indent=4
            )

        mlflow.log_artifact(
            str(metadata_path)
        )

        # --------------------------------------------------
        # 15. FINAL OUTPUT
        # --------------------------------------------------

        print("\nModel saved:", model_path)

        print("Metadata saved:", metadata_path)

        print(
            "\nMLflow Run ID:",
            mlflow.active_run().info.run_id
        )

        print("\nMODEL TRAINING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--splits-dir",
        required=True
    )

    parser.add_argument(
        "--model-dir",
        required=True
    )

    args = parser.parse_args()

    train_model(
        args.splits_dir,
        args.model_dir
    )
