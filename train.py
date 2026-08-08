"""
train.py
Trains a scikit-learn classifier on the Iris dataset, tracks the run
(params, metrics, model artifact) in MLflow, and saves the trained model
to disk for the FastAPI service to load.

Usage:
    python train.py --n_estimators 200 --max_depth 5
"""

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path("data/sample.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
TARGET_COLUMN = "target"
CLASS_NAMES = ["setosa", "versicolor", "virginica"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train(n_estimators: int, max_depth: int, random_state: int = 42):
    X_train, X_test, y_train, y_test = load_data()

    mlflow.set_experiment("iris-classifier")

    with mlflow.start_run():
        mlflow.log_params(
            {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": random_state}
        )

        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
            "precision_macro": precision_score(y_test, y_pred, average="macro"),
            "recall_macro": recall_score(y_test, y_pred, average="macro"),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        METADATA_PATH.write_text(
            json.dumps(
                {
                    "feature_columns": FEATURE_COLUMNS,
                    "class_names": CLASS_NAMES,
                    "metrics": metrics,
                    "mlflow_run_id": mlflow.active_run().info.run_id,
                },
                indent=2,
            )
        )

        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Metrics: {metrics}")
        print(f"Model saved to {MODEL_PATH}")

    return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train the Iris classifier and log to MLflow.")
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=5)
    args = parser.parse_args()
    train(args.n_estimators, args.max_depth)


if __name__ == "__main__":
    main()
