import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    #  Explicitly set tracking URI for local MLflow tracking
    mlflow.set_tracking_uri("file:./mlruns")

    # Read training & test data
    file_path = sys.argv[3] if len(sys.argv) > 3 else "clean_train.csv"
    file_path_test = sys.argv[4] if len(sys.argv) > 4 else "clean_test.csv"

    data_train = pd.read_csv(file_path)
    data_test = pd.read_csv(file_path_test)

    X_train = data_train.drop("Attrition", axis=1)
    y_train = data_train["Attrition"]
    X_test = data_test.drop("Attrition", axis=1)
    y_test = data_test["Attrition"]

    # Parameters
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    # Use input_example for model schema logging
    input_example = X_train.head(5)

    #  Start MLflow run
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)

        predicted = model.predict(X_test)
        accuracy = model.score(X_test, y_test)

        # Log model (safe usage with local tracking URI)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",  # OK to keep if tracking URI is local
            input_example=input_example
        )

        mlflow.log_metric("accuracy", accuracy)

        print(f"Logged model with accuracy: {accuracy:.4f}")
