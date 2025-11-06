import os
import json
from pathlib import Path

import joblib
import pandas as pd
import dvc.api
import mlflow
import dagshub
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from src.logger import get_logger

logger = get_logger(__name__)

def setup_mlflow(cfg):
    mlflow.set_tracking_uri("https://dagshub.com/ahmed.sobeih/MLOps2.mlflow")
    client = mlflow.client.MlflowClient(tracking_uri="https://dagshub.com/ahmed.sobeih/MLOps2.mlflow")
    logger.info("MLflow client initialized.")
    return client

def load_model(cfg):
    registry_model_name = cfg.get("registry_model_name", "Titanic_Classifier_Model")
    registry_stage = cfg.get("registry_stage")

    if registry_stage:
        model_uri = f"models:/{registry_model_name}@{registry_stage}"
        logger.info(f"Loading model from MLflow Registry at {model_uri}...")
        try:
            model = mlflow.sklearn.load_model(model_uri)
            return model, model_uri
        except Exception as exc:
            logger.warning(
                "Failed to load model from MLflow registry (%s). Falling back to local model if available. Error: %s",
                model_uri,
                exc,
            )

    model_path = cfg.get("model_path")
    if model_path:
        local_path = Path(model_path)
        logger.info(f"Loading model from local path: {local_path.resolve()}")
        model = joblib.load(local_path)
        return model, str(local_path)

    raise ValueError("No valid model source configured. Provide 'registry_stage' or 'model_path'.")


def evaluate(client, cfg):
    logger.info("Loading test data...")
    test_data_path = Path(cfg["test_data_path"])
    test_df = pd.read_parquet(test_data_path)
    y_test = test_df[cfg["target_col"]]
    X_test = test_df.drop(columns=[cfg["target_col"]])

    model, model_source = load_model(cfg)
    logger.info(f"Model loaded from: {model_source}")
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    cm = confusion_matrix(y_test, preds).tolist()

    report = {

        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm
    }

    report_path = Path(cfg["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report saved to {report_path}")

if __name__ == "__main__":
    logger.info("Starting evaluation...")

    # 1. Load params
    load_dotenv()
    params = dvc.api.params_show()
    cfg = params["evaluation"]

    # 2. DagsHub Auth
    dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init(
    repo_owner=os.getenv("DAGSHUB_USERNAME"),
    repo_name="MLOps2",  
    mlflow=True
    )

    # 3. Setup and run
    client = setup_mlflow(cfg)
    evaluate(client, cfg)
