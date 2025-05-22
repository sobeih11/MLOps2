import os
import json
import pandas as pd
import dvc.api
import mlflow
import dagshub
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from logger import get_logger

logger = get_logger(__name__)

def setup_mlflow(cfg):
    mlflow.set_tracking_uri("https://dagshub.com/ahmed.sobeih/MLOps2.mlflow")
    client = mlflow.client.MlflowClient(tracking_uri="https://dagshub.com/ahmed.sobeih/MLOps2.mlflow")
    logger.info("MLflow client initialized.")
    return client

def evaluate(client, cfg):
    logger.info("Loading test data...")
    test_df = pd.read_parquet(cfg["test_data_path"])
    y_test = test_df[cfg["target_col"]]
    X_test = test_df.drop(columns=[cfg["target_col"]])

    logger.info(f"Loading model from MLflow Registry...")
    model_uri = "models:/Titanic_Classifier_Model@champion1"
    model = mlflow.sklearn.load_model(model_uri)
    logger.info(f"Model loaded from URI: {model_uri}")
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

    os.makedirs(os.path.dirname(cfg["report_path"]), exist_ok=True)
    with open(cfg["report_path"], "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report saved to {cfg['report_path']}")

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
