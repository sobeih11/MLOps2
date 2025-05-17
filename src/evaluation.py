import pandas as pd
import joblib
import os
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import hydra
from omegaconf import DictConfig
from src.logger import get_logger
logger = get_logger(__name__)

def evaluate_pipeline(cfg: DictConfig):
    cfg = cfg.pipeline.evaluation  

    test_df = pd.read_parquet(cfg.test_data_path)
    y_test = test_df[cfg.target_col]
    X_test = test_df.drop(columns=[cfg.target_col])

    model = joblib.load(cfg.model_path)
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

    os.makedirs(os.path.dirname(cfg.report_path), exist_ok=True)
    with open(cfg.report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report saved to {cfg.report_path}")


