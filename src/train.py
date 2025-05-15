import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
from src.data import load_data
from src.pipeline import build_pipeline
import joblib
from src.logger import get_logger
from sklearn.metrics import confusion_matrix
import os
logger = get_logger(__name__)

# =========================
# CONFIGURABLE SECTION
# =========================
target_col = "Survived"
model_name = "xgboost" 
n_trials = 20
# =========================

def objective(trial):
    logger.info(f"Trial {trial.number} started.")

    if model_name == "random_forest":
        model_params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"])
        }
    elif model_name == "xgboost":
        model_params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0)
        }
    elif model_name == "lightgbm":
        model_params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3)
        }

    logger.info(f"Trial {trial.number} parameters: {model_params}")

    df = load_data("data/raw/Titanic-Dataset.csv")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    try:
        pipe = build_pipeline(model_name=model_name, model_params=model_params)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        logger.info(f"Trial {trial.number} accuracy: {acc:.4f}")
        logger.info(f"Trial {trial.number} precision: {precision:.4f}")
        logger.info(f"Trial {trial.number} recall: {recall:.4f}")
        logger.info(f"Trial {trial.number} f1_score: {f1:.4f}")

        return acc
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        raise

def run_tuning(n_trials=n_trials):
    logger.info("Starting hyperparameter tuning...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    logger.info("Best trial:")
    logger.info(study.best_trial)

    best_params = study.best_params
    logger.info(f"Retraining final model with best parameters: {best_params}")

    df = load_data("data/raw/Titanic-Dataset.csv")
    y = df[target_col]
    X = df.drop(columns=[target_col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    final_model = build_pipeline(model_name=model_name, model_params=best_params)
    final_model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    joblib.dump(final_model, f"models/{model_name}_pipeline.pkl")
    logger.info(f"Final model saved to models/{model_name}_pipeline.pkl")
