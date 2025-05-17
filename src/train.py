import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import hydra
from omegaconf import DictConfig
from src.data import load_data
from src.pipeline import build_pipeline
import joblib
from src.logger import get_logger
from sklearn.metrics import confusion_matrix
import os
logger = get_logger(__name__)



def objective(trial, cfg):
    model_name = cfg.model.name
    search = cfg.search_space[model_name]
    target_col = cfg.data.target
    data_path = cfg.data.path
    test_size = cfg.training.test_size

    logger.info(f" Trial {trial.number} started for model: {model_name}")


    if model_name == "random_forest":
        model_params = {
            "n_estimators": trial.suggest_int("n_estimators", search.n_estimators[0], search.n_estimators[1]),
            "max_depth": trial.suggest_int("max_depth", search.max_depth[0], search.max_depth[1]),
            "criterion": trial.suggest_categorical("criterion", search.criterion)
        }

    elif model_name == "xgboost":
        model_params = {
            "n_estimators": trial.suggest_int("n_estimators", search.n_estimators[0], search.n_estimators[1]),
            "max_depth": trial.suggest_int("max_depth", search.max_depth[0], search.max_depth[1]),
            "learning_rate": trial.suggest_float("learning_rate", search.learning_rate[0], search.learning_rate[1]),
            "subsample": trial.suggest_float("subsample", search.subsample[0], search.subsample[1])
        }

    elif model_name == "lightgbm":
        model_params = {
            "n_estimators": trial.suggest_int("n_estimators", search.n_estimators[0], search.n_estimators[1]),
            "max_depth": trial.suggest_int("max_depth", search.max_depth[0], search.max_depth[1]),
            "learning_rate": trial.suggest_float("learning_rate", search.learning_rate[0], search.learning_rate[1])
        }

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    logger.info(f"Trial {trial.number} parameters: {model_params}")

    try:
        df = load_data(data_path)
        y = df[target_col]
        X = df.drop(columns=[target_col])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        pipe = build_pipeline(model_name=model_name, model_params=model_params)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        acc = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds)
        recall = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)

        logger.info(f"Trial {trial.number} accuracy: {acc:.4f}")
        logger.info(f"Trial {trial.number} precision: {precision:.4f}")
        logger.info(f"Trial {trial.number} recall: {recall:.4f}")
        logger.info(f"Trial {trial.number} f1_score: {f1:.4f}")
        logger.info(f"Trial {trial.number} Confusion Matrix:\n{cm}")

        return acc
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        raise
    
def run_tuning(cfg: DictConfig):
    logger.info("Starting hyperparameter tuning...")

    model_name = cfg.model.name
    data_path = cfg.data.path
    target_col = cfg.data.target
    test_size = cfg.training.test_size
    n_trials = cfg.training.n_trials

    study = optuna.create_study(direction="maximize")
    
    study.optimize(lambda trial: objective(trial, cfg), n_trials=n_trials)

    logger.info("Best trial:")
    logger.info(study.best_trial)

    best_params = study.best_params
    logger.info(f"Retraining final model with best parameters: {best_params}")

 
    df = load_data(data_path)
    y = df[target_col]
    X = df.drop(columns=[target_col])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    final_model = build_pipeline(model_name=model_name, model_params=best_params)
    final_model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    save_path = f"models/model_pipeline.pkl"
    joblib.dump(final_model, save_path)
    logger.info(f"Final model saved to: {save_path}")
