import os
import joblib
import optuna
import pandas as pd
import dvc.api
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from pipeline import build_pipeline
from logger import get_logger
import dagshub

# ================== Setup ====================
load_dotenv()
logger = get_logger(__name__)

dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
dagshub.init(
    repo_owner=os.getenv("DAGSHUB_USERNAME"),
    repo_name="MLOps2",
    mlflow=True
)
mlflow.set_tracking_uri("https://dagshub.com/ahmed.sobeih/MLOps2.mlflow")

# ================== Objective Function ====================
def objective(trial, params):
    model_name = params["model"]["name"]
    search = params["search_space"][model_name]
    target_col = params["data"]["target"]
    data_path = params["data"]["path"]
    test_size = params["training"]["test_size"]
    seed = params["training"].get("random_state", 42)

    logger.info(f"Trial {trial.number} started for model: {model_name}")

    # Suggest params
    model_params = {
        k: trial.suggest_float(k, *v) if isinstance(v[0], float) else
           trial.suggest_int(k, *v) if isinstance(v[0], int) else
           trial.suggest_categorical(k, v)
        for k, v in search.items()
    }
    model_params["random_state"] = seed

    try:
        df = pd.read_parquet(data_path)
        y = df[target_col]
        X = df.drop(columns=[target_col])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

        pipe = build_pipeline(model_name=model_name, model_params=model_params)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        acc = accuracy_score(y_test, preds)
        return acc

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        raise

# ================== Run Tuning ====================
def run_tuning(params):
    logger.info("Starting hyperparameter tuning...")
    model_name = params["model"]["name"]
    target_col = params["data"]["target"]
    data_path = params["data"]["path"]
    test_size = params["training"]["test_size"]
    n_trials = params["training"]["n_trials"]
    seed = params["training"].get("random_state", 42)

    from functools import partial
    f_objective = partial(objective, params=params)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))

    with mlflow.start_run(run_name="Titanic_Run") as run:
        mlflow.set_tag("project", "Titanic")
        mlflow.set_tag("model_type", model_name)

        study.optimize(f_objective, n_trials=n_trials)
        best_params = study.best_params
        mlflow.log_params(best_params)
        mlflow.log_param("model_name", model_name)

        df = pd.read_parquet(data_path)
        y = df[target_col]
        X = df.drop(columns=[target_col])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

        final_model = build_pipeline(model_name=model_name, model_params={**best_params, "random_state": seed})
        final_model.fit(X_train, y_train)
        preds = final_model.predict(X_test)

        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
        })

        os.makedirs("models", exist_ok=True)
        save_path = "models/model_pipeline.pkl"
        joblib.dump(final_model, save_path)
        logger.info(f"Final model saved to: {save_path}")

        # Log the pipeline directly (no wrapper)
        mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="model",
            registered_model_name="Titanic_Classifier_Model"
        )

if __name__ == "__main__":
    params = dvc.api.params_show()
    run_tuning(params)
