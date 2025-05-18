from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from preprocess import build_preprocessor

def build_pipeline(model_name="random_forest", model_params: dict = None):
    if model_params is None:
        model_params = {}

    preprocessor = build_preprocessor()

    if model_name == "random_forest":
        clf = RandomForestClassifier(**model_params)

    elif model_name == "xgboost":
        clf = XGBClassifier(use_label_encoder=False, eval_metric="logloss", **model_params)

    elif model_name == "lightgbm":
        clf = LGBMClassifier(**model_params)

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    return pipe
