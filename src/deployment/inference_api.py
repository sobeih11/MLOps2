import os
import pickle
import litserve as ls
import pandas as pd
from pydantic import ValidationError
import joblib
from src.deployment.requests import InferenceRequest

class InferenceAPI(ls.LitAPI):
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_batch_size = 1
        self.enable_async = False
        self.batch_timeout = 0.1

    def setup(self, device="cpu"):
        model_path = "models/model_pipeline.pkl"  # ← Hardcoded path
        print(f"📦 Loading model from: {model_path}")
        self._model = joblib.load(model_path)
        print(f"✅ Model type: {type(self._model)}")


    def decode_request(self, request):
        print("🔥 Incoming request:", request)
        try:
            columns = request["dataframe_split"]["columns"]
            rows = request["dataframe_split"]["data"]
            validated_rows = []

            for row in rows:
                row_dict = dict(zip(columns, row))
                print("🧩 Row dict:", row_dict)
                validated = InferenceRequest(**row_dict)
                validated_rows.append(validated.dict())

            df = pd.DataFrame(validated_rows)
            print("📦 Final DataFrame:", df)
            return df
        except Exception as e:
            print("❌ Decode error:", e)
            return None

    def predict(self, x):
        print("📤 Predict input:", x)
        try:
            result = self._model.predict(x)
            print("✅ Predict output:", result)
            return result
        except Exception as e:
            print("❌ Prediction error:", e)
            return None

    def encode_response(self, output):
        if output is None:
            return {"message": "Prediction failed", "data": []}
        return {"message": "Prediction succeeded", "data": output.tolist()}
