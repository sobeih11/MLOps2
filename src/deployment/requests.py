from pydantic import BaseModel

class InferenceRequest(BaseModel):
    Age: float
    Fare: float
    Sex: str
    Embarked: str
    Pclass: int
