# Put the code for your API here.
import fastapi
import joblib
from fastapi import FastAPI, Body
from pydantic import BaseModel
from starter.ml.model import inference
from starter.ml.data import process_data
import pandas as pd

app = FastAPI()


class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float
    capital_loss: float
    hours_per_week: int
    native_country: str

    class Config:
        schema_extra = {
            "example": {
                "age": 24,
                "workclass": "Private",
                "fnlgt": 176580,
                "education": "5th-6th",
                "education-num": 3,
                "marital-status": "Married-spouse-absent",
                "occupation": "Farming-fishing",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 0,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "Mexico",
            }
        }


@app.get("/")
async def read_root():
    return {"Hello": "Stranger"}


@app.post("/predict/")
async def predict(data: Data
                  # = Body(
                  #     example={
                  #         "age": 24,
                  #         "workclass": "Private",
                  #         "fnlgt": 176580,
                  #         "education": "5th-6th",
                  #         "education-num": 3,
                  #         "marital-status": "Married-spouse-absent",
                  #         "occupation": "Farming-fishing",
                  #         "relationship": "Not-in-family",
                  #         "race": "White",
                  #         "sex": "Male",
                  #         "capital-gain": 0,
                  #         "capital-loss": 0,
                  #         "hours-per-week": 40,
                  #         "native-country": "Mexico",
                  #     },
                  # ),
                  ):
    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]
    model = joblib.load("./model/model.joblib")
    encoder = joblib.load("./model/encoder.joblib")
    lb = joblib.load("./model/lb.joblib")
    data = dict(data)
    data = pd.DataFrame(data, index=[0])

    X, y, _, _ = process_data(
        data, categorical_features=cat_features, training=False,
        encoder=encoder, lb=lb
    )
    pred = inference(model, X)[0]
    classes = ["<=50k", ">50k"]
    return {"salary": classes[pred]}
