# Put the code for your API here.
import fastapi
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from starter.ml.model import inference
from starter.ml.data import process_data
import pandas as pd

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    global model, encoder, lb
    model = joblib.load("./model/model.joblib")
    encoder = joblib.load("./model/encoder.joblib")
    lb = joblib.load("./model/lb.joblib")


def hyphen_to_underscore(field_name):
    return f"{field_name}".replace("_", "-")


class Data(BaseModel):
    age: int = Field(..., example=45)
    capital_gain: int = Field(..., example=2174)
    capital_loss: int = Field(..., example=0)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., example=13)
    fnlgt: int = Field(..., example=2334)
    hours_per_week: int = Field(..., example=60)
    marital_status: str = Field(..., example="Never-married")
    native_country: str = Field(..., example="Cuba")
    occupation: str = Field(..., example="Prof-specialty")
    race: str = Field(..., example="Black")
    relationship: str = Field(..., example="Wife")
    sex: str = Field(..., example="Female")
    workclass: str = Field(..., example="State-gov")

    class Config:
        alias_generator = hyphen_to_underscore
        allow_population_by_field_name = True


@app.get("/")
async def read_root():
    return {"Hello": "Stranger"}


@app.post("/predict/")
async def predict(data: Data):
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
    data = dict(data)
    data = pd.DataFrame(data, index=[0])

    X, y, _, _ = process_data(
        data, categorical_features=cat_features, training=False,
        encoder=encoder, lb=lb
    )
    pred = inference(model, X)[0]
    classes = ["<=50k", ">50k"]
    return {"salary": classes[pred]}
