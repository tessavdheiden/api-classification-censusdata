import json

from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

from main import app

client = TestClient(app)


def test_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"Hello": "Stranger"}


def test_post_fixed_input():
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"}

    row = {}
    for k, v in data.items():
        row[k.replace("-", '_')] = v
    response = client.post(
        "/predict/",
        json=row,
    )
    assert response.status_code == 200


def test_post_input():
    df = pd.read_csv("./data/clean_census.csv", sep=",")
    df.pop('salary')
    # data = dict(df.iloc[0])
    data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"}

    row = {}
    for k, v in data.items():
        row[k.replace("-", '_')] = v

    response = client.post(
        "/predict/",
        json=row,
    )
    assert response.status_code == 200
    assert response.json() == {
        "salary": "<=50k"
    }
