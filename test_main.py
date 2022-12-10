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

    with TestClient(app) as client:
        response = client.post(
            "/predict/",
            json=data,
        )
        assert response.status_code == 200


def test_post_input_result_high():
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

    with TestClient(app) as client:
        response = client.post(
            "/predict/",
            json=data,
        )
        assert response.status_code == 200
        assert response.json() == {
            "salary": ">50k"
        }


def test_post_input_result_low():
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
        "race": "Black",
        "sex": "Female",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "Cuba"}

    with TestClient(app) as client:
        response = client.post(
            "/predict/",
            json=data,
        )
        assert response.status_code == 200
        assert response.json() == {
            "salary": "<=50k"
        }
