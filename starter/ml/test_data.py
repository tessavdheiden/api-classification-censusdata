import pandas as pd
import pytest
import numpy as np
import pickle

import sklearn.preprocessing

from starter.ml.model import train_model, compute_model_metrics, inference
from starter.ml.data import process_data
from sklearn.model_selection import train_test_split


@pytest.fixture
def data():
    # """ Read csv file """
    # df = pd.read_csv("./data/clean_census.csv")
    # return df
    """ Simple function to generate some fake Pandas data."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "numeric_feat": [3.14, 2.72, 1.62],
            "categorical_feat": ["dog", "dog", "cat"],
        }
    )
    return df


def test_process_data_return_data_type(data):
    train, test = train_test_split(data, test_size=0.20)
    X_test, y_test, _, _ = process_data(
        test, categorical_features=["categorical_feat"], label="id", training=True
    )
    assert type(X_test) == np.ndarray
    assert type(y_test) == np.ndarray
    assert np.all(X_test[:, 0] > 0)


def test_process_data_return_model_type(data):
    train, test = train_test_split(data, test_size=0.20)
    _, _, encoder, lb = process_data(
        test, categorical_features=["categorical_feat"], label="id", training=True
    )
    assert type(encoder) == sklearn.preprocessing.OneHotEncoder
    assert type(lb) == sklearn.preprocessing.LabelBinarizer


def test_train_model(data):
    train, test = train_test_split(data, test_size=0.20)
    X_train, y_train, encoder, lb = process_data(
        train, ["categorical_feat"], label="id", training=True
    )
    model = train_model(X_train, y_train)
    assert type(model) == sklearn.linear_model.LogisticRegression
