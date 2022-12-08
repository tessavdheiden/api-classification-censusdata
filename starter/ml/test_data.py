import pandas as pd
import pytest
import pickle
from starter.ml.model import train_model, compute_model_metrics, inference


@pytest.fixture
def data():
    """ Read csv file """
    df = pd.read_csv("./data/clean_census.csv")
    return df


# @pytest.fixture
# def model():
#     """ Load model """
#     filename = './model/model_params.pkl'
#     loaded_model = pickle.load(open(filename, "rb"))
#     return loaded_model


def test_train_model(data):
    """ If your data is assumed to have no null values then this is a valid test. """
    model = train_model(X_train=X_train, y_train=y_train)



# def test_slice_averages(data):
#     """ Test to see if our mean per categorical slice is in the range 1.5 to 2.5."""
#     for cat_feat in data["categorical_feat"].unique():
#         avg_value = data[data["categorical_feat"] == cat_feat]["numeric_feat"].mean()
#         assert (
#             2.5 > avg_value > 1.5
#         ), f"For {cat_feat}, average of {avg_value} not between 2.5 and 3.5."