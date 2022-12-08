from sklearn.model_selection import train_test_split

import pandas as pd
import json
from joblib import load
from ml.data import process_data
from ml.model import model_slice_performance

data = pd.read_csv("./data/clean_census.csv")

train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# required to fetch encoder and lb (when training=True)
_, _, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)


X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

filename = './model/model_params.joblib'
model = load(filename)

for feature in cat_features:
    result = model_slice_performance(test, X_test, y_test, model, feature)
    print(json.dumps(result, indent=4))

