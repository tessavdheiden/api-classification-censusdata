from sklearn.model_selection import train_test_split

import pandas as pd
import json
from joblib import load
from ml.data import process_data
from ml.model import model_slice_performance, compute_model_metrics

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

model = load('./model/model.joblib')
encoder = load('./model/encoder.joblib')
lb = load('./model/lb.joblib')

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb
)

_, _, fscore = compute_model_metrics(y_test, model.predict(X_test))
print(f"F-score: {fscore}")

all_results = {}
for feature in cat_features:
    result = model_slice_performance(test, X_test, y_test, model, feature)
    print(feature, ":")
    fscores = {}
    for key in result.keys():
        fscores[key] = result[key]["fbeta"]
    fscores = dict(sorted(fscores.items(), key=lambda item: item[1]))
    keys = list(fscores.keys())
    if len(keys) > 1:
        print(
            json.dumps(
                {
                    "low": f"{keys[0]}, {fscores[keys[0]]}",
                    "high": f"{keys[-1]}, {fscores[keys[-1]]}"},
                indent=4))
    all_results[feature] = result


with open("slice_output.txt", 'w') as file:
    for feature, results in all_results.items():
        file.write(f"{feature} : \n")
        for val, score in results.items():
            file.write(f"\t{val} : \n")
            for metric, value in score.items():
                file.write(f"\t\t{metric}:{value:.2f}\n")
