import requests

r = requests.get('https://little-turtle.herokuapp.com/')

print(r.status_code)
print(r.json())

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

r = requests.post('https://little-turtle.herokuapp.com/predict', json=row)

print(r.status_code)
print(r.json())
