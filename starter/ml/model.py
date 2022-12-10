from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    return lr


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def model_slice_performance(data, X, y, model, category):
    """ Calculates performance of model for slicing data by category

    Inputs
    ------
    data : pd.Dataframe
    X : np.ndarray
    y : np.ndarray
    model : sklearn model
    category : str
        Slices data
    Returns
    -------
    result : dict
        Per value in category the performance metrics
    """
    result = {}
    for val in data[category].unique():
        mask = (data[category] == val).values
        preds = inference(model, X[mask])
        precision, recall, fbeta = compute_model_metrics(y[mask], preds)
        result[val] = {"precision": precision, "recall": recall, "fbeta": fbeta}
    return result
