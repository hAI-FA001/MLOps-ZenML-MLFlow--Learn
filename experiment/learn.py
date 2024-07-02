import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def train_test() -> None:
    digits = load_digits()
    data = digits.images.reshape((len(digits.images), -1))
    X_tr, X_ts, y_tr, y_ts = train_test_split(data, digits.target, test_size=0.2, shuffle=False)
    
    model = SVC(gamma=0.001)
    model.fit(X_tr, y_tr)

    test_acc = model.score(X_ts, y_ts)
    print(f"Test accuracy: {test_acc}")


# make it into a pipeline

from zenml import step
from typing_extensions import Annotated
import pandas as pd
from typing import Tuple

@step
def importer() -> Tuple[
    Annotated[np.ndarray, "X_train"],
    Annotated[np.ndarray, "X_test"],
    Annotated[np.ndarray, "y_train"],
    Annotated[np.ndarray, "y_test"],
]:
    digits = load_digits()
    data = digits.images.reshape((len(digits.images), -1))
    X_tr, X_ts, y_tr, y_ts = train_test_split(data, digits.target, test_size=0.2, shuffle=False)

    return X_tr, X_ts, y_tr, y_ts


@step
def svc_trainer(X_train: np.ndarray, y_train: np.ndarray) -> ClassifierMixin:
    model = SVC(gamma=0.001)
    model.fit(X_train, y_train)
    
    return model

@step
def evaluator(X_test: np.ndarray, y_test: np.ndarray, model: ClassifierMixin) -> float:
    test_acc = model.score(X_test, y_test)
    print(f"Test accuracy: {test_acc}")
    
    return test_acc


# connect the steps

from zenml import pipeline

@pipeline
def digits_pipeline():
    X_train, X_test, y_train, y_test = importer()
    model = svc_trainer(X_train, y_train)
    evaluator(X_test, y_test, model)


# run the pipeline
digits_svc_pipeline = digits_pipeline()
