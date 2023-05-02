import pytest
from logisticRegression.regressor import LogisticRegression
import numpy as np

@pytest.fixture
def binary_data():
    X = np.array([
        [0.3, 0.5, 0.1],
        [0.6, 0.1, 0.9],
        [0.2, 0.4, 0.7],
        [0.8, 0.1, 0.3]
    ])
    y = np.array([0, 1, 1, 0])
    return X, y

def test_binary_classification(binary_data):
    X, y = binary_data
    model = LogisticRegression(lr=0.1, num_iter=5000, fit_intercept=True, verbose=False)
    model.fit(X, y)
    y_pred = model.predict(X)
    assert y_pred.shape == y.shape
    assert np.all(np.isin(y_pred, [0, 1]))

@pytest.fixture
def multi_class_data():
    X = np.array([
        [0.3, 0.5, 0.1],
        [0.6, 0.1, 0.9],
        [0.2, 0.4, 0.7],
        [0.8, 0.1, 0.3],
        [0.4, 0.5, 0.1],
        [0.9, 0.2, 0.7],
        [0.5, 0.4, 0.2],
        [0.7, 0.3, 0.5]
    ])
    y = np.array([0, 1, 1, 0, 0, 2, 1, 2])
    return X, y

def test_multi_class_classification(multi_class_data):
    X, y = multi_class_data
    model = LogisticRegression(lr=0.1, num_iter=5000, fit_intercept=True, verbose=False)
    model.fit(X, y)
    y_pred = model.predict_one_vs_all(X, labels=np.unique(y))
    assert y_pred.shape == (len(np.unique(y)), len(y))
    assert np.all(np.isin(y_pred.argmax(axis=0), np.unique(y)))