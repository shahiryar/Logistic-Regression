import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    
    n = X.shape[0]
    test_indices = np.random.choice(n, int(n * test_size), replace=False)
    train_indices = np.array(list(set(range(n)) - set(test_indices)))
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test