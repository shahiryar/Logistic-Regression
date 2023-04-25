import numpy as np

class LogisticRegression:
    """Creates Logictic Regression model from a given set of data

        Attributes:
            lr (float): learning rate, a float representing the size of the step used for updating the parameters during training
            num_iter (int): The number of iterations to run the optimization algorithm during training
            fit_intercept (Boolean): A boolean indicating wheather to fit an intercept term (bias) in the logistic regression model
            verbose (Boolean): A boolan indicating whether to print the progress during training
    """
    
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        """Initialize a new instance of Logictic Regression model

            Args:
                lr (float): learning rate, a float representing the size of the step used for updating the parameters during training
                num_iter (int): The number of iterations to run the optimization algorithm during training
                fit_intercept (Boolean): A boolean indicating wheather to fit an intercept term (bias) in the logistic regression model
                verbose (Boolean): A boolan indicating whether to print the progress during training
            Returns:
                Instance of LogisticRegression
        """
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    
    def fit(self, X, y):
        """Train the Logistic Regression model for the given dataset.

        Args:
            X (numpy.array): A numpy array containing the input features.
            y (numpy.array): A numpy array containing the output features corresponding to the given input features.

        Returns:
            self (LogisticRegression): Returns an instance of the LogisticRegression class with the trained weights.

        Examples:
            >>> # Create a logistic regression model and fit the model to the training data
            >>> model = LogisticRegression(lr=0.01, num_iter=5000, fit_intercept=True, verbose=False)
            >>> model.fit(X_train, y_train)
        """
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # initialize weights as zeros
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            if self.verbose and i % 10000 == 0:
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold=0.5):
        """Predict the output for the given data using a trained logistic regression model.

        Args:
            X (numpy.array): A numpy array containing the input data to predict the output for.
            threshold (float, optional): A float representing the minimum probability threshold for classification.
                Default is 0.5.

        Returns:
            numpy.array: A numpy array containing the predicted output values for the input data.

        Examples:
            >>> # Create and train a logistic regression model
            >>> model = LogisticRegression()
            >>> model.fit(X_train, y_train)
            >>>
            >>> # Predict the output for new input data
            >>> y_pred = model.predict(X_test, threshold=0.7)
        """
        return self.predict_prob(X) >= threshold

    def vizualize_results(X_test, y_test, method='confusion_matrix'):
        from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
        import matplotlib.pyplot as plt
        import seaborn as sns
        """Visualize the accuracy of the model using various metrics.

        Args:
            X_test (numpy.array): A numpy array containing the test input data.
            y_test (numpy.array): A numpy array containing the expected output values for the test data.
            method (str, optional): A string specifying the visualization method to use. 
                Default is 'confusion_matrix'. Possible values are 'confusion_matrix', 
                'roc_curve', and 'precision_recall_curve'.

        Returns:
            None: This function does not return anything, it only produces visualizations.

        Raises:
            ValueError: If an invalid method name is provided.

        Examples:
            >>> # Create and train a logistic regression model
            >>> model = LogisticRegression()
            >>> model.fit(X_train, y_train)
            >>> 
            >>> # Visualize the accuracy of the model using a confusion matrix
            >>> model.vizualize_results(X_test, y_test, method='confusion_matrix')
        """
        if method == 'confusion_matrix':
            y_pred = self.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, cmap='Blues')
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.show()
            
        elif method == 'roc_curve':
            y_pred_prob = self.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.show()
            
        elif method == 'precision_recall_curve':
            y_pred_prob = self.predict_proba(X_test)[:, 1]
            precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.show()
            
        else:
            print('Invalid method name.')

    def accuracy(X_test, y_test, threshold=0.5):
        """
        Calculates the accuracy of the logistic regression model on the test data.

        Parameters:
        -----------
        X_test : array-like of shape (n_samples, n_features)
            The test input samples.

        y_test : array-like of shape (n_samples,)
            The true target values for the test input samples.

        threshold : float, optional (default=0.5)
            The threshold value to use for the predicted probabilities.
            All probabilities above this threshold are considered positive.

        Returns:
        --------
        float
            The accuracy of the logistic regression model on the test data.
            This is defined as the number of correct predictions divided by
            the total number of predictions.

        Raises:
        -------
        ValueError
            If X_test and y_test have incompatible shapes, or if y_test contains
            values other than 0 or 1.
        """
        y_pred = self.predict(X_test, threshold=threshold)
        return (np.equal(y_pred, y_test).sum()/len(y_test))
    
    def confusion_matrix(X_test, y_test):
        """Compute the confusion matrix for the logistic regression model.
        Parameters:
        -----------
        X_test: array-like of shape (n_samples, n_features)
        Test data.

        y_test: array-like of shape (n_samples,)
        True labels for `X_test`.

        Returns:
        --------
        confusion_m: array-like of shape (n_classes, n_classes)
        Confusion matrix, where `n_classes` is the number of unique classes in `y_test`.
        The rows represent the actual classes and the columns represent the predicted classes.
        The (i, j) element of the matrix represents the number of instances where the actual class
        was i and the predicted class was j.
        """

        y_pred = self.predict(X_test)
        classes = np.unique(y_test)
        confusion_m = []
        for pred_c in classes:
            idx = np.where(y_pred == pred_c)
            _test_pred = y_test[idx]
            _row = []
            for actual_c in classes:
                _row.append(np.equal(_test_pred, actual_c).sum())
            confusion_m.append(_row)

        return np.array(confusion_m).T