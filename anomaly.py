class Anomaly():
    """
    Anomaly is an anomaly detector class.
    
    It fits a normal distribution model to each feature of the input data set,
    then predicting the probability of a new input of being within this
    distribution.
    
    An example is considered anomalous if the calculated probability is less
    than a specific threshold EPSILON, and a normal example otherwise.
    """
    def __init__(self, epsilon=0.5):
        self.epsilon = epsilon
        self.mu = 0
        self.std = 0
        
    def fit(self, X):
        """
        X.shape must be (M, N) where M, N is a positive integer.
        M is the number of examples in the training set.
        N is the number of features in each training example.
        
        **If X is an image, IT MUST BE FLATTENED TO THE PREVIOUS SHAPE.**
        """
        self.mu = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        
        ### DEBUG CODE ###
#        print("Means:")
#        plt.hist(np.asarray(self.mu).reshape((-1)), bins=100)
#        print("Standard Deviations:")
#        plt.hist(np.asarray(self.std).reshape((-1)), bins=100)
        ### DEBUG CODE ENDS ###
    
    
    def featureProb(self, f, mu, std):
        """
        Returns the probability estimation of feature f in some example.
        
        f, mu and std MUST BE the same shape (1, N) where N is the number of 
        features.
        """
        from math import exp
        prob = exp(-((f-mu)**2 / (2*std**2))) / (2.506628*std)
        ### DEBUG CODE ###
#        print("Feature Probability:", prob)
        ### DEBUG CODE ENDS ###
        return prob
    
    
    def predict(self, x):
        """
        x is an input example.
        x.shape MUST BE (1, N) where N is the number of features.
        returns a boolean value:
            True: anomalous example - p(x) < EPSILON
            False: non-anomalous example - p(x) >= EPSILON
        """
        probs = np.vectorize(self.featureProb)(x, self.mu, self.std)
        ### DEBUG CODE ###
#        print("Image Probability:", probs)
        ### DEBUG CODE ENDS ###
        return np.product(probs) < self.epsilon