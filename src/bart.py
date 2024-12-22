import numpy as np

from samplers import DefaultSampler
from priors import DefaultBARTPrior


class BART:
    """
    API for the BART model.
    """
    def __init__(self, prior="default", sampler="default", x_preprocessor="default", y_preprocessor="default", 
                 generator=np.random.Generator, ndpost=1000, nskip=100):
        """
        Initialize the BART model.

        Parameters:
        - prior: BARTPrior
            Priors for the BART model.
        - posterior: BARTPosterior
            Posterior for the BART model.
        - sampler: Sampler
            Sampler for the BART model.
        - x_preprocessor: XPreprocessor
            Preprocessor for the input data.
        - y_preprocessor: YPreprocessor
            Preprocessor for the output data.
        """
        self.prior = DefaultBARTPrior()
        self.sampler = sampler
        self.x_preprocessor = x_preprocessor
        self.y_preprocessor = y_preprocessor
        self.generator = generator
        self.ndpost = ndpost
        self.nskip = nskip
        self.trace = []

    def fit(self, X, y):
        """
        Fit the BART model.
        """
        X_tf = self.x_preprocessor.fit_transform(X)
        y_tf = self.y_preprocessor.fit_transform(y)
        self.sampler = DefaultSampler(X_tf, y_tf, self.prior, self.nskip+self.ndpost, self.generator, 200)
        self.sampler.run()

    def posterior_f(self, X):
        """
        Get the posterior distribution of f(x) for each row in X.
        """
        preds = np.zeros((X.shape[0], self.ndpost))
        X_tf = self.x_preprocessor.transform(X)
        for k in range(self.nskip, self.nskip + self.ndpost):
            preds[:, k] = self.y_preprocessor.inverse_transform(self.sampler.trace[k].evaluate(X_tf))
        return preds
    
    def predict(self, X):
        """
        Predict using the BART model.
        """
        return np.mean(self.predict_distribution(X), axis=1)