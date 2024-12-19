class BART:
    """
    API for the BART model.
    """
    def __init__(self, prior, posterior, sampler, x_preprocessor, y_preprocessor):
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
        self.prior = prior
        self.posterior = posterior
        self.sampler = sampler
        self.x_preprocessor = x_preprocessor
        self.y_preprocessor = y_preprocessor
        self.trace = []

    def fit(self, X, y):
        """
        Fit the BART model.
        """
        pass

    def predict(self):
        """
        Predict using the BART model.
        """
        pass