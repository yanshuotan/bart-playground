import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OrdinalEncoder

def load_mushroom(dataset_size: int = 8000):
    X, y = fetch_openml('mushroom', version=1, return_X_y=True)
    for col in X.select_dtypes('category'):
        # -1 in codes indicates NaN by pandas convention
        X[col] = X[col].cat.codes
        
    X_array = X.to_numpy()
    y_array = y.to_numpy().reshape(-1, 1)
    y_arm = OrdinalEncoder(dtype=int).fit_transform(y_array) 
    
    # make the dataset a little bit smaller
    indices = np.random.choice(X_array.shape[0], size=dataset_size, replace=False)
    # indices = range(X.shape[0])
    X_array = X_array[indices, :]
    y_arm = y_arm[indices]
    
    return X_array, y_arm

def load_mushroom_encoded():
    X, y_arm = load_mushroom()
    
    n_arm = np.max(y_arm) + 1
    dim = X.shape[1] * n_arm # total number of encoded covariates (location-encoded for each arm) 
    act_dim = X.shape[1] # number of covariates
    covariates = np.zeros((X.shape[0], dim))
    rewards = np.zeros((X.shape[0], ))
    for cursor in range(X.shape[0]):
        a = np.random.randint(0, n_arm)
        covariates[cursor, a * act_dim:(a * act_dim + act_dim)] = X[cursor]
        if y_arm[cursor] == a:
            rewards[cursor] = 1 # reward is 1 if the true category matches the chosen arm

    return covariates, rewards
