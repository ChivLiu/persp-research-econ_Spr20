import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def error(bootstrap, seed, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.65, test_size = 0.35, random_state = seed)
    LogReg1 = LogisticRegression(n_jobs = 1, max_iter = 10000)
    LogReg1.fit(X_train, y_train)
    y_pred = LogReg1.predict(X_test)
    mse = np.mean((y_pred-y_test)**2)
    
    return mse