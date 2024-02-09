import numpy as np
import pandas as pd
import pdb

from sklearn.datasets import make_regression, make_friedman1
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys

# Input parameters
if len(sys.argv) != 3:
    print("Error: incorrect number of parameters.")
    quit()

n_train = int(sys.argv[1])
batch = int(sys.argv[2])


# Parameters of experiment
n_test = 1000

def run_experiment(random_seed):
    n_samples = n_train + n_test
    X, Y = make_regression(n_samples=n_samples, n_features=10, random_state=random_seed)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=n_test, 
                                                        random_state=random_seed)
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    Y_train = scaler_Y.fit_transform(Y_train.reshape((-1,1))).flatten()

    model = RandomForestRegressor(max_depth=2, random_state=random_seed)
    model.fit(X_train, Y_train)
    X_test = scaler_X.transform(X_test)
    Y_test = scaler_Y.transform(Y_test.reshape((-1,1))).flatten()
    Y_test_hat = model.predict(X_test)
    test_mse = np.sqrt(np.mean(np.power(Y_test - Y_test_hat, 2) ))
    results = pd.DataFrame({'n_train':[n_train], 'random_seed':[random_seed],
                            'test_mse':[test_mse]})
    return results

out_file = "results/n"+str(n_train) + "_batch"+str(batch)+".txt"
full_results = pd.DataFrame({})
for r in range(10):
    print("Running experiment {:d} of {:d}...".format(r+1, 10))
    random_seed = 1000*batch + r
    results = run_experiment(random_seed)
    full_results = pd.concat([full_results, results])
    full_results.to_csv(out_file, index=False)
    print("Completed experiment {:d} of {:d}.".format(r+1, 10))

print(full_results)
