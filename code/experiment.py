import numpy as np
import pandas as pd
import pdb

from sklearn.datasets import make_regression, make_friedman1
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import sys

# Input parameters
if len(sys.argv) != 4:
    print("Error: incorrect number of parameters.")
    quit()

n_train = int(sys.argv[1])
batch = int(sys.argv[2])
model_name = sys.argv[3]

# Parameters of experiment
n_test = 1000
n_samples = n_train + n_test

def run_experiment(random_seed):
    X, Y = make_regression(n_samples=n_samples, n_features=10, random_state=random_seed)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=n_test, 
                                                        random_state=random_seed)
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    Y_train = scaler_Y.fit_transform(Y_train.reshape((-1,1))).flatten()
    
    if model_name=="rf":
        model = RandomForestRegressor(random_state=random_seed)
    elif model_name=="dnn":
        model = MLPRegressor(random_state=random_seed, max_iter=500)

    model.fit(X_train, Y_train)
    X_test = scaler_X.transform(X_test)
    Y_test = scaler_Y.transform(Y_test.reshape((-1,1))).flatten()
    Y_test_hat = model.predict(X_test)
    test_mse = np.sqrt(np.mean(np.power(Y_test - Y_test_hat, 2) ))
    results = pd.DataFrame({'n_train':[n_train], 
                            'random_seed':[random_seed],
                            'model':[model_name],
                            'test_mse':[test_mse]})
    return results

out_file = "results/n"+str(n_train) + "_batch"+str(batch)+"_" + model_name + ".txt"
full_results = pd.DataFrame({})
for r in range(10000):
    print("Running experiment {:d} of {:d}...".format(r+1, 10))
    sys.stdout.flush() 
    random_seed = 1000*batch + r
    results = run_experiment(random_seed)
    full_results = pd.concat([full_results, results])
    full_results.to_csv(out_file, index=False)
    print("Completed experiment {:d} of {:d}.".format(r+1, 10))
    sys.stdout.flush() 

print(full_results)
