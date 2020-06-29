# -*- coding: utf-8 -*-
"""
# **I. Imports**
"""

# @title **I. 0. Import libraries**
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn import preprocessing
import matplotlib.pyplot as plt
from math import sqrt
import seaborn as sns
import scipy.stats.stats as stats


import logging

logger = logging.getLogger(__name__)


# @title **I. 1. Import data**
def import_data():
    """Import raw data

    Returns:
        pd.DataFrame, pd.DataFrame: Tables of train and test data
        Name of imported variables:
            - train (pd.DataFrame) : The table of train data
            - test (pd.DataFrame) : The table of test data
            - mapping_constellation (dict) : Mapping galaxy => constellation
    """

    BASE_PATH = "./data"
    train = pd.read_csv(f"{BASE_PATH}/train.csv")
    test = pd.read_csv(f"{BASE_PATH}/test.csv")
    mapping_constellation = pd.read_excel(f"{BASE_PATH}/mapping_galaxies.xlsx")
    mapping_constellation = mapping_constellation.set_index("galaxy").to_dict(
        orient="dict"
    )["constellation"]
    logger.info(
        f"Shape of train data {train.shape} \n" f"Shape of test data {test.shape}"
    )
    return train, test, mapping_constellation


# @title **I. 2. Evaluation function**
# @markdown You can find here the different function to use for evaluation <br/>
# @markdown The functions can be used in the following situations <br/>
# @markdown - `RMSE(y_true, y_pred)` Get the RMSE of y_true and y_pred
# @markdown - `train_and_evaluate(model, X, y)` Get all the scores of the 5-fold cross validation of your model
# @markdown - `get_mean_score(model, X, y, verbose=True)` Get the mean and std of the scores of the 5-fold cross validation
# @markdown - `compare_models(model, X, y, verbose=False)` Given a list of models, it compares them and return a DataFrame of the results


def RMSE(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))


def train_and_evaluate(model, X, y):
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = y.loc[train_index], y.loc[test_index]
        X_test, X_train, y_train = preprocess(X_test, X_train, y_train)
        columns_here = [*set(X_train.columns).intersection(X_test.columns)]
        model.fit(X_train[columns_here], y_train)
        scores.append(RMSE(y_test, model.predict(X_test[columns_here])))
    return scores


def get_mean_score(model, X, y, verbose=True):
    scores = train_and_evaluate(model, X, y)
    mean = np.mean(scores)
    std = np.std(scores)
    if verbose:
        logger.info(
            f"5-Fold RMSE for the model {type(model).__name__}: "
            f"{mean:.2e} +/- {std:.2e}"
        )
        logger.info(scores)
    return mean, std


def compare_models(model_list, X, y, verbose=False):
    model_names = [type(model).__name__ for model in model_list]
    df = pd.DataFrame(np.nan, index=model_names, columns=["Mean RMSE", "Std RMSE"])
    for model in models:
        model_name = type(model).__name__
        try:
            mean, std = get_mean_score(model, X, y, verbose=verbose)
            df.loc[model_name, "Mean RMSE"] = mean
            df.loc[model_name, "Std RMSE"] = std
        except:
            logger.info(f"The model {model_name} was skipped")
    return df


"""# **IV. Model**
Train and evaluate the model(s) here

### **IV. 1. Cross validation**
Here we can compare models by doing 5-fold cross validation
"""


model_to_eval = ExtraTreesRegressor(
    bootstrap=False,
    ccp_alpha=0,
    criterion="mae",
    max_depth=None,
    max_features="auto",
    max_leaf_nodes=None,
    max_samples=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    min_samples_leaf=1,
    min_samples_split=2,
    min_weight_fraction_leaf=0.0,
    n_estimators=10,
    n_jobs=None,
    oob_score=False,
    random_state=None,
    verbose=0,
    warm_start=False,
)

# @title **Train Test Validation**
X_train, X_test, y_train, y_test = train_test_split(
    train.drop("y", axis=1), train[["y"]], test_size=0.33
)
X_test_preprocessed, X_train_p, y_train_p = preprocess(X_test, X_train, y_train)
columns_here = [*set(X_train_p.columns).intersection(X_test_preprocessed.columns)]
model_to_eval.fit(X_train_p[columns_here], y_train_p)
y_hat = model_to_eval.predict(X_test_preprocessed[columns_here])
RMSE(y_test, y_hat)

from sklearn.linear_model import (
    OrthogonalMatchingPursuit,
    ARDRegression,
    PoissonRegressor,
    TweedieRegressor,
)

model_to_eval = PoissonRegressor()
model_to_eval.fit(X_train_p[columns_here], y_train_p)
y_hat = model_to_eval.predict(X_test_preprocessed[columns_here])
RMSE(y_test, y_hat)

# @title **5-fold Cross Validation**
get_mean_score(model_to_eval, train.drop("y", axis=1), train[["y"]])

"""### **IV. 2. Fine Tuning**
Here we can compare models by doing 5-fold cross validation
"""

# @title **Utils Functions**

# get a list of models to evaluate
def get_models(parameter):
    models = dict()
    if parameter == "n_estimators":
        models["10"] = ExtraTreesRegressor(n_estimators=10)
        models["50"] = ExtraTreesRegressor(n_estimators=50)
        models["100"] = ExtraTreesRegressor(n_estimators=100)
        models["500"] = ExtraTreesRegressor(n_estimators=500)
        models["1000"] = ExtraTreesRegressor(n_estimators=1000)
        models["5000"] = ExtraTreesRegressor(n_estimators=5000)
    elif parameter == "max_features":
        for i in range(1, 21):
            models[str(i)] = ExtraTreesRegressor(max_features=i)
    elif parameter == "min_samples_split":
        for i in range(2, 15):
            models[str(i)] = ExtraTreesRegressor(min_samples_split=i)

    return models


# evaluate a given model using cross-validation
def evaluate_model(model, data):
    scores = train_and_evaluate(model, data.drop("y", axis=1), data[["y"]])
    return scores


# get the models to evaluate
def plot_results(parameter: str):
    models = get_models(parameter)
    # evaluate the models and store results
    results, names = list(), list()
    for name, model in models.items():
        scores = evaluate_model(model, train)
        results.append(scores)
        names.append(name)
        logger.info(">%s %.3f (%.3f)" % (name, np.mean(scores), np.std(scores)))
    # plot model performance for comparison
    pyplot.boxplot(results, labels=names, showmeans=True)
    pyplot.show()


plot_results("max_features")

plot_results("min_samples_split")

"""### **IV. 3. Train chosen model**
Here we train the chosen model
"""

model = ExtraTreesRegressor()  # @param
model.fit(X, y)
# @markdown Output model listed as the variable `model`

"""# **V. Deployment on test data**
Here you can deploy your model on the test data
"""

# @title **Make predictions**
# @markdown The predictions are stored in the variable `y_pred`
X_pred = preprocess(test, train)
y_pred = model.predict(X_pred)

# @title **Save predictions**
# @markdown Please make sure you create the directory **Predictions** in your **BASE_PATH**
time = str(datetime.now())
now = time[:16].replace(" ", "_").replace("-", "_").replace(":", "h")
# @markdown Enter a filename
# @markdown > If you don't choose a filename, the predictions are saved with a timestamp in the directory **Predictions** of your **BASE_PATH**.

filename = ""  # @param {type:"string"}
if filename == "":
    filename = f"{BASE_PATH}/Predictions/predictions_{now}"
pd.Series(y_pred).to_csv(f"{filename}.csv")
