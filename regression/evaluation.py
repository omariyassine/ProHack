"""Evaluation function**

You can find here the different function to use for evaluation <br/>
The functions can be used in the following situations <br/>
    - `RMSE(y_true, y_pred)` Get the RMSE of y_true and y_pred
    - `train_and_evaluate(model, X, y)` Get all the scores of the 5-fold cross validation of your model
    - `get_mean_score(model, X, y, verbose=True)` Get the mean and std of the scores of the 5-fold cross validation
    - `compare_models(model, X, y, verbose=False)` Given a list of models, it compares them and return a DataFrame of the results
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from math import sqrt
import logging

from regression.preprocess import preprocess

logger = logging.getLogger(__name__)


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
    return mean, std, scores


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


def train_test_val(train, model_to_eval):
    X_train, X_test, y_train, y_test = train_test_split(
        train.drop("y", axis=1), train[["y"]], test_size=0.33
    )
    X_test_preprocessed, X_train_p, y_train_p = preprocess(X_test, X_train, y_train)
    columns_here = [*set(X_train_p.columns).intersection(X_test_preprocessed.columns)]
    model_to_eval.fit(X_train_p[columns_here], y_train_p)
    y_hat = model_to_eval.predict(X_test_preprocessed[columns_here])
    rmse = RMSE(y_test, y_hat)
    logger.info(f"RMSE {type(model_to_eval).__name__}")
    return rmse
