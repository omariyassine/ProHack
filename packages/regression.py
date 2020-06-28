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
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback

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


"""# **II. Data Preprocesing**
All data preprocessing on data should be done here (filtering, NaN droping and filling, cleaning functions...)
1. For each cleaning or preprocessing, create a function
2.  Then add it to the prprocess function
"""

# @title **Utils Functions and Constants**
# @markdown Here you can store your utils functions.

# @markdown - `compute_info_value` : compute information value of column against target
# @markdown - `GENDER_COLS_COUPLES` : couples of male/female column

GENDER_COLS_COUPLES = [
    (
        "Estimated_gross_galactic_income_per_capita_female",
        "Estimated_gross_galactic_income_per_capita_male",
    ),
    (
        "Expected_years_of_education_female_galactic_years",
        "Expected_years_of_education_male_galactic_years",
    ),
    (
        "Expected_years_of_education_female_galactic_years",
        "Expected_years_of_education_male_galactic_years",
    ),
    (
        "Intergalactic_Development_Index_IDI_female",
        "Intergalactic_Development_Index_IDI_male",
    ),
    (
        "Intergalactic_Development_Index_IDI_female_Rank",
        "Intergalactic_Development_Index_IDI_male_Rank",
    ),
    (
        "Labour_force_participation_rate__ages_15_and_older_female",
        "Labour_force_participation_rate__ages_15_and_older_male",
    ),
    (
        "Labour_force_participation_rate__ages_15_and_older_female",
        "Labour_force_participation_rate__ages_15_and_older_male",
    ),
    (
        "Mean_years_of_education_female_galactic_years",
        "Mean_years_of_education_male_galactic_years",
    ),
    (
        "Population_with_at_least_some_secondary_education_female__ages_25_and_older",
        "Population_with_at_least_some_secondary_education_male__ages_25_and_older",
    ),
]


max_bin = 20
force_bin = 3


def mono_bin(Y, X, n=max_bin):

    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[["X", "Y"]][df1.X.isnull()]
    notmiss = df1[["X", "Y"]][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame(
                {"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)}
            )
            d2 = d1.groupby("Bucket", as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
            n = n - 1
        except Exception:
            n = n - 1

    if len(d2) == 1:
        n = force_bin
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1] - (bins[1] / 2)
        d1 = pd.DataFrame(
            {
                "X": notmiss.X,
                "Y": notmiss.Y,
                "Bucket": pd.cut(notmiss.X, np.unique(bins), include_lowest=True),
            }
        )
        d2 = d1.groupby("Bucket", as_index=True)

    d3 = pd.DataFrame({}, index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3 = d3.reset_index(drop=True)

    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({"MIN_VALUE": np.nan}, index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4, ignore_index=True)

    d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT / d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT - d3.DIST_NON_EVENT) * np.log(
        d3.DIST_EVENT / d3.DIST_NON_EVENT
    )
    d3["VAR_NAME"] = "VAR"
    d3 = d3[
        [
            "VAR_NAME",
            "MIN_VALUE",
            "MAX_VALUE",
            "COUNT",
            "EVENT",
            "EVENT_RATE",
            "NONEVENT",
            "NON_EVENT_RATE",
            "DIST_EVENT",
            "DIST_NON_EVENT",
            "WOE",
            "IV",
        ]
    ]
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()

    return d3


def char_bin(Y, X):

    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[["X", "Y"]][df1.X.isnull()]
    notmiss = df1[["X", "Y"]][df1.X.notnull()]
    df2 = notmiss.groupby("X", as_index=True)

    d3 = pd.DataFrame({}, index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y

    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({"MIN_VALUE": np.nan}, index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4, ignore_index=True)

    d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT / d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT - d3.DIST_NON_EVENT) * np.log(
        d3.DIST_EVENT / d3.DIST_NON_EVENT
    )
    d3["VAR_NAME"] = "VAR"
    d3 = d3[
        [
            "VAR_NAME",
            "MIN_VALUE",
            "MAX_VALUE",
            "COUNT",
            "EVENT",
            "EVENT_RATE",
            "NONEVENT",
            "NON_EVENT_RATE",
            "DIST_EVENT",
            "DIST_NON_EVENT",
            "WOE",
            "IV",
        ]
    ]
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)

    return d3


def compute_info_value(df1, target):

    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r"\((.*?)\).*$").search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]

    x = df1.dtypes.index
    count = -1

    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                conv = mono_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1

            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv, ignore_index=True)

    iv = pd.DataFrame({"IV": iv_df.groupby("VAR_NAME").IV.max()})
    iv = iv.reset_index()
    return (iv_df, iv)


# @title **Preprocessing Functions**
# @markdown Here you can create your preprocssing functions


def clean_column_names(data):
    data.columns
    data.columns = [
        (
            "_".join(col.encode("ascii", errors="ignore").decode().split(" "))
            .replace(",", "")
            .replace("(", "")
            .replace("[", "")
            .replace("]", "")
            .replace(")", "")
            .replace("%", "")
        )
        for col in data.columns
    ]
    return data


def remove_year(data, train_data):
    data = data.drop(columns=["galactic_year"])
    return data


def remove_constellation_and_year(data, train_data):
    data = data.drop(columns=["galaxy", "galactic_year"])
    return data


def fill_na(data, train_data):
    mean = data.mean()
    data = data.fillna(mean)
    return data


def create_na_columns(data, train_data):
    for col in data.columns:
        if data[col].isna().any():
            data[f"{col}_isna"] = (data[col].isna()).astype("int")
    return data


def drop_sparse(data, train_data):
    na_mean = train_data.isna().mean()
    is_sparse = na_mean[na_mean > 0.5]
    data = data.drop(is_sparse.index.values, axis=1)
    return data


def dummify_galaxy(data, train_data):
    data = pd.get_dummies(data)
    return data


def label_encode_galaxy(data, data_train):
    le = preprocessing.LabelEncoder()
    le.fit(data_train["galaxy"])
    data["galaxy"] = le.transform(data["galaxy"])
    return data


def one_hot_galaxy(data, data_train):
    data = pd.get_dummies(data)
    galaxies_not_in_test = [
        "Andromeda XII",
        "Andromeda XIX[60]",
        "Andromeda XVIII[60]",
        "Andromeda XXII[57]",
        "Andromeda XXIV",
        "Hercules Dwarf",
        "NGC 5253",
        "Triangulum Galaxy (M33)",
        "Tucana Dwarf",
    ]
    galaxies_not_in_test = [f"galaxy_{col}" for col in galaxies_not_in_test]
    data["galaxy_not_in_test"] = 0
    if set(galaxies_not_in_test).issubset(data.columns):
        data["galaxy_not_in_test"] = (
            data[galaxies_not_in_test].sum(axis=1) > 0
        ).astype("int")
        data = data.drop([*galaxies_not_in_test], axis=1)
    return data


def create_period_column(data, train_data):
    start = data["galactic_year"].min()
    end = data["galactic_year"].max()
    pas = 2510
    m = start
    Period = [start]
    while m < end:
        m += pas
        Period.append(m)

    def get_period(x):
        p = 0
        while x >= Period[p]:
            p += 1
        return p

    data["period"] = data["galactic_year"].apply(get_period)

    return data


def fill_na_period_galaxy(data, train_data):
    data = data.fillna(data.groupby(["galaxy", "period"]).transform("mean"))
    data = data.fillna(data.groupby(["period"]).transform("mean"))
    data = data.fillna(data.groupby(["galaxy"]).transform("mean"))
    data = data.fillna(data.mean())
    return data


def compute_dwarf_planet(data, train_data):
    data["dwarf_planet"] = (
        data["galaxy"].str.lower().str.contains("dwarf").astype("int")
    )
    return data


def compute_constellation(data, train_data):
    data["constellation"] = data["galaxy"].replace(mapping_constellation)
    return data


def one_hot_constellation(data, data_train):
    data = pd.get_dummies(data)
    constellations_not_in_test = ["constellation_Hercules"]
    data["constellation_not_in_test"] = 0
    if set(constellations_not_in_test).issubset(data.columns):
        data["constellation_not_in_test"] = (
            data[constellations_not_in_test].sum(axis=1) > 0
        ).astype("int")
        data = data.drop([*constellations_not_in_test], axis=1)
    return data


def one_hot_galaxy_constellation(data, data_train):
    data = pd.get_dummies(data)
    constellations_not_in_test = ["constellation_Hercules"]
    data["constellation_not_in_test"] = 0
    if set(constellations_not_in_test).issubset(data.columns):
        data["constellation_not_in_test"] = (
            data[constellations_not_in_test].sum(axis=1) > 0
        ).astype("int")
        data = data.drop([*constellations_not_in_test], axis=1)
    galaxies_not_in_test = [
        "Andromeda XII",
        "Andromeda XIX[60]",
        "Andromeda XVIII[60]",
        "Andromeda XXII[57]",
        "Andromeda XXIV",
        "Hercules Dwarf",
        "NGC 5253",
        "Triangulum Galaxy (M33)",
        "Tucana Dwarf",
    ]
    galaxies_not_in_test = [f"galaxy_{col}" for col in galaxies_not_in_test]
    data["galaxy_not_in_test"] = 0
    if set(galaxies_not_in_test).issubset(data.columns):
        data["galaxy_not_in_test"] = (
            data[galaxies_not_in_test].sum(axis=1) > 0
        ).astype("int")
        data = data.drop([*galaxies_not_in_test], axis=1)
    return data


def remove_weight_of_evidence(data, train_data):
    final_iv, IV = compute_info_value(train_data, train_data["y"])
    to_drop = IV[IV["IV"] < 0.002]
    to_drop = to_drop.VAR_NAME.to_list()
    data = data.drop(to_drop, axis=1)
    return data


def log_scale_data(data, train_data):
    col_log = data.select_dtypes("float64").columns
    data[col_log] = data[col_log].abs().apply(lambda x: np.log(1 + x))
    return data


def get_gender_ratio(data, train_data, gender_columns_couples=GENDER_COLS_COUPLES):
    for (male, female) in gender_columns_couples:
        new_col_name = female.replace("_female", "_gender_ration")
        data[new_col_name] = data[female] / data[male]
    return data


def remove_negative_values(data, train_data):
    data = data[(data >= 0).all(1)]
    return data


# @title **The Function preprocess**
# @markdown Here you can add the created functions to the function `preprocess`


def preprocess(X_test, X_train, y_train, target="y"):
    """This method will be deployed on data to do the preprocesing

   Args:
    data (pd.DataFrame): The data on which the preprocessing is applied
    train_data (pd.DataFrame): The train data used to train encoders

  Returns:
    pd.DataFrame: The preprocessed data
  """
    data = X_test.copy()
    train_data = pd.concat([X_train, y_train], axis=1)
    data, train_data = clean_column_names(data), clean_column_names(train_data)
    data, train_data = (
        get_gender_ratio(data, train_data),
        get_gender_ratio(train_data, train_data),
    )
    data, train_data = (
        drop_sparse(data, train_data),
        drop_sparse(train_data, train_data),
    )
    data, train_data = (
        create_period_column(data, train_data),
        create_period_column(train_data, train_data),
    )
    data, train_data = (
        create_na_columns(data, train_data),
        create_na_columns(train_data, train_data),
    )
    data, train_data = (
        fill_na_period_galaxy(data, train_data),
        fill_na_period_galaxy(train_data, train_data),
    )
    data, train_data = (
        remove_year(data, train_data),
        remove_year(train_data, train_data),
    )
    data, train_data = (
        compute_dwarf_planet(data, train_data),
        compute_dwarf_planet(train_data, train_data),
    )
    data, train_data = (
        compute_constellation(data, train_data),
        compute_constellation(train_data, train_data),
    )
    data, train_data = (
        one_hot_galaxy_constellation(data, train_data),
        one_hot_galaxy_constellation(train_data, train_data),
    )
    data, train_data = data, remove_negative_values(train_data, train_data)
    data, train_data = clean_column_names(data), clean_column_names(train_data)

    return data, train_data.drop(target, axis=1), train_data[target]


def train_preprocessing(data, target="y"):
    _, X, y = preprocess(data, data, target)
    X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    return X, y


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
