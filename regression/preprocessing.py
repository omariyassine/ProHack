import pandas as pd
import numpy as np
from sklearn import preprocessing
import logging

from regression.utils import (
    MAPPING_CONSTELLATIONS,
    GENDER_COLS_COUPLES,
    compute_info_value,
)

logger = logging.getLogger(__name__)


def import_data():
    """Import raw data

    Returns:
        pd.DataFrame, pd.DataFrame: Tables of train and test data
        Name of imported variables:
            - train (pd.DataFrame) : The table of train data
            - test (pd.DataFrame) : The table of test data
            - mapping_constellation (dict) : Mapping galaxy => constellation
    """

    BASE_PATH = "./data/input"
    train = pd.read_csv(f"{BASE_PATH}/train.csv")
    test = pd.read_csv(f"{BASE_PATH}/test.csv")
    logger.info(
        f"\nShape of train data {train.shape} \n" f"Shape of test data {test.shape}"
    )
    return train, test


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
    data["constellation"] = data["galaxy"].replace(MAPPING_CONSTELLATIONS)
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


def get_gender_ratio(data, train_data):
    for (male, female) in GENDER_COLS_COUPLES:
        new_col_name = female.replace("_female", "_gender_ration")
        data[new_col_name] = data[female] / data[male]
    return data


def remove_negative_values(data, train_data):
    data = data[(data >= 0).all(1)]
    return data
