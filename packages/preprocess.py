"""The Function preprocess

Here you can add the created functions to the function `preprocess`

All data preprocessing on data should be done here (filtering, NaN droping and filling, cleaning functions...)
    1. For each cleaning or preprocessing, create a function in `preprocessing.py`
    2.  Then add it to the prprocess function
"""


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
