"""# **V. Deployment on test data**
Here you can deploy your model on the test data
"""
import pandas as pd
from datetime import datetime
import logging
from sklearn.ensemble import ExtraTreesRegressor

from regression.evaluation import get_mean_score
from regression.preprocess import preprocess
from regression.preprocessing import import_data

logger = logging.getLogger(__name__)

BASE_PATH = "data"

# evaluate a given model using cross-validation
def evaluate_model():
    train = import_data()[0]
    model = ExtraTreesRegressor()
    scores = get_mean_score(model, train.drop("y", axis=1), train[["y"]])[-1]
    time = str(datetime.now())
    now = time[:16].replace(" ", "_").replace("-", "_").replace(":", "h")
    filename_time = f"{BASE_PATH}/output/evaluation/model_{type(model).__name__}_{now}"
    pd.DataFrame(scores).to_csv(filename_time)
    return scores


def deploy_model():
    test = import_data()[1]
    model = ExtraTreesRegressor()
    X_pred = preprocess(test, train)
    y_pred = model.predict(X_pred)
    time = str(datetime.now())
    now = time[:16].replace(" ", "_").replace("-", "_").replace(":", "h")
    filename = f"{BASE_PATH}/output/predictions"
    filename_time = f"{BASE_PATH}/output/history/predictions_{now}"
    pd.Series(y_pred).to_csv(f"{filename}.csv")
    pd.Series(y_pred).to_csv(f"{filename_time}.csv")
