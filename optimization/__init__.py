"""# **VI. Optimization**
Here you can create your optimization model
"""
import numpy as np
import pandas as pd
from scipy import optimize
import logging
from datetime import datetime

from optimization.cost import cost_function
from optimization.constraint import under_threshold, total_energy
from config.own import MAX_ENERGY_PER_GALAXY

logger = logging.getLogger(__file__)


def import_regression_result():
    """Create optimization dataframe

    - Create `opt_df` a dataframe from the prediction
    - Create the column `potential_increase` in `opt_df` using the formula `-np.log(Index+0.01)+3`
    """
    y_pred = pd.read_csv("data/output/predictions.csv")["0"].values
    test = pd.read_csv("data/input/test.csv")
    opt_df = pd.DataFrame(y_pred, columns=["pred"]).reset_index()
    opt_df["potential_increase"] = -np.log(opt_df.pred + 0.01) + 3
    opt_df["existence_expectancy_index"] = test["existence expectancy index"]
    return opt_df


def get_optimal_ditrib():
    opt_df = import_regression_result()
    x0 = np.array([0] * len(opt_df))
    result = optimize.minimize(
        lambda x: cost_function(
            x, potential_increase=opt_df.potential_increase, pred=opt_df.pred
        ),
        x0,
        method="SLSQP",
        bounds=[(0, MAX_ENERGY_PER_GALAXY) for i in range(len(opt_df))],
        constraints=[
            {
                "fun": lambda x: under_threshold(
                    x, existency_index=opt_df.existence_expectancy_index
                ),
                "type": "ineq",
            },
            {"fun": total_energy, "type": "ineq"},
        ],
        options={"disp": True},
    )

    opt_df["pred_opt"] = result["x"]
    logger.info(opt_df.head())

    # Save final results
    final_df = opt_df[["index", "pred", "pred_opt"]]
    final_df.columns = ["index", "pred", "opt_pred"]
    time = str(datetime.now())
    now = time[:16].replace(" ", "_").replace("-", "_").replace(":", "h")
    filename = "data/output/output"
    f"data/Predictions/output_{now}"
    final_df.to_csv(f"{filename}.csv", index=False)
