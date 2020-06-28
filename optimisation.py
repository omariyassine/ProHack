"""# **VI. Optimization**
Here you can create your optimization model
"""

# @title **Create optimization dataframe**
# @markdown - Create `opt_df` a dataframe from the prediction
# @markdown - Create the column `potential_increase` in `opt_df` using the formula `-np.log(Index+0.01)+3`

opt_df = pd.DataFrame(y_pred, columns=["pred"]).reset_index()
opt_df["potential_increase"] = -np.log(opt_df.pred + 0.01) + 3
opt_df["existence_expectancy_index"] = test["existence_expectancy_index"]
opt_df.head()

# @title **Optimization parameters**

TOTAL_ENERGY = 50000  # @param {type:"number"}
MAX_ENERGY_PER_GALAXY = 100  # @param {type:"number"}
THRESHOLD = 0.7  # @param {type:"number"}
MINIMAL_ENERGY_UNDER_THRESHOLD = 0.1  # @param {type:"number"}

# @title **Cost Function and Constraints**
# @markdown We want to maximize `extra_energy * (potential_increase**2) / 1000` , i.e. minimize the cost function ` - extra_energy * (potential_increase**2) / 1000`


def cost_function(
    extra_energy, potential_increase=opt_df.potential_increase.values, pred=opt_df.pred
):
    cost = np.sum(-extra_energy * (potential_increase ** 2) / 1000)
    return cost


def under_threshold(extra_energy, existency_index=opt_df.existence_expectancy_index):
    here = pd.DataFrame(
        {"extra_energy": extra_energy, "existency_index": existency_index,}
    )
    sum_ = here[here.existency_index <= THRESHOLD]["extra_energy"].sum()
    return sum_ - MINIMAL_ENERGY_UNDER_THRESHOLD * TOTAL_ENERGY


def total_energy(extra_energy):
    sum_ = np.sum(extra_energy)
    return TOTAL_ENERGY - sum_


# @title **Optimization**
# @markdown Here we minimize the cost function subject to the constraints
from scipy import optimize

x0 = np.array([0] * len(opt_df))
result = optimize.minimize(
    cost_function,
    x0,
    method="SLSQP",
    bounds=[(0, MAX_ENERGY_PER_GALAXY) for i in range(len(opt_df))],
    constraints=[
        {"fun": total_energy, "type": "ineq"},
        {"fun": under_threshold, "type": "ineq"},
    ],
)

opt_df["pred_opt"] = result["x"]
opt_df.head()

# @title **Save final results**
final_df = opt_df[["index", "pred", "pred_opt"]]
final_df.columns = ["index", "pred", "opt_pred"]
# @markdown Please make sure you create the directory **Predictions** in your **BASE_PATH**
time = str(datetime.now())
now = time[:16].replace(" ", "_").replace("-", "_").replace(":", "h")
# @markdown Enter a filename
# @markdown > If you don't choose a filename, the predictions are saved with a timestamp in the directory **Predictions** of your **BASE_PATH**.

filename = ""  # @param {type:"string"}
if filename == "":
    filename = f"{BASE_PATH}/Predictions/output_{now}"
final_df.to_csv(f"{filename}.csv", index=False)
