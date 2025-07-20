
from logging import getLogger
import pandas as pd
import os
from compare_agents import _ca_logger

def get_DrinkLess_all_day():
    # 1. Define Feather file paths relative to this script
    base = os.path.dirname(__file__)
    paths = {
        "a": os.path.join(base, "data/DrinkLess/FINAL_Dataset_A.feather"),
        "c": os.path.join(base, "data/DrinkLess/FINAL_Dataset_C.feather"),
        "d": os.path.join(base, "data/DrinkLess/FINAL_Dataset_D.feather"),
    }

    # 2. Read Feather files into a dict
    #    Feather stores UTF-8 text and full columnar metadata
    data = {k: pd.read_feather(v) for k, v in paths.items()}

    # 3. Parse and convert date/time columns
    #    Convert screen_view to London timezone
    data["c"]["screen_view"] = (
        pd.to_datetime(data["c"]["screen_view"], unit="s", origin="unix", utc=True)
          .dt.tz_convert("Europe/London")
    )
    #    Convert Download_at to date
    data["c"]["Download_at"] = pd.to_datetime(
        data["c"]["Download_at"], unit="D", origin="unix"
    ).dt.date

    #    Parse downloaded_at from d to London timezone
    data["d"]["downloaded_at"] = (
        pd.to_datetime(
            data["d"]["downloaded_at"],
            format="%d/%m/%Y %H:%M",
            dayfirst=True,
        )
        .dt.tz_localize("Europe/London")
    )

    # 4. Extract unique background info from D
    bg = data["d"].drop_duplicates(subset="ID")

    # 5. General mismatch-finder function
    def find_mismatches(df, suffix, cols):
        m = df.merge(bg, on="ID", suffixes=(f"_{suffix}", "_bg"))
        mask = pd.Series(False, index=m.index)
        for c in cols:
            mask |= (m[f"{c}_{suffix}"] != m[f"{c}_bg"])
        return m[mask]

    # 6. Check mismatches in A and C
    mismatches_a = find_mismatches(
        data["a"], "a",
        ["age", "AUDIT_score", "gender", "employment_type", "AUDIT_score_cat"]
    )
    mismatches_c = find_mismatches(
        data["c"], "c",
        ["age", "AUDIT_score", "gender", "employment_type",
         "AUDIT_score_cat", "trialVersion"]
    )

    if mismatches_a.empty and mismatches_c.empty:
        print("No mismatches in Dataset A or C.")
    else:
        print("Mismatches found in A or C:")
        print(mismatches_a)
        print(mismatches_c)

    # 7. Simplify A & C
    cols_a = [
        "ID", "message", "subversion", "days_since_download", "treatment",
        "primary_outcome", "before_8pm", "outcome_24hour", "habituation",
        "already_engaged", "after_9pm_day_before", "prob_A", "prob_B"
    ]
    a_simple = data["a"][cols_a]
    c_simple = data["c"][["ID", "screen_view", "days_since_download", "Download_at"]]

    # 8. Count 8–9 pm screen-views
    views_8pm = (
        c_simple[c_simple["screen_view"].dt.hour == 20]
        .groupby(["ID", "days_since_download"])  
        .size()
        .reset_index(name="screen_views_8_to_9pm")
    )

    # 9. Merge with A, fill missing with 0
    all_day = (
        a_simple[["ID", "days_since_download", "before_8pm", 
                  "after_9pm_day_before", "treatment"]]
        .merge(views_8pm, on=["ID", "days_since_download"], how="left")
        .fillna({"screen_views_8_to_9pm": 0})
    )

    # 10. Alignment check against primary_outcome
    merged = a_simple.merge(all_day, on=["ID", "days_since_download"])
    bad = merged.query(
        "(primary_outcome == 1 and screen_views_8_to_9pm == 0) or "
        "(primary_outcome == 0 and screen_views_8_to_9pm > 0)"
    )
    if bad.empty:
        print("Dataset A and C align.")
    else:
        print("Alignment issues:")
        print(bad)
        
    return all_day

def get_DrinkLess():
    all_day = get_DrinkLess_all_day()
    print("Loaded DrinkLess data for outcome simulation.")
    
    result = {
        "context": all_day[[
            "ID", "days_since_download", "before_8pm", "after_9pm_day_before"
            ]],
        "action": all_day["treatment"] + 1,
        "reward": all_day["screen_views_8_to_9pm"],
        }
    return result

from bart_playground.bandit.sim_util import Scenario
import numpy as np

# We define a scenario for the DrinkLess dataset only for the simulation purposes.
class DrinkLessScenario(Scenario):
    def __init__(self, random_generator=None):
        # Load the DrinkLess data
        self.all_day_data = get_DrinkLess_all_day()
        
        # Extract unique context combinations for covariates
        context_cols = ["days_since_download", "before_8pm", "after_9pm_day_before"]
        self.contexts = self.all_day_data[context_cols].drop_duplicates().reset_index(drop=True)
        
        P = len(context_cols)  # Number of features
        K = len(self.all_day_data["treatment"].unique())  # Number of treatment arms
        
        super().__init__(P, K, sigma2=0.0, random_generator=random_generator)
        self._cursor = 0
    
    def init_params(self):
        # Shuffle the contexts
        self.contexts = self.contexts.sample(
            frac=1, random_state=self.rng.integers(0, 2**31 - 1)
            )
        self._cursor = 0
    
    def generate_covariates(self):
        if self._cursor >= len(self.contexts):
            self._cursor = 0  # Reset if we've exhausted all contexts
        
        context = self.contexts.iloc[self._cursor]
        self._cursor += 1
        
        # Return as numpy array: [days_since_download, before_8pm, after_9pm_day_before]
        return np.array([
            context["days_since_download"],
            context["before_8pm"], 
            context["after_9pm_day_before"]
        ], dtype=np.float32)
    
    def reward_function(self, x):
        """
        Given a feature vector x, compute outcome probabilities for each treatment arm.
        Follows the R logic: outcome_DL = function(x, arm) {...}
        """
        days = round(x[0])  # days_since_download
        before_8pm = bool(x[1])
        after_9pm_day_before = bool(x[2])
        
        outcome_means = np.zeros(self.K)
        rewards = np.zeros(self.K)
        
        for arm in range(self.K):
            # Filter data based on context and treatment (arm is 0-indexed, treatment is 1-indexed)
            filtered_data = self.all_day_data[
                (self.all_day_data["days_since_download"] == days) &
                (self.all_day_data["before_8pm"] == before_8pm) &
                (self.all_day_data["after_9pm_day_before"] == after_9pm_day_before) &
                (self.all_day_data["treatment"] == arm)  # treatment is 0-indexed
            ]
            
            if len(filtered_data) > 0:
                # Probability of having screen views > 0 
                prob = (filtered_data["screen_views_8_to_9pm"] > 0).mean()
                outcome_means[arm] = prob
                # Generate binary reward based on probability
                rewards[arm] = self.rng.binomial(1, prob)
            else:
                # If no data for this combination, use neutral values
                prob = 0.5
                outcome_means[arm] = prob
                rewards[arm] = self.rng.binomial(1, prob)

                _ca_logger.warning(
                    f"No data for context {x} and treatment arm {arm + 1}. "
                    "Ensure the context matches the dataset."
                )
        
        return {"outcome_mean": outcome_means, "reward": rewards}
    
    @property
    def max_draws(self):
        return len(self.contexts)
