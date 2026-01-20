### Imports and installs ###

# Installs: pip install pandas numpy scikit-learn joblib pyarrow scikit-learn-intelex

from sklearnex import patch_sklearn
patch_sklearn()

import pandas as pd
import numpy as np
import itertools
import pickle as pkl
from sklearn.ensemble import RandomForestRegressor

from helpers import expanding_window_split, r_squared_oos

# Set seed
seed = 42
np.random.seed(seed)

### Load data ###
path = "/work/Data for Thesis/finalized_true.parquet"
df = pd.read_parquet(path)

### Load column lists ###
info_cols = pd.read_csv("/work/Data for Thesis/info_cols.txt", header=None)[0].tolist()
gkx_cols = pd.read_csv("/work/Data for Thesis/gkx_cols.txt", header=None)[0].tolist()
insider_cols = pd.read_csv("/work/Data for Thesis/insider_cols.txt", header=None)[0].tolist()

target_col = "ret_excess"

# Continuous GKX + insider continuous
feature_cols = ([c for c in gkx_cols])

### Hyperparameters ###
n_estimators = [300]
max_depth = [1, 2, 3]
max_features = [3, 5, 10, 20, 30, 50]

# Tuning grid
tuning_grid = list(itertools.product(n_estimators, max_depth, max_features))

### Random Forest Function ###
def random_forest(df, train_size, val_size, test_size, step_size,
                  feature_cols, target_col, tuning_grid,
                  seed=42, n_jobs=-1, verbose=0, retrain=False):

    results = []
    predictions = []

    for split, (train, val, test) in enumerate(
        expanding_window_split(df, train_size, val_size, test_size, step_size)
    ):

        print(f"\n=== Split {split+1} ===")

        # Extract arrays
        X_train = train[feature_cols].to_numpy("float32")
        y_train = train[target_col].to_numpy("float32")

        X_val = val[feature_cols].to_numpy("float32")
        y_val = val[target_col].to_numpy("float32")

        X_test = test[feature_cols].to_numpy("float32")
        y_test = test[target_col].to_numpy("float32")

        # Track best hyperparameters
        best_mse = np.inf
        best_params = None
        best_model = None

        # Hyperparameter tuning
        for n_estimators, max_depth, max_features in tuning_grid:
            print(f"Testing params: n_estimators={n_estimators}, max_depth={max_depth}, max_features={max_features}")

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                random_state=seed + split,
                bootstrap=True,
                oob_score=False,
                min_samples_leaf=20,
                min_samples_split=40,
                n_jobs=n_jobs,
                verbose=verbose
            )

            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)
            
            mse = np.mean((y_val - y_val_pred) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_params = (n_estimators, max_depth, max_features)
                best_model = model

        print(f"Best parameters for split {split+1}: {best_params}, validation MSE={best_mse:.6f}")

        # Optionally retrain on train + val
        if retrain:
            n_estimators, max_depth, max_features = best_params

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                random_state=seed + split,
                bootstrap=True,
                oob_score=False,
                min_samples_leaf=20,
                min_samples_split=40,
                n_jobs=n_jobs,
                verbose=verbose
            )

            X_combined = np.concatenate([X_train, X_val])
            y_combined = np.concatenate([y_train, y_val])

            model.fit(X_combined, y_combined)
            model_to_test = model
        else:
            model_to_test = best_model
        
        # Save model to pickle
        model_path = f"rf_model_split_{split+1}.pkl"
        with open(model_path, "wb") as f:
            pkl.dump(model_to_test, f)

        print(f"Saved model for split {split+1} to {model_path}")
        
        # Test prediction
        y_test_pred = model_to_test.predict(X_test)
        r2_oos_split = r_squared_oos(y_test, y_test_pred)

        print(f"Split {split+1} test R2: {r2_oos_split:.4%}")

        # Store predictions for this split
        pred_df = test[["month", "cik", "permno", "ret_excess", "prc", "shrout", "mktcap_lag"]].copy()
        pred_df["pred_ret_excess"] = y_test_pred
        predictions.append(pred_df)

        # Store summary results
        results.append({
            "train_end": train["month"].max(),
            "val_end": val["month"].max(),
            "test_end": test["month"].max(),
            "val_mse": best_mse,
            "best_n_estimators": best_params[0],
            "best_max_depth": best_params[1],
            "best_max_features": best_params[2],
            "test_r2": r2_oos_split,
        })

    # Combine all predictions
    prediction_df = pd.concat(predictions, axis=0)

    # Compute global OOS R2
    overall_r2 = r_squared_oos(prediction_df["ret_excess"], prediction_df["pred_ret_excess"])
    print(f"\nOverall test R-squared across all splits: {overall_r2:.4%}")

    # Add overall R2 to last entry
    results[-1]["overall_r2"] = overall_r2

    results_df = pd.DataFrame(results)

    return results_df, prediction_df


### RUN RF ###
results_df, prediction_df = random_forest(
    df,
    train_size=60,
    val_size=36,
    test_size=12,
    step_size=12,
    feature_cols=feature_cols,
    target_col=target_col,
    tuning_grid=tuning_grid,
    seed=seed,
    retrain=True,
    n_jobs=-1,
    verbose=10,
)