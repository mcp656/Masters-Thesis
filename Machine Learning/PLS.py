### Imports and installs ###

from sklearnex import patch_sklearn
patch_sklearn()

import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed
import pickle as pkl

from helpers import expanding_window_split, r_squared_oos

seed = 42
np.random.seed(seed)

### Load data ###
path = "/work/Data for Thesis/with_insider.parquet"
df = pd.read_parquet(path)

### Load column lists ###
info_cols = pd.read_csv("/work/Data for Thesis/info_cols.txt", header=None)[0].tolist()
gkx_cols = pd.read_csv("/work/Data for Thesis/gkx_cols.txt", header=None)[0].tolist()
insider_cols = pd.read_csv("/work/Data for Thesis/insider_cols.txt", header=None)[0].tolist()

target_col = "ret_excess"

### Features: GKX only ###
feature_cols = [c for c in gkx_cols if not c.startswith("sic")]

### Hyperparameter grid ###
n_components = range(1, 25, 1)
tuning_grid = n_components


### PLS MODEL ###
def pls_model(df, train_size, val_size, test_size, step_size,
              feature_cols, target_col, tuning_grid,
              seed=42, retrain=False, n_jobs=-1, verbose=0):

    results = []
    predictions = []

    for split, (train, val, test) in enumerate(
        expanding_window_split(df, train_size, val_size, test_size, step_size)
    ):

        print(f"\nSplit {split + 1}")

        X_train = train[feature_cols]
        X_val = val[feature_cols]
        X_test = test[feature_cols]

        y_train = train[target_col]
        y_val = val[target_col]
        y_test = test[target_col]

        def pls_fit_score(n_comp):
            pls = PLSRegression(n_components=n_comp, scale=False)
            pls.fit(X_train, y_train)

            y_val_pred = pls.predict(X_val)
            mse = mean_squared_error(y_val, y_val_pred)

            return mse, pls, n_comp

        parallel_results = Parallel(
            n_jobs=n_jobs, backend="loky", verbose=verbose
        )(
            delayed(pls_fit_score)(n_comp) for n_comp in tuning_grid
        )

        best_mse, best_pls, best_n_components = min(
            parallel_results, key=lambda x: x[0]
        )

        print(f"Best n_components = {best_n_components}")

        if retrain:
            X_combined = pd.concat([X_train, X_val])
            y_combined = pd.concat([y_train, y_val])

            pls_final = PLSRegression(n_components=best_n_components, scale=False)
            pls_final.fit(X_combined, y_combined)

            model_pls = pls_final
        else:
            model_pls = best_pls

        y_test_pred_np = model_pls.predict(X_test).flatten()
        y_test_pred = pd.Series(y_test_pred_np, index=y_test.index)

        r2_oos = r_squared_oos(y_test, y_test_pred_np)

        print(f"Validation MSE: {best_mse:.6f}")
        print(f"Test R²: {r2_oos:.4%}")

        pred_panel = test[[
            "month", "cik", "permno", "ret_excess",
            "prc", "shrout", "mktcap_lag"
        ]].copy()
        pred_panel["pred_ret_excess"] = y_test_pred.values
        predictions.append(pred_panel)

        results.append({
            "split": split + 1,
            "train_end": train["month"].max(),
            "val_end": val["month"].max(),
            "test_end": test["month"].max(),
            "val_mse": best_mse,
            "best_n_components": best_n_components,
            "test_r2": r2_oos
        })

    prediction_df = pd.concat(predictions, axis=0)

    full_period_r2 = r_squared_oos(
        prediction_df["ret_excess"],
        prediction_df["pred_ret_excess"]
    )
    print(f"\nFull-period Test R² OOS: {full_period_r2:.4%}")

    results_df = pd.DataFrame(results)

    return results_df, prediction_df


### Run PLS ###
results_df, prediction_df = pls_model(
    df=df,
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
    verbose=50
)