### Imports and installs ###

# Installs
# pip install pandas numpy scikit-learn joblib pyarrow scikit-learn-intelex

from sklearnex import patch_sklearn
patch_sklearn()

# Libraries
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed

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

# Continuous GKX features
feature_cols = [c for c in gkx_cols]

# Dummy SIC variables
dummy_cols = [c for c in gkx_cols if c.startswith("sic2_")]

### Hyperparameter grid ###
n_components = range(1, 129, 1)
tuning_grid = n_components

### PCR MODEL ###
def pcr_model(df, train_size, val_size, test_size, step_size,
              feature_cols, dummy_cols, target_col, tuning_grid,
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

        D_train = train[dummy_cols]
        D_val = val[dummy_cols]
        D_test = test[dummy_cols]

        y_train = train[target_col]
        y_val = val[target_col]
        y_test = test[target_col]

        # Inner function for parallel tuning
        def pcr_fit_score(n_comp):
            pca = PCA(n_components=n_comp, random_state=seed)
            Z_train = pca.fit_transform(X_train)
            Z_val = pca.transform(X_val)

            X_train_full = np.hstack([Z_train, D_train.to_numpy()])
            X_val_full = np.hstack([Z_val, D_val.to_numpy()])

            reg = LinearRegression()
            reg.fit(X_train_full, y_train)

            y_val_pred = reg.predict(X_val_full)
            mse = mean_squared_error(y_val, y_val_pred)

            return mse, pca, reg, n_comp

        # Parallel tuning
        parallel_results = Parallel(
            n_jobs=n_jobs, backend="loky", verbose=verbose
        )(
            delayed(pcr_fit_score)(n_comp) for n_comp in tuning_grid
        )

        # Best validation result
        best_mse, best_pca, best_reg, best_n_components = min(
            parallel_results, key=lambda x: x[0]
        )

        print(f"Best n_components = {best_n_components}")

        # Retrain on train+val if requested
        if retrain:
            X_combined = pd.concat([X_train, X_val])
            D_combined = pd.concat([D_train, D_val])
            y_combined = pd.concat([y_train, y_val])

            pca_final = PCA(n_components=best_n_components, random_state=seed)
            Z_combined = pca_final.fit_transform(X_combined)
            X_combined_full = np.hstack([Z_combined, D_combined.to_numpy()])

            reg_final = LinearRegression()
            reg_final.fit(X_combined_full, y_combined)

            model_pca = pca_final
            model_reg = reg_final
        else:
            model_pca = best_pca
            model_reg = best_reg

        # Test prediction
        Z_test = model_pca.transform(X_test)
        X_test_full = np.hstack([Z_test, D_test.to_numpy()])
        y_test_pred_np = model_reg.predict(X_test_full)
        y_test_pred = pd.Series(y_test_pred_np, index=y_test.index)

        # Test R² OOS
        r2_oos = r_squared_oos(y_test, y_test_pred_np)

        print(f"Validation MSE: {best_mse:.6f}")
        print(f"Test R²: {r2_oos:.4%}")

        # Save predictions for full-period R²
        pred_panel = test[[
            "month", "cik", "permno", "ret_excess",
            "prc", "shrout", "mktcap_lag"
        ]].copy()
        pred_panel["pred_ret_excess"] = y_test_pred.values
        predictions.append(pred_panel)

        # Store results for this split
        results.append({
            "split": split + 1,
            "train_end": train["month"].max(),
            "val_end": val["month"].max(),
            "test_end": test["month"].max(),
            "val_mse": best_mse,
            "best_n_components": best_n_components,
            "test_r2": r2_oos
        })

    # Combine all predictions for full-period R²
    prediction_df = pd.concat(predictions, axis=0)

    # Full-period R²
    full_period_r2 = r_squared_oos(
        prediction_df["ret_excess"],
        prediction_df["pred_ret_excess"]
    )
    print(f"\nFull-period Test R² OOS: {full_period_r2:.4%}")

    results_df = pd.DataFrame(results)

    return results_df, prediction_df


### Run PCR ###
results_df, prediction_df = pcr_model(
    df=df,
    train_size=60,
    val_size=36,
    test_size=12,
    step_size=12,
    feature_cols=feature_cols,
    dummy_cols=dummy_cols,
    target_col=target_col,
    tuning_grid=tuning_grid,
    seed=seed,
    retrain=True,
    n_jobs=-1,
    verbose=50
)