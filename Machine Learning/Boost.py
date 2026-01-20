### Imports and installs ###

# Installs
# pip install pandas numpy scikit-learn pyarrow scikit-learn-intelex joblib

#from sklearnex import patch_sklearn
#patch_sklearn()

import pandas as pd
import numpy as np
import itertools
from sklearn.ensemble import HistGradientBoostingRegressor
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

# Feature cols
feature_cols = [c for c in gkx_cols]

### Set hyperparameters ###
max_iter = np.logspace(np.log10(32 + 1), np.log10(320), 8).astype(int)
max_depth = [1, 2]
learning_rate = [0.01]

# Define grid
tuning_grid = list(itertools.product(max_iter, max_depth, learning_rate))

def need_for_speed(split, X_train, y_train, X_val, y_val, params, seed=seed):
    
    # Unpack hyperparameters
    max_iter, max_depth, learning_rate = params
    
    # Initialize model
    model = HistGradientBoostingRegressor(
        loss='squared_error',
        max_iter=max_iter,
        max_depth=max_depth,
        max_features = 0.055,
        learning_rate=learning_rate,
        validation_fraction=None,
        early_stopping=False,
        random_state=seed + split,
        verbose=0,
    )
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Predict on validation set
    y_pred = model.predict(X_val)
    
    # Calculate MSE
    mse = np.mean((y_val - y_pred)**2)
    
    return mse, model, max_iter, max_depth, learning_rate

### Gradient Boosting Function ###
def gradient_boost(df, train_size, val_size, test_size, step_size,
                   feature_cols, target_col, tuning_grid,
                   loss='squared_error', n_jobs=-1,
                   seed=seed, verbose=0, retrain=False):
    
    # Store predictions and true values
    predictions = []
    
    # Store results
    results = []

    for split, (train, val, test) in enumerate(expanding_window_split(df, train_size, val_size, test_size, step_size)):

        print(f"split: {split+1}")
        
        # Split data into X and y
        X_train, y_train = train[feature_cols].to_numpy(), train[target_col].to_numpy()
        X_val, y_val = val[feature_cols].to_numpy(), val[target_col].to_numpy()
        X_test, y_test = test[feature_cols].to_numpy(), test[target_col].to_numpy()
        
        # Hyperparameter tuning              
        parallel_results = Parallel(n_jobs=n_jobs)(
            delayed(need_for_speed)(
                split, X_train, y_train, X_val, y_val, params
            )
            for params in tuning_grid
        )

        # Select best by smallest MSE
        best_mse, best_model, best_iter, best_max_depth, best_learning_rate = min(
            parallel_results, key=lambda x: x[0]
        )

        # Retrain on train + val if requested
        if retrain:
            best_model_retrain = HistGradientBoostingRegressor(
                loss=loss,
                max_iter=best_iter,
                max_depth=best_max_depth,
                max_features = 0.055,                
                learning_rate=best_learning_rate,
                validation_fraction=None,
                early_stopping=False,
                random_state=seed + split,
                verbose=verbose,
            )

            # Combine train and val sets
            X_combined = np.concatenate([X_train, X_val], axis=0)
            y_combined = np.concatenate([y_train, y_val], axis=0)
            
            # Retrain the cloned model
            best_model_retrain.fit(X_combined, y_combined) 
            
            # Use retrained model for testing
            model_to_test = best_model_retrain 
        else:
            # Use best model from validation for testing
            model_to_test = best_model  
            
        # Test metrics
        y_test_pred_np = model_to_test.predict(X_test)
        r2_oos = r_squared_oos(y_test, y_test_pred_np)

        # Make y_test_pred into a Pandas Series with the same index as test
        y_test_pred = pd.Series(y_test_pred_np, index=test.index)
        
        # Combine predictions with test identifiers and save for later
        prediction_cols = test[["month", "cik", "permno", "ret_excess", "prc", "shrout", "mktcap_lag"]].copy()
        prediction_cols["pred_ret_excess"] = y_test_pred.values
        predictions.append(prediction_cols)

        # Print results for current split
        print(f"Best Hyperparameters: max_iter={best_iter}, max_depth={best_max_depth}, learning_rate={best_learning_rate}")
        print(f"Validation MSE: {best_mse:.4f}")
        print(f"Test R2: {r2_oos:.4f}")

        # Store results
        results.append({
            "train_end": train["month"].max(),
            "val_end": val["month"].max(),
            "test_end": test["month"].max(),
            "val_mse": best_mse,
            "best_max_iter": best_iter,
            "best_max_depth": best_max_depth,
            "best_learning_rate": best_learning_rate,
            "test_r2": r2_oos,
        })
        
    # Combine all test periods into one dataframe
    prediction_df = pd.concat(predictions, axis=0)
    
    # Compute overall test R-squared and store in the last results entry
    overall_r2 = r_squared_oos(prediction_df["ret_excess"], prediction_df["pred_ret_excess"])
    results[-1]["overall_r2"] = overall_r2

    # Print overall R-squared
    print(f"\nOverall Test R-squared across all splits: {overall_r2:.4%}")
    
    # Create DataFrame for results
    results_df = pd.DataFrame(results)  

    return results_df, prediction_df

### Run Gradient Boosting Function ###
results_df, prediction_df = gradient_boost(
    df,
    train_size=60,
    val_size=36,
    test_size=12,
    step_size=12,
    feature_cols=feature_cols,
    target_col=target_col,
    tuning_grid=tuning_grid,
    loss='squared_error',
    seed=seed,
    verbose=100,
    retrain=True,
    n_jobs=-1,
)