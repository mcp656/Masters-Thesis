### Imports and installs ###

# Installs
# pip install pandas numpy scikit-learn joblib pyarrow scikit-learn-intelex

from sklearnex import patch_sklearn
patch_sklearn()

# Libraries
import pandas as pd
import numpy as np
import itertools
from joblib import Parallel, delayed
from sklearn.linear_model import ElasticNet

from helpers import expanding_window_split, r_squared_oos

# Set seed
seed = 42
np.random.seed(seed)

### Load data ###
path = "/work/Data for Thesis/with_outsider.parquet"
df = pd.read_parquet(path)

### Load column lists ###
info_cols = pd.read_csv("/work/Data for Thesis/info_cols.txt", header=None)[0].tolist()
gkx_cols = pd.read_csv("/work/Data for Thesis/gkx_cols.txt", header=None)[0].tolist()
insider_cols = pd.read_csv("/work/Data for Thesis/insider_cols.txt", header=None)[0].tolist()

target_col = "ret_excess"

# Continuous GKX + insider continuous
feature_cols = ([c for c in gkx_cols])

### Hyperparameter grid ###
alphas = np.logspace(-4, 1, 6)

l1_ratios = np.linspace(0, 1, 11) 

tuning_grid = list(itertools.product(alphas, l1_ratios))

### Parallel fit helper ###
def need_for_speed(split, X_train, y_train, X_val, y_val, params, seed):
    
    # Unpack parameters
    alpha, l1_ratio = params
    
    # Initialize and fit model
    model = ElasticNet(
        fit_intercept=True,
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=5000,
        tol=1e-4,
        random_state=seed + split,
        selection='cyclic', # Needs to be 'cyclic' for sklearnex compatibility
    )

    model.fit(X_train, y_train)
    
    # Validate model
    y_val_pred = model.predict(X_val)
    
    # Compute MSE
    mse = np.mean((y_val - y_val_pred) ** 2)
    
    # Return MSE, model, and params
    return mse, model, params

### Elastic Net ###
def elastic_net(df, train_size, val_size, test_size, step_size, feature_cols, target_col, tuning_grid, seed, retrain=False, n_jobs=-1):

    # Results storage
    results = []
    
    # Predictions storage
    predictions = []
    
    # Coefficients storage
    coef_list = []

    for split, (train, val, test) in enumerate(
        expanding_window_split(df, train_size, val_size, test_size, step_size)
    ):
        print(f"\nSplit {split + 1}")

        # Split data
        X_train, y_train = train[feature_cols].to_numpy("float32"), train[target_col].to_numpy("float32")
        X_val, y_val = val[feature_cols].to_numpy("float32"), val[target_col].to_numpy("float32")
        X_test, y_test = test[feature_cols].to_numpy("float32"), test[target_col].to_numpy("float32")

        # Parallel hyperparameter optimization
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(need_for_speed)(split, X_train, y_train, X_val, y_val, params, seed)
            for params in tuning_grid
        )
        results_split = min(results_list, key=lambda x: x[0])

        # Select best combination
        best_mse, best_model, best_params = results_split
        best_alpha, best_l1_ratio = best_params

        # Retrain is True
        if retrain:
            
            # Combine train and val sets
            X_combined = np.concatenate([X_train, X_val], axis=0)
            y_combined = np.concatenate([y_train, y_val], axis=0)
            
            # Initialize model with best hyperparameters
            best_model_retrain = ElasticNet(
                fit_intercept=True,
                alpha=best_alpha,
                l1_ratio=best_l1_ratio,
                max_iter=5000,
                tol=1e-4,
                random_state=seed + split,
                selection='cyclic',
            )
            
            # Fit on combined data
            best_model_retrain.fit(X_combined, y_combined)
            
            # Set model to test
            model_to_test = best_model_retrain
            
        else:
            # Use best model without retraining
            model_to_test = best_model

        # Store coefficients from trained model
        coefs = pd.Series(model_to_test.coef_, index=feature_cols)
        coef_list.append(coefs)
        
        # Find top 10 coefficients by absolute value
        top_ten = coefs.abs().nlargest(10)
        
        # Test metrics
        y_test_pred_np = model_to_test.predict(X_test)
        r2_oos = r_squared_oos(y_test, y_test_pred_np)

        # Make test predictions into a Pandas Series with the same index as test
        y_test_pred = pd.Series(y_test_pred_np, index=test.index)
        
        # Combine predictions with test identifiers and save for later
        prediction_cols = test[["month", "cik", "permno", "ret_excess", "prc", "shrout", "mktcap_lag"]].copy()
        prediction_cols["pred_ret_excess"] = y_test_pred.values
        predictions.append(prediction_cols)
        
        # Print results for current split
        print(f"Best Hyperparameters: alpha={best_alpha}, l1_ratio={best_l1_ratio}")
        print(f"Validation MSE: {best_mse:.6f}")
        print(f"Test R2: {r2_oos:.4%}")
        print(f"Top 10 Coefficients:\n{top_ten}\n") 

        # Save results summary
        results.append({
            "train_end": train["month"].max(),
            "val_end": val["month"].max(),
            "test_end": test["month"].max(),
            "val_mse": best_mse,
            "best_alpha": best_alpha,
            "best_l1_ratio": best_l1_ratio,
            "test_r2": r2_oos,
            "model": model_to_test,
        })

    # Combine all test periods into one dataframe
    prediction_df = pd.concat(predictions, axis=0)
    
    # Compute overall test R-squared and store in the last results entry
    overall_r2 = r_squared_oos(prediction_df["ret_excess"], prediction_df["pred_ret_excess"])
    results[-1]["overall_r2"] = overall_r2

    # Print overall R-squared
    print(f"\nOverall Test R-squared across all splits: {overall_r2:.4%}")

    # Combine coefficients into a wide DataFrame: index = split, columns = features
    coef_df = pd.DataFrame(coef_list)
    coef_df.index = np.arange(1, len(coef_df) + 1)
    coef_df.index.name = "split"
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df, prediction_df, coef_df

### Run ElasticNet ###
results_df, prediction_df, coef_df = elastic_net(
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
    n_jobs=32,
)