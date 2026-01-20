import pandas as pd
import numpy as np
import os

def convert_formats(df):
    """Convert DataFrame columns to more memory-efficient formats that are ML-friendly.

    Args:
        df (DataFrame): Containing unformatted columns.

    Returns:
        df (DataFrame): Containing formatted columns. 
    """
    # Normalize column names to lowercase
    df.columns = df.columns.str.lower()
    
    # Lists of columns to handle specifically
    date_cols = [c for c in df.columns if any(k in c for k in ["date", "month"])]
    id_cols   = [c for c in df.columns if any(k in c for k in ["lpermno", "permno", "gvkey", "cik"])]
    sic_cols = [c for c in df.columns if any(k in c for k in ["siccd", "sic2"])]
    sic_dummies = [c for c in df.columns if c.startswith("sic2_")]

    # For loop to handle each column
    for col in df.columns:
        
        # Date columns are handled seperately due to messy formatting
        if col in date_cols:
            continue
        
        # Convert ID columns to string, strip leading zeros and convert to Int32 
        elif col in id_cols:
            df[col] = df[col].astype(str).str.lstrip("0")
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")
            continue

        # Convert SIC columns to category
        elif col in sic_cols:
            df[col] = df[col].astype("category") # Convert SIC codes to categorical
            continue
        
        # Do not handle SIC dummies as they are one-hot encoded
        elif col in sic_dummies:
            continue

        # Create a variable for column dtypes
        dtype = df[col].dtype
        
        # Downcast float64 to float32
        if dtype == "float64":
            df[col] = pd.to_numeric(df[col], downcast="float")

        # Downcast int64 to Int32 
        elif dtype == "Int64" or dtype == "int64":
            df[col] = df[col].astype("Int32")
            
        # Downcast object to numeric if the mean of the columns is strictly larger than 1 + eps, else convert to category    
        elif dtype == "object":
            temp = pd.to_numeric(df[col], errors="coerce") 
            temp = temp.dropna()
            
            if not temp.empty:
                temp_mean = temp.mean()
                eps = 1e-6
                
                if temp_mean > 1 + eps: 
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32") # Convert to float32
                else:
                    df[col] = df[col].astype("category")
            else:
                df[col] = df[col].astype("category")
    return df

def expanding_window_split(
    df: pd.DataFrame,
    train_size: int,
    val_size: int,
    test_size: int,
    step_size: int,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """Expanding window split for time series data.

    Args:
        df (pd.DataFrame): DataFrame containing a 'month' column in datetime format.
        train_size (int): Number of months to include in the training set.
        val_size (int): Number of months to include in the validation set.
        test_size (int): Number of months to include in the test set.
        step_size (int): Number of months to step forward for each iteration.
        start_date (str | None, optional): Start date for the data split. Defaults to None.
        end_date (str | None, optional): End date for the data split. Defaults to None.

    Raises:
        TypeError: If 'month' column is not in datetime format.

    Yields:
        train, val, test (pd.DataFrame): DataFrames containing the train, validation, and test sets.
    """
    
    # Ensure 'month' column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df["month"]):
        raise TypeError("'month' column must be in datetime format")

    # Apply date filters if provided
    mask = pd.Series(True, index=df.index)
    
    if start_date:
        mask &= df["month"] >= pd.Timestamp(start_date)
    if end_date:
        mask &= df["month"] <= pd.Timestamp(end_date)

    months = sorted(df.loc[mask, "month"].unique())

    # Set end index
    end_idx = train_size + val_size + test_size + 1
    
    # Create a while loop to iterate until the end index exceeds the number of unique months
    while end_idx <= len(months):
        train_months = months[: end_idx - (val_size + test_size)]
        val_months   = months[end_idx - (val_size + test_size) : end_idx - test_size]
        test_months  = months[end_idx - test_size : end_idx]

        # Slice firm-month panel
        train = df[df["month"].isin(train_months)]
        val = df[df["month"].isin(val_months)]
        test = df[df["month"].isin(test_months)]

        # Stream one result at a time
        yield train, val, test

        # Expand by step_size months
        end_idx += step_size
            
def r_squared_oos(y_test, y_test_pred):
    """Calculate out-of-sample R-squared as defined by Gu, Kelly & Xiu (2020).

    Args:
        y_test (np.ndarray): True values.
        y_test_pred (np.ndarray): Predicted values.
    Returns:
        r2_oos (float): Out-of-sample R-squared.
    """
    ss_res = np.sum((y_test - y_test_pred) ** 2)
    ss_tot = np.sum(y_test ** 2)
    r2_oos = 1 - (ss_res / ss_tot)
    return r2_oos