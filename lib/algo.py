import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
import pickle
from lib import data
import os

def create_time_features(data_df):
    data_df = data_df.copy()
    data_df['hour'] = data_df.index.hour
    data_df['day_of_week'] = data_df.index.dayofweek
    data_df['day_of_month'] = data_df.index.day
    data_df['is_weekend'] = (data_df['day_of_week'] >= 5).astype(int)
    return data_df

def create_lagged_features(data_df, target_col='Wh'):
    lags=[1, 24, 72, 168]
    data_df = data_df.copy()
    for lag in lags:
        data_df[f'lag_{lag}'] = data_df[target_col].shift(lag)
    return data_df

def train_xgboost_model(X_train, y_train):
    model = xgb.XGBRegressor(
        max_depth=3, learning_rate=0.01, n_estimators=2000,
        # colsample_bytree=1, subsample=0.3, reg_alpha=0, reg_lambda=1,
        n_jobs=None, early_stopping_rounds=50, eval_metric='rmse', 
    )
    # Using a simple train/validation split for the profile model is sufficient.
    # TimeSeriesSplit is more critical for recursive/DMS where sequence matters more.
    # For simplicity and speed, we can use a standard validation set.
    split_point = int(len(X_train) * 0.9)
    X_train_fold, X_val_fold = X_train.iloc[:split_point], X_train.iloc[split_point:]
    y_train_fold, y_val_fold = y_train.iloc[:split_point], y_train.iloc[split_point:]
    
    model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
    print(f"    Validation Score for the profile model: {model.best_score:.4f}")
    return model

# --- MODIFIED: Replaced DMS training with Profile training ---
def train_profile_model(
    base_data_for_training: pd.DataFrame,
    features_list: list[str],
    target_col: str,
):
    """
    Trains a single XGBoost model using the Time-Based Profile method (X(t) -> y(t)).
    """
    print(f"Profile model training started. Using base data of shape: {base_data_for_training.shape}.")

    # 1. Create features. The target is the column itself, no shifting.
    df = base_data_for_training.copy()
    df = create_time_features(df)
    df = create_lagged_features(df, target_col=target_col)
    
    # Drop NaNs created by lags
    df = df.dropna(subset=features_list + [target_col])
    
    X_train = df[features_list]
    y_train = df[target_col] # Target is at the same time `t` as features
    
    if X_train.empty:
        print("    Not enough data to create training set. Aborting.")
        return

    print(f"    Training XGBoost model with {len(X_train)} samples...")
    model = train_xgboost_model(X_train=X_train, y_train=y_train)

    if model is None:
        print("    Failed to train model. Skipping save.")
        return

    print("    XGBoost model training complete.")
    
    model_filename = f'models/profile_model.pkl'
    os.makedirs('models', exist_ok=True)
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"    Saved profile model to {model_filename}")
    print("Profile model training complete.")

# --- MODIFIED: Replaced DMS prediction with Profile prediction ---
def predict_profile(
    history_df: pd.DataFrame,
    max_horizon_hours: int,
    features_list: list[str], 
    target_col: str,          
    models_base_path: str = 'models/',
    future_exog_series: pd.DataFrame = None
):
    """
    Generates a forecast by creating a block of future features and predicting them all at once.
    """
    model_filename = f"{models_base_path}profile_model.pkl"
    try:
        with open(model_filename, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Profile model not found at {model_filename}. Please retrain the model.")
    except Exception as e:
        raise IOError(f"Error loading model from {model_filename}: {e}")

    last_known_time = history_df.index[-1]
    print(f"Generating profile-based forecast for {max_horizon_hours} hours, starting after: {last_known_time}")

    # 1. Create future timestamps
    future_dates = pd.date_range(start=last_known_time + pd.Timedelta(hours=1), periods=max_horizon_hours, freq='h')
    future_df = pd.DataFrame(index=future_dates)
    
    # 2. Combine history with future skeleton to create features
    # This allows lag features to look back into the actual history
    combined_df = pd.concat([history_df, future_df])

    # 3. Create all features for the combined dataframe
    features_df = create_time_features(combined_df)
    features_df = create_lagged_features(features_df, target_col=target_col)
    
    # 4. Add known future temperatures
    if future_exog_series is not None:
        features_df['temperature'].update(future_exog_series['temperature'])
        
    # 5. Select only the future rows that we need to predict
    X_future = features_df.loc[future_dates]
    
    # Handle any potential NaNs in the future features (e.g., from long lags into sparse history)
    X_future = X_future[features_list].fillna(method='ffill').bfill()
    
    # 6. Predict all future steps in one shot
    predictions = model.predict(X_future)
    
    # Ensure predictions are non-negative
    predictions[predictions < 0] = 0
    
    final_forecast_series = pd.Series(predictions, index=future_dates, name='DateTime')
    print(f"Profile-based forecast generation complete.")
    return final_forecast_series