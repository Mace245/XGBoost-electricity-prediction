import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

def create_time_features(data):
    """Add temporal features"""
    data = data.copy()
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek  # 0=Monday
    data['day_of_month'] = data.index.day
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
    return data

def add_seasonal_components(data, period=24):
    """Add seasonal decomposition features (daily cycle)"""
    decomposition = seasonal_decompose(
        data['Wh'], 
        model='additive', 
        period=period
    )
    data['trend'] = decomposition.trend
    data['seasonal'] = decomposition.seasonal
    data['residual'] = decomposition.resid
    return data.dropna()


# mape (rmse) = 173, 151

# # Modeling with Optuna hyperparameter optimization
# def train_xgboost_model(X_train, y_train):
#     """Train XGBoost model using Optuna for hyperparameter tuning."""
    
#     def objective(trial):
#         param = {
#             'max_depth': trial.suggest_int('max_depth', 3, 10),
#             'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2, log=True),
#             'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
#             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
#             'subsample': trial.suggest_float('subsample', 0.5, 0.9),
#             'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1.0,log=True),
#             'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 1.0, log=True),
#             'n_jobs': -1,
#             'eval_metric': 'rmse',
#             'tree_method': 'hist',
#             'device': 'cuda'
#         }
        
#         # Initialize the model with early stopping
#         model = xgb.XGBRegressor(**param, early_stopping_rounds=50)
#         tss = TimeSeriesSplit(n_splits=3)
#         scores = []
        
#         for train_idx, val_idx in tss.split(X_train):
#             X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
#             y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
#             model.fit(
#                 X_train_fold, y_train_fold,
#                 eval_set=[(X_val_fold, y_val_fold)],
#                 verbose=False
#             )
#             scores.append(model.best_score)
        
#         return np.mean(scores)
    
#     study = optuna.create_study(direction='minimize')
#     study.optimize(objective, n_trials=69)
    
#     print("Best trial:")
#     trial = study.best_trial
#     print(f"  RMSE: {trial.value}")
#     print("  Best hyperparameters:")
#     for key, value in trial.params.items():
#         print(f"    {key}: {value}")
    
#     # Build the final model with the best hyperparameters and train on full data
#     best_params = trial.params
#     best_params.update({
#         'n_jobs': -1,
#         'eval_metric': 'rmse',
#     })
    
#     best_model = xgb.XGBRegressor(**best_params)
#     best_model.fit(X_train, y_train, verbose=False)
#     return best_model

# Modeling
def train_xgboost_model(X_train, y_train):
    """Train with paper-inspired parameters"""
    model = xgb.XGBRegressor(
        max_depth=6,
        learning_rate=0.069,
        n_estimators=1000,
        colsample_bytree=0.7,
        subsample=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric='rmse',
    )
    # mape (rmse) = 173, 151
    
    # Time-series cross-validation
    tss = TimeSeriesSplit(n_splits=3)
    scores = []
    for train_idx, val_idx in tss.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )
        scores.append(model.best_score)
    
    print(f"Avg Validation Score: {np.mean(scores):.4f}")
    return model

# Feature Engineering
def create_lagged_features(data, target_col='Wh', lags=[1, 24, 168]):
    """Add lagged features (1h, 24h, 168h=1 week)"""
    for lag in lags:
        data[f'lag_{lag}'] = data[target_col].shift(lag)
    return data.dropna()

# Forecasting
def predict_on_window(model, last_observed_window):
    predictions = model.predict(last_observed_window)
    print("X_test columns:", type(last_observed_window))
    predictions = pd.Series(predictions, index=last_observed_window.index)  # Add datetime index
    return predictions

def _create_time_features_single(dt_object):
    """Creates time features for a single future timestamp."""
    # Ensure dt_object is a pandas Timestamp for attribute access
    dt_object = pd.Timestamp(dt_object)
    return {
        'hour': dt_object.hour,
        'day_of_week': dt_object.dayofweek, # Monday=0, Sunday=6
        'day_of_month': dt_object.day,
        'is_weekend': 1 if dt_object.dayofweek >= 5 else 0
    }

def predict_on_window_recursive(
    model: xgb.XGBRegressor,
    history_df: pd.DataFrame,
    future_temps_series: pd.DataFrame,
    hours_to_forecast: int,
    model_features: list,
    target_variable: str,
    max_lag: int,
):
    if model is None:
        raise ValueError("Model not loaded.")
    if len(history_df) < max_lag:
        raise ValueError(f"Insufficient historical data. Need at least {max_lag} readings, got {len(history_df)}.")
    if not isinstance(history_df.index, pd.DatetimeIndex) or history_df.index.tz is None:
         raise ValueError("history_df index must be a timezone-aware DatetimeIndex.")
    if not isinstance(future_temps_series.index, pd.DatetimeIndex) or future_temps_series.index.tz is None:
         raise ValueError("future_temps_series index must be a timezone-aware DatetimeIndex.")
    if future_temps_series.index.tz != history_df.index.tz:
         raise ValueError("Timezones of history_df and future_temps_series must match.")

    last_known_time = history_df.index[-1]
    forecast_start_dt = last_known_time + pd.Timedelta(hours=1)

    current_data = history_df[[target_variable]].copy()

    predictions_list = []
    timestamps_list = []

    print(f"Starting recursive prediction for {hours_to_forecast} hours...")

    for h in range(hours_to_forecast):
        current_pred_time = forecast_start_dt + pd.Timedelta(hours=h)
        timestamps_list.append(current_pred_time)

        # 1. Create time features
        features = _create_time_features_single(current_pred_time)

        # 2. Get future temperature
        try:
            features['temperature'] = future_temps_series.loc[current_pred_time].values[0]
            print("temperature feature:", features['temperature'])
            print("printing features:", features)
        except KeyError:
            print(f"Warning: Temperature for {current_pred_time} not found in forecast, using last available.")
            # Fallback: use the last value in the future_temps_series or history
            features = future_temps_series.iloc[-1] if not future_temps_series.empty else history_df['Temperature'].iloc[-1]


        # 3. Calculate Lag Features (using current_data)
        for lag in [1, 24, 168]: # Assuming these are the standard lags used
            lag_time = current_pred_time - pd.Timedelta(hours=lag)
            lag_key = f'lag_{lag}'
            # Use .get for robust lookup, provide NaN if lag goes beyond available data
            features[lag_key] = current_data[target_variable].get(lag_time, np.nan)

        # 4. Prepare feature vector
        try:
            # Create DataFrame row with columns in the correct order specified by model_features
            current_features_df = pd.DataFrame([features], columns=model_features)
        except KeyError as e:
             raise ValueError(f"Feature mismatch: Missing expected feature {e} during recursive prediction. Features available: {list(features.keys())}")

        # Handle potential NaNs (e.g., from lags at the very beginning)
        current_features_df = current_features_df.fillna(0) # Adjust NaN strategy if needed (e.g., mean/median)

        # 5. Predict
        try:
            prediction = model.predict(current_features_df)[0]
            prediction = max(0.0, float(prediction)) # Ensure non-negative float
        except Exception as pred_err:
            print(f"Error during prediction for {current_pred_time}: {pred_err}")
            prediction = 0 # Simple fallback

        predictions_list.append(prediction)

        # 6. Add prediction to current_data for next iteration's lag calculation
        new_row = pd.DataFrame({target_variable: [prediction]}, index=[current_pred_time])
        current_data = pd.concat([current_data, new_row])
        print("\n\n")

    print("Recursive prediction complete.")

    # Create and return final Pandas Series
    predictions_series = pd.Series(predictions_list, index=pd.DatetimeIndex(timestamps_list, name='DateTime'))
    return predictions_series