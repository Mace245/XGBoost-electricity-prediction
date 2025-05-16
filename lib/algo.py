import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
import pickle
import data

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

def train_all_dms_horizon_models(
    # Parameter name changed for clarity
    base_data_for_dms_training: pd.DataFrame,
    max_forecast_horizon_hours: int,
    features_list: list[str],
    target_col: str
):
    """
    Trains and saves a separate XGBoost model for each forecast horizon
    from 1 hour up to max_forecast_horizon_hours, using the provided base data.
    """
    print(f"Starting DMS model training for horizons 1 to {max_forecast_horizon_hours} hours...")
    print(f"Using base data of shape: {base_data_for_dms_training.shape}")

    for h in range(1, max_forecast_horizon_hours + 1):
        print(f"  Training DMS model for horizon h={h}...")

        training_data_for_h = data.create_dms_training_data_for_horizon(
            # Pass the (potentially filtered) base data here
            merged_data_full=base_data_for_dms_training,
            features_list=features_list,
            target_col=target_col,
            horizon=h
        )

        if training_data_for_h.X.empty or training_data_for_h.y.empty:
            print(f"    Skipping horizon h={h} due to insufficient data after processing.")
            print(f"    Input data shape for h={h} before create_dms_training_data_for_horizon: {base_data_for_dms_training.shape}")
            # If you want to see what create_dms_training_data_for_horizon produced before dropna:
            # temp_data_with_target = base_data_for_dms_training.copy()
            # temp_data_with_target = algo.create_time_features(temp_data_with_target)
            # temp_data_with_target = algo.create_lagged_features(temp_data_with_target, target_col=target_col)
            # temp_data_with_target[f'{target_col}_h{h}'] = temp_data_with_target[target_col].shift(-h)
            # print(f"    Shape after feature/target creation for h={h} (before dropna): {temp_data_with_target.shape}")
            # print(f"    NaNs in key columns for h={h} (before dropna):")
            # for col_to_check in features_list + [f'{target_col}_h{h}']:
            #     if col_to_check in temp_data_with_target.columns:
            #         print(f"      {col_to_check}: {temp_data_with_target[col_to_check].isna().sum()} NaNs")
            continue

        print(f"    Training XGBoost model with {len(training_data_for_h.X)} samples for h={h}...")
        model_for_h = train_xgboost_model(
            X_train=training_data_for_h.X,
            y_train=training_data_for_h.y
        )
        print(f"    XGBoost model training complete for h={h}.")

        model_filename = f'models/dms_model_horizon_{h}h.pkl' # Add _TEST to differentiate
        try:
            with open(model_filename, 'wb') as f:
                pickle.dump(model_for_h, f)
            print(f"    Saved DMS model for horizon h={h} to {model_filename}")
        except Exception as e:
            print(f"    Error saving model for h={h}: {e}")

    print("All DMS horizon model training complete.")

# Feature Engineering
def create_lagged_features(data, target_col='Wh', lags=[1, 24, 168]):
    """Add lagged features (1h, 24h, 168h=1 week)"""
    for lag in lags:
        data[f'lag_{lag}'] = data[target_col].shift(lag)
    return data.dropna()

# Forecasting
def predict_on_window(model, last_observed_window):
    predictions = model.predict(last_observed_window)
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

def predict_dms(
    history_df: pd.DataFrame,         # DataFrame containing historical target_col and all feature columns
                                      # up to the current time 't'. Index must be DatetimeIndex.
                                      # Must be long enough for max_lag calculation.
    max_horizon_hours: int,           # The total number of hours into the future to forecast.
    models_base_path: str = 'models/', # Base path where dms_model_horizon_{h}h.pkl files are stored.
    future_exog_series: pd.DataFrame = None # Optional: DataFrame of known future exogenous variables,
                                           # indexed by DateTime. E.g., future temperatures.
                                           # Columns should match exogenous feature names in features_list.
):
    """
    Generates a multi-step forecast using pre-trained DMS models (one model per horizon).

    Args:
        history_df: Historical data up to the point of forecast. Must include the target_col
                    (for lag calculation) and any other features used by the models if they
                    are not re-derived from the timestamp (e.g., historical temperature).
        max_horizon_hours: How many hours ahead to forecast.
        models_base_path: Path to the directory containing saved DMS models.
        future_exog_series: DataFrame with future values of exogenous variables.
                             Its index should be DatetimeIndex covering the forecast period.
                             Its columns should match the names of exogenous features.

    Returns:
        A pandas Series with the forecast, indexed by DateTime.
    """
    global features_list, target_col # Assuming these are defined in the script's scope
                                     # Cleaner: pass them as arguments

    if not isinstance(history_df.index, pd.DatetimeIndex):
        raise ValueError("history_df must have a DatetimeIndex.")
    if history_df.empty:
        raise ValueError("history_df cannot be empty.")

    # Determine the starting point for the forecast
    last_known_time = history_df.index[-1]
    print(f"Generating DMS forecast for {max_horizon_hours} hours, starting after: {last_known_time}")

    all_predictions = []
    forecast_timestamps = []

    # --- Prepare the base feature set from the latest historical data ---
    # This feature set (X_t) will be used as input for each model_h, potentially
    # augmented with future exogenous variables for that specific horizon h.
    # `create_dms_feature_set_for_prediction` generates features for `last_known_time`
    # using data within `history_df`.
    try:
        base_features_at_t = data.create_dms_feature_set_for_prediction(
            history_df_slice=history_df, # Pass the relevant slice of history
            features_list=features_list,
            target_col=target_col
        )
    except Exception as e:
        print(f"Error creating base feature set for prediction at {last_known_time}: {e}")
        return pd.Series(dtype=float) # Return empty series on error


    # Loop for each hour in the forecast horizon
    for h in range(1, max_horizon_hours + 1):
        current_forecast_dt = last_known_time + pd.Timedelta(hours=h)
        forecast_timestamps.append(current_forecast_dt)

        # Load the specialized model for horizon 'h'
        model_filename = f"{models_base_path}dms_model_horizon_{h}h.pkl" # Or your _TEST filename
        try:
            with open(model_filename, 'rb') as f:
                model_h = pickle.load(f)
        except FileNotFoundError:
            print(f"  Error: Model for horizon h={h} ({model_filename}) not found. Appending NaN.")
            all_predictions.append(np.nan)
            continue
        except Exception as e:
            print(f"  Error loading model for h={h} ({model_filename}): {e}. Appending NaN.")
            all_predictions.append(np.nan)
            continue

        # Prepare the specific feature set for this model_h
        # Start with a copy of the base features derived from time 't'
        features_for_model_h = base_features_at_t.copy()

        # If future exogenous variables are provided and needed by the model, update them.
        # This assumes model_h was trained expecting features like 'temperature_at_t+h'.
        # If model_h was trained only on features from time 't', this section can be simpler.
        if future_exog_series is not None and not future_exog_series.empty:
            for exog_col_name in future_exog_series.columns:
                if exog_col_name in features_for_model_h.columns: # Check if this exog var is a feature
                    try:
                        # Get the future value of the exogenous variable for the current_forecast_dt
                        future_val = future_exog_series.loc[current_forecast_dt, exog_col_name]
                        features_for_model_h[exog_col_name] = future_val
                        # print(f"    Updated {exog_col_name} for h={h} to {future_val} for time {current_forecast_dt}")
                    except KeyError:
                        print(f"    Warning: Future value for '{exog_col_name}' at {current_forecast_dt} not found in future_exog_series.")
                        # Decide on fallback: keep value from base_features_at_t (i.e., value at time 't'), or use NaN, or error.
                        # Current behavior: keeps the value from base_features_at_t (value at time 't')
                        pass
                    except Exception as e_exog:
                        print(f"    Error accessing future exog '{exog_col_name}' for {current_forecast_dt}: {e_exog}")


        # Make the prediction using model_h
        try:
            # model_h.predict expects a 2D array or DataFrame
            prediction_value = model_h.predict(features_for_model_h)[0]
            prediction_value = max(0.0, float(prediction_value)) # Ensure non-negative
            all_predictions.append(prediction_value)
            # print(f"  h={h}: Predicted {prediction_value:.2f} for {current_forecast_dt}")
        except Exception as e_pred:
            print(f"  Error during prediction with model_h for h={h} ({current_forecast_dt}): {e_pred}. Appending NaN.")
            all_predictions.append(np.nan)

    # Assemble the forecast into a pandas Series
    if not forecast_timestamps: # Should not happen if max_horizon_hours >= 1
        return pd.Series(dtype=float)

    final_forecast_series = pd.Series(all_predictions, index=pd.DatetimeIndex(forecast_timestamps, name='DateTime'))
    print(f"DMS forecast generation complete for {max_horizon_hours} hours.")
    return final_forecast_series
