
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
# from statsmodels.tsa.seasonal import seasonal_decompose # Not used in current DMS
import pickle
from lib import data
import os

# --- Feature Engineering (Keep as is) ---
def create_time_features(data_df):
    data_df = data_df.copy()
    data_df['hour'] = data_df.index.hour
    data_df['day_of_week'] = data_df.index.dayofweek
    data_df['day_of_month'] = data_df.index.day
    data_df['is_weekend'] = (data_df['day_of_week'] >= 5).astype(int)
    return data_df

def create_lagged_features(data_df, target_col='Wh', lags=[1, 24, 168]):
    data_df = data_df.copy()
    for lag in lags:
        data_df[f'lag_{lag}'] = data_df[target_col].shift(lag)
    # Drop NaNs created by the shift *within this function*
    # data_df = data_df.dropna(subset=[f'lag_{lag}' for lag in lags]) # Careful with this, might drop too much if called sequentially.
                                                                # create_dms_training_data will handle final dropna
    return data_df


# --- XGBoost Model Training (Keep as is) ---
def train_xgboost_model(X_train, y_train):
    model = xgb.XGBRegressor(
        max_depth=6, learning_rate=0.069, n_estimators=1000,
        colsample_bytree=0.7, subsample=0.8, reg_alpha=0.1, reg_lambda=0.1,
        n_jobs=-1, early_stopping_rounds=50, eval_metric='rmse',
    )
    tss = TimeSeriesSplit(n_splits=3)
    scores = []
    # print(f"    Inside train_xgboost_model. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    if X_train.empty or y_train.empty:
        print("    Cannot train XGBoost model with empty X_train or y_train.")
        return None # Or raise error

    for train_idx, val_idx in tss.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
        scores.append(model.best_score)
    print(f"    Avg Validation Score for this horizon model: {np.mean(scores):.4f}")
    return model

# --- DMS Training Orchestration ---
def train_all_dms_horizon_models(
    base_data_for_dms_training: pd.DataFrame,
    max_forecast_horizon_hours: int,
    features_list: list[str],
    target_col: str,
    models_save_path: str = 'models/' # Allow specifying save path
):
    print(f"Starting DMS model training for horizons 1 to {max_forecast_horizon_hours} hours.")
    print(f"Using base data of shape: {base_data_for_dms_training.shape}. TZ: {base_data_for_dms_training.index.tz if hasattr(base_data_for_dms_training.index, 'tz') else 'None'}")

    if not models_save_path.endswith('/'):
        models_save_path += '/'
    os.makedirs(models_save_path, exist_ok=True) # Ensure models directory exists


    for h in range(1, max_forecast_horizon_hours + 1):
        print(f"  Training DMS model for horizon h={h}...")

        # Use the data module's function to prepare X and y for this horizon
        training_data_for_h = data.create_dms_training_data_for_horizon(
            merged_data_full=base_data_for_dms_training, # Pass the training portion
            features_list=features_list,
            target_col=target_col,
            horizon=h
        )

        if training_data_for_h.X.empty or training_data_for_h.y.empty:
            print(f"    Skipping horizon h={h} due to insufficient data after processing.")
            # ... (optional debug prints from your train2.py if needed) ...
            continue

        print(f"    Training XGBoost model with {len(training_data_for_h.X)} samples for h={h}...")
        model_for_h = train_xgboost_model( # This is the local train_xgboost_model
            X_train=training_data_for_h.X,
            y_train=training_data_for_h.y
        )
        if model_for_h is None:
            print(f"    Failed to train model for h={h}. Skipping save.")
            continue
        print(f"    XGBoost model training complete for h={h}.")

        model_filename = f'{models_save_path}dms_model_horizon_{h}h.pkl'
        try:
            with open(model_filename, 'wb') as f:
                pickle.dump(model_for_h, f)
            print(f"    Saved DMS model for horizon h={h} to {model_filename}")
        except Exception as e:
            print(f"    Error saving model for h={h}: {e}")
    print("All DMS horizon model training complete.")


# --- DMS Prediction ---
def predict_dms(
    history_df: pd.DataFrame,
    max_horizon_hours: int,
    features_list: list[str], # Added parameter
    target_col: str,          # Added parameter
    models_base_path: str = 'models/',
    future_exog_series: pd.DataFrame = None
):
    if not isinstance(history_df.index, pd.DatetimeIndex):
        raise ValueError("history_df must have a DatetimeIndex.")
    if history_df.empty:
        raise ValueError("history_df cannot be empty.")

    last_known_time = history_df.index[-1]
    print(f"Generating DMS forecast for {max_horizon_hours} hours, starting after: {last_known_time} (TZ: {last_known_time.tz})")

    all_predictions = []
    forecast_timestamps = []

    try:
        # Make sure history_df used by create_dms_feature_set_for_prediction has target_col
        if target_col not in history_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in history_df for feature creation.")

        base_features_at_t = data.create_dms_feature_set_for_prediction(
            history_df_slice=history_df,
            features_list=features_list,
            target_col=target_col
        )
    except Exception as e:
        print(f"Error creating base feature set for prediction at {last_known_time}: {e}")
        import traceback; traceback.print_exc();
        return pd.Series(dtype=float)

    for h in range(1, max_horizon_hours + 1):
        current_forecast_dt = last_known_time + pd.Timedelta(hours=h)
        forecast_timestamps.append(current_forecast_dt)

        model_filename = f"{models_base_path}dms_model_horizon_{h}h.pkl"
        try:
            with open(model_filename, 'rb') as f:
                model_h = pickle.load(f)
        except FileNotFoundError:
            print(f"  Error: Model for horizon h={h} ({model_filename}) not found. Appending NaN.")
            all_predictions.append(np.nan)
            continue
        # ... (rest of error handling for model loading) ...
        except Exception as e: # Catch other load errors
            print(f"  Error loading model for h={h} ({model_filename}): {e}. Appending NaN.")
            all_predictions.append(np.nan)
            continue


        features_for_model_h = base_features_at_t.copy()
        if future_exog_series is not None and not future_exog_series.empty:
            for exog_col_name in future_exog_series.columns:
                if exog_col_name in features_for_model_h.columns:
                    try:
                        # Ensure future_exog_series is aligned with current_forecast_dt timezone
                        if future_exog_series.index.tz != current_forecast_dt.tz:
                             future_exog_series_aligned = future_exog_series.tz_convert(current_forecast_dt.tz)
                        else:
                             future_exog_series_aligned = future_exog_series
                        future_val = future_exog_series_aligned.loc[current_forecast_dt, exog_col_name]
                        features_for_model_h[exog_col_name] = future_val
                    except KeyError:
                        print(f"    Warning: Future value for '{exog_col_name}' at {current_forecast_dt} not found in future_exog_series.")
                        pass # Keeps value from base_features_at_t
                    # ... (rest of exog handling) ...

        try:
            prediction_value = model_h.predict(features_for_model_h)[0]
            prediction_value = max(0.0, float(prediction_value))
            all_predictions.append(prediction_value)
        except Exception as e_pred:
            print(f"  Error during prediction with model_h for h={h} ({current_forecast_dt}): {e_pred}. Appending NaN.")
            all_predictions.append(np.nan)

    final_forecast_series = pd.Series(all_predictions, index=pd.DatetimeIndex(forecast_timestamps, name='DateTime'))
    print(f"DMS forecast generation complete for {max_horizon_hours} hours.")
    return final_forecast_series