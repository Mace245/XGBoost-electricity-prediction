import algo
import data
import pickle
import pandas as pd
import numpy as np # Added for np.timedelta64
import matplotlib.pyplot as plt

# --- Constants ---
LATITUDE = 14.5833
LONGITUDE = 121.0
MODEL_FILENAME = 'model.pkl'
TRAINING_DATA_CSV = 'training_data.csv' # Assumes create_training_data ran once
PROCESSED_DATA_CSV = 'processed_hourly_Wh_data.csv' # Original source if needed
TIMEZONE = 'Asia/Kuala_Lumpur' # Define timezone consistently

# Define features (ensure these match columns in training_data.csv after processing)
features = [
    'hour', 'day_of_week', 'day_of_month', 'is_weekend',
    'lag_1', 'lag_24', 'lag_168',
    'temperature'
]
target = 'Wh'
max_lag_hours = 168 # Based on lag_168

def _load_data(data_path: str) -> pd.DataFrame:
    """Loads and prepares the base dataframe."""
    try:
        df = pd.read_csv(data_path, parse_dates=['DateTime'], index_col='DateTime')
        if df.index.tz is None:
            df = df.tz_localize(TIMEZONE)
        else:
            df = df.tz_convert(TIMEZONE)

        # Ensure temperature is present if expected in features
        if 'temperature' not in df.columns and 'temperature' in features:
             raise ValueError(f"'temperature' column missing from {data_path}. Ensure data preparation includes weather.")
        # Ensure target is present
        if target not in df.columns:
             raise ValueError(f"Target column '{target}' missing from {data_path}")

        # Simple check for necessary columns based on features list
        missing_features = [f for f in features if f not in df.columns and f != 'temperature'] # Temp checked above
        if missing_features:
            raise ValueError(f"Missing feature columns in {data_path}: {missing_features}. Ensure feature engineering steps were run.")

        return df.sort_index() # Ensure data is sorted by time
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {data_path}. Run data preparation first.")
    except Exception as e:
        raise RuntimeError(f"Error loading data from {data_path}: {e}")

def verify_model(base_data_path: str, verification_window_str: str):
    """
    Trains a model on historical data excluding the verification window,
    predicts the window, and compares against actuals.
    Example verification_window_str: '1D', '7D', '30D'
    """
    print(f"\n--- Verifying Model Accuracy for Last {verification_window_str} ---")
    full_df = _load_data(base_data_path)
    expected_freq = pd.Timedelta(hours=1) # Define expected frequency

    try:
        verification_td = pd.Timedelta(verification_window_str)
        # Ensure Timedelta is positive
        if verification_td <= pd.Timedelta(0):
            raise ValueError("verification_window_str must represent a positive duration.")
    except ValueError:
        raise ValueError(f"Invalid verification_window_str: '{verification_window_str}'. Use formats like '1D', '7D', '30D'.")


    # 1. Split Data
    verification_end_time = full_df.index.max()
    # Calculate start time precisely using explicit Timedelta
    # FIX: Replace full_df.index.freq with expected_freq
    verification_start_time = verification_end_time - verification_td + expected_freq
    # FIX: Replace full_df.index.freq with expected_freq
    training_end_time = verification_start_time - expected_freq

    training_df = full_df.loc[full_df.index <= training_end_time]
    verification_df_actual = full_df.loc[verification_start_time : verification_end_time] # Inclusive slice

    if training_df.empty or verification_df_actual.empty:
        raise ValueError(f"Data splitting resulted in empty dataframes. Check data range and verification window '{verification_window_str}'. Training ends: {training_end_time}, Verification starts: {verification_start_time}")

    print(f"Training data range: {training_df.index.min()} to {training_df.index.max()}")
    print(f"Verification data range: {verification_df_actual.index.min()} to {verification_df_actual.index.max()}")

    # 2. Train Model on Past Data
    X_train = training_df[features]
    y_train = training_df[target]
    print("Training model on historical data (excluding verification period)...")
    model = algo.train_xgboost_model(X_train, y_train) # Assuming eval_set within train_xgboost is sufficient

    # 3. Prepare Inputs for Prediction
    # History needed for lags: Use the end of the actual training data
    history_for_prediction = training_df.iloc[-max_lag_hours:] # Get last max_lag hours from training set

    # Fetch HISTORICAL weather for the verification period
    weather_start_date = verification_df_actual.index.min().strftime('%Y-%m-%d')
    weather_end_date = verification_df_actual.index.max().strftime('%Y-%m-%d')
    print(f"Fetching historical weather for verification: {weather_start_date} to {weather_end_date}")
    weather_for_verification = data.temp_fetch(
        start_date=weather_start_date,
        end_date=weather_end_date,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        historical=True
    )
    # Ensure timezone consistency
    if weather_for_verification.index.tz is None:
        weather_for_verification = weather_for_verification.tz_localize(TIMEZONE)
    else:
        weather_for_verification = weather_for_verification.tz_convert(TIMEZONE)


    # 4. Predict Verification Period
    hours_to_forecast = len(verification_df_actual)
    print(f"Predicting verification period ({hours_to_forecast} hours)...")
    forecast_series = algo.predict_on_window_recursive(
        model=model,
        history_df=training_df, # Pass the whole training_df, recursive func uses tail
        future_temps_series=weather_for_verification,
        hours_to_forecast=hours_to_forecast,
        model_features=features,
        target_variable=target,
        max_lag=max_lag_hours
    )

    # 5. Evaluate
    print("Evaluating verification forecast...")
    actual_target_values = verification_df_actual[target]

    # Align forecast index to actual verification index before comparison
    # Important if prediction returns slightly fewer/more points due to edge cases
    forecast_series = forecast_series.reindex(actual_target_values.index)

    # # Plot loss curves (if available from algo.train_xgboost_model)
    # if hasattr(model, 'evals_result'):
    #     evals_result = model.evals_result()
    #     plt.figure(figsize=(10, 6))
    #     for metric in evals_result['validation_0']:
    #         plt.plot(evals_result['validation_0'][metric], label=f"Train {metric}")
    #     if 'validation_1' in evals_result:
    #         for metric in evals_result['validation_1']:
    #             plt.plot(evals_result['validation_1'][metric], label=f"Validation {metric}")
    #     plt.title("Training and Validation Loss Curves")
    #     plt.xlabel("Iterations")
    #     plt.ylabel("Loss")
    #     plt.legend()
    #     plt.show()

    comparison_df = forecast_series.to_frame(name='Forecast').join(
        actual_target_values.to_frame(name='Actual')
    )
    output_filename = f'forecast_actual_comparison_{verification_window_str}.csv'
    comparison_df.to_csv(output_filename)
    print(f"Forecast and actual values exported to {output_filename}")

    data.visualize(
        model=model, # Pass model for feature importance plot
        features=features,
        actual_values=actual_target_values,
        forecast_values=forecast_series,
        period_label=f"{verification_window_str} Verification"
    )
    print(f"--- Verification for {verification_window_str} Complete ---")


def predict_future(base_data_path: str, forecast_window_str: str, train_fresh_model: bool = True):
    """
    Trains a model on ALL available data (optionally) and predicts the future window.
    Example forecast_window_str: '1D', '7D', '30D'
    """
    print(f"\n--- Predicting Future {forecast_window_str} ---")
    full_df = _load_data(base_data_path)
    expected_freq = pd.Timedelta(hours=1) # Define expected frequency

    try:
        forecast_td = pd.Timedelta(forecast_window_str)
        if forecast_td <= pd.Timedelta(0):
            raise ValueError("forecast_window_str must represent a positive duration.")
    except ValueError:
         raise ValueError(f"Invalid forecast_window_str: '{forecast_window_str}'. Use formats like '1D', '7D', '30D'.")


    # 1. Train or Load Model
    if train_fresh_model:
        print("Training model on ALL available data...")
        X_train_full = full_df[features]
        y_train_full = full_df[target]
        model = algo.train_xgboost_model(X_train_full, y_train_full)
        # Save the trained model
        with open(MODEL_FILENAME, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model trained and saved to {MODEL_FILENAME}")
    else:
        try:
            with open(MODEL_FILENAME, 'rb') as f:
                model = pickle.load(f)
            print(f"Loaded pre-trained model from {MODEL_FILENAME}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file {MODEL_FILENAME} not found. Set train_fresh_model=True or provide the file.")

    # 2. Prepare Inputs for Prediction
    last_known_time = full_df.index.max()
    # FIX: Replace full_df.index.freq with expected_freq
    forecast_start_dt = last_known_time + expected_freq
    # FIX: Replace full_df.index.freq with expected_freq
    forecast_end_dt = forecast_start_dt + forecast_td - expected_freq

    weather_start_date = forecast_start_dt.strftime('%Y-%m-%d')
    weather_end_date = forecast_end_dt.strftime('%Y-%m-%d')

    print(f"Fetching forecast weather for prediction: {weather_start_date} to {weather_end_date}")
    weather_forecast = data.temp_fetch(
        start_date=weather_start_date,
        end_date=weather_end_date,
        latitude=LATITUDE,
        longitude=LONGITUDE,
        historical=True # IMPORTANT: Fetch future forecast
    )
    # Ensure timezone consistency
    if weather_forecast.index.tz is None:
        weather_forecast = weather_forecast.tz_localize(TIMEZONE)
    else:
        weather_forecast = weather_forecast.tz_convert(TIMEZONE)


    # 3. Predict Future Period
    hours_to_forecast = int(forecast_td.total_seconds() / 3600)
    print(f"Predicting future period ({hours_to_forecast} hours)...")

    forecast_series = algo.predict_on_window_recursive(
        model=model,
        history_df=full_df, # Use all historical data for lags
        future_temps_series=weather_forecast,
        hours_to_forecast=hours_to_forecast,
        model_features=features,
        target_variable=target,
        max_lag=max_lag_hours
    )

    # 4. Output/Visualize Forecast
    print("\nFuture Forecast:")
    print(forecast_series)
    forecast_csv_filename = f'forecast_{forecast_window_str}.csv'
    forecast_series.to_csv(forecast_csv_filename)
    print(f"Forecast saved to {forecast_csv_filename}")

    # Visualize the forecast (no actuals to compare here)
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_series.index, forecast_series, label='Forecast', linestyle='--')
    plt.title(f"Future {forecast_window_str} Forecast")
    plt.xlabel("DateTime")
    plt.ylabel("Wh")
    plt.legend()
    plt.show()

    print(f"--- Future {forecast_window_str} Prediction Complete ---")
    return forecast_series


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the base training data with features exists.
    # If not, you might want to run data.create_training_data(features, target) first.
    try:
       test_load = _load_data(TRAINING_DATA_CSV) # Test load
       print(f"Successfully loaded {TRAINING_DATA_CSV}. Index frequency: {test_load.index.freq}")
    except FileNotFoundError:
       print(f"{TRAINING_DATA_CSV} not found. Running data preparation...")
       # Ensure data.py functions handle timezone correctly
       data.create_training_data(features, target) # This function now needs to exist and work
       print("Data preparation complete.")
    except Exception as e:
        print(f"Error during initial data load/preparation: {e}")
        # Decide whether to exit or continue if data loading fails


    # --- Verification Examples ---

    # Verify model performance on the last 1 day of data
    verify_model(TRAINING_DATA_CSV, '1D')

    # Verify model performance on the last 7 days of data
    verify_model(TRAINING_DATA_CSV, '7D')

    # Verify model performance on the last 30 days of data (if data permits)
    # verify_model(TRAINING_DATA_CSV, '30D') # Uncomment if desired and data available




    # --- Future Prediction Example ---
    try:
        # Train a fresh model on all data and predict the next 7 days
        predict_future(TRAINING_DATA_CSV, '7D', train_fresh_model=True)

        # Predict the next 1 day using the model saved above (or previously)
        # predict_future(TRAINING_DATA_CSV, '1D', train_fresh_model=False) # Uncomment if desired

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Error during future prediction: {e}")
    except Exception as e:
         print(f"An unexpected error occurred during future prediction: {e}")
