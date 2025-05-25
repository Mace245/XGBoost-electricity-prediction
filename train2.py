import pandas as pd
import numpy as np
import pickle
from lib import data
from lib import algo

train = True
TEST_SET_HOURS = 24 * 7

features_list = [
'hour', 'day_of_week', 'day_of_month', 'is_weekend',
'lag_1', 'lag_24', 'lag_72', 'lag_168',
'temperature'
]
target_col = 'Wh'


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
        model_filename = f"{models_base_path}dms_model_horizon_{h}h.pkl"
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
                    print('features', features_for_model_h)
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


if __name__ == "__main__":
    print("Fetching and preparing base historical data...")
    try:
        electricity_raw, temperature_raw = data.fetch_elec_temp()
        # Ensure consistent timezones from the start, e.g., all UTC
        if electricity_raw.index.tz is None: electricity_raw.index = electricity_raw.index.tz_localize('Asia/Kuala_Lumpur', ambiguous='infer') # Or your source TZ
        if temperature_raw.index.tz is None: temperature_raw.index = temperature_raw.index.tz_localize('Asia/Kuala_Lumpur', ambiguous='infer') # Or your source TZ

        if str(electricity_raw.index.tz).lower() != 'utc': electricity_raw.index = electricity_raw.index.tz_convert('UTC')
        if str(temperature_raw.index.tz).lower() != 'utc': temperature_raw.index = temperature_raw.index.tz_convert('UTC')
        
        merged_data_complete = data.prepare_data(electricity_raw, temperature_raw) # Should now be consistently UTC
        print(f"Base historical data prepared. Shape: {merged_data_complete.shape}, TZ: {merged_data_complete.index.tz}")
    except Exception as e:
        print(f"Failed to load or prepare data: {e}")
        import traceback; traceback.print_exc(); exit()

    if merged_data_complete.empty:
        print("Error: Base historical data is empty. Cannot proceed.")
        exit()


    # --- Define Train/Test Split Parameters ---
    # How many hours to use for the test set (this will also be our forecast horizon for evaluation)
  # e.g., 7 days for testing
    # MAX_HORIZON_FOR_TRAINING should be at least TEST_SET_HOURS if we want to evaluate the full test period
    # Or, if you want to train models for a shorter horizon than the test set, adjust accordingly.
    # For this example, we train models up to the length of the test set.
    MAX_HORIZON_MODELS_TO_TRAIN = TEST_SET_HOURS

    # Ensure we have enough data for training, max_lag, and the test set
    # Max lag is 168 hours (from features_list)
    max_lag_hours = 0
    for f in features_list:
        if f.startswith("lag_"):
            try: max_lag_hours = max(max_lag_hours, int(f.split("_")[1]))
            except: pass
    
    min_data_needed = max_lag_hours + MAX_HORIZON_MODELS_TO_TRAIN + TEST_SET_HOURS 
    # The + MAX_HORIZON_MODELS_TO_TRAIN is because create_dms_training_data_for_horizon
    # shifts the target, so the effective length of data used for training the last model (h=MAX_HORIZON_MODELS_TO_TRAIN)
    # is reduced by MAX_HORIZON_MODELS_TO_TRAIN from the end of the training data.
    # More simply, training data for model_h needs data up to t+h.
    # If training data ends at T_train_end, the last usable X for model_h is at T_train_end - h.

    if len(merged_data_complete) < min_data_needed:
        print(f"Not enough data ({len(merged_data_complete)} hours) for the desired setup.")
        print(f"Need at least {min_data_needed} hours (max_lag: {max_lag_hours} + training_horizon_shift: {MAX_HORIZON_MODELS_TO_TRAIN} + test_set: {TEST_SET_HOURS}).")
        exit()

    # --- Split Data into Training and Test Sets ---
    # Training data will be all data EXCEPT the last `TEST_SET_HOURS`
    train_data_end_index = len(merged_data_complete) - TEST_SET_HOURS
    training_data_for_dms = merged_data_complete.iloc[:train_data_end_index].copy()
    
    # Test data (actuals) will be the last `TEST_SET_HOURS`
    test_actuals_full_df = merged_data_complete.iloc[train_data_end_index:].copy()
    test_actuals_series = test_actuals_full_df[target_col]

    print(f"Training data shape: {training_data_for_dms.shape}")
    print(f"Test actuals shape: {test_actuals_series.shape}")

    # --- 1. Train DMS Models on the Training Data ---
    # Set to True to re-train, False to load existing if you've run this part before
    if train:
        FORCE_RETRAIN_MODELS = True 
        if FORCE_RETRAIN_MODELS:
            print("\n--- Training DMS Models ---")
            algo.train_all_dms_horizon_models(
                base_data_for_dms_training=training_data_for_dms, # Use only the training portion
                max_forecast_horizon_hours=MAX_HORIZON_MODELS_TO_TRAIN,
                features_list=features_list,
                target_col=target_col
            )
        else:
            print("\n--- Skipping Model Training (assuming models exist) ---")


    # --- 2. Make Predictions for the Test Set Period ---
    print("\n--- Generating Forecast for Test Period ---")
    # The `history_df` for predict_dms should be the data *just before* the test period starts.
    # This is effectively our `training_data_for_dms` because `create_dms_feature_set_for_prediction`
    # will use its last rows to derive lags for the first prediction point of the test period.
    history_for_test_prediction = training_data_for_dms.copy() 

    # Exogenous variables for the test period (if needed by models)
    # These are the *actual* exogenous values from our hold-out test set.
    future_exog_for_test_period = None
    if 'temperature' in features_list: # Or any other exogenous features
        future_exog_for_test_period = test_actuals_full_df[['temperature']].copy() # Ensure it's a DataFrame

    dms_test_predictions = predict_dms(
        history_df=history_for_test_prediction,
        max_horizon_hours=TEST_SET_HOURS, # Predict for the length of our test set
        models_base_path='models/',
        future_exog_series=future_exog_for_test_period
    )

    # --- 3. Evaluate and Visualize ---
    if not dms_test_predictions.empty and not test_actuals_series.empty:
        print(f"\n--- Evaluating Forecast for Test Period (First 5 predictions vs actuals) ---")
        comparison_df_head = pd.DataFrame({'Actual': test_actuals_series, 'Forecast': dms_test_predictions}).dropna().head()
        print(comparison_df_head)

        models_to_inspect_for_viz = {}
        horizons_to_load_for_viz = [1, TEST_SET_HOURS // 2 if TEST_SET_HOURS > 1 else 1, TEST_SET_HOURS]
        horizons_to_load_for_viz = sorted(list(set(h for h in horizons_to_load_for_viz if h > 0)))


        for h_inspect in horizons_to_load_for_viz:
            if h_inspect > MAX_HORIZON_MODELS_TO_TRAIN: # Cant inspect models not trained
                continue
            model_file_viz = f"models/dms_model_horizon_{h_inspect}h.pkl"
            try:
                with open(model_file_viz, 'rb') as f: models_to_inspect_for_viz[h_inspect] = pickle.load(f)
            except FileNotFoundError: print(f"  Warning: Model for h={h_inspect} ({model_file_viz}) not found.")
            except Exception as e_load: print(f"  Error loading model for h={h_inspect}: {e_load}")
        
        data.visualize_dms_forecast(
            dms_forecast_series=dms_test_predictions,
            actual_values_series=test_actuals_series,
            period_label=f"DMS {TEST_SET_HOURS}-Hour Test Set Evaluation",
            models_for_importance=models_to_inspect_for_viz if models_to_inspect_for_viz else None,
            features_list=features_list
        )
    else:
        print("Could not generate predictions or actuals for evaluation.")

    print("\n--- Script Finished ---")