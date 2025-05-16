import algo
import data
import pickle
import pandas as pd

train = True

# Define features_list and target_col (as before)
features_list = [
    'hour', 'day_of_week', 'day_of_month', 'is_weekend',
    'lag_1', 'lag_24', 'lag_168', # Max lag is 168 hours (7 days)
    'temperature'
]
target_col = 'Wh'

def train_all_dms_horizon_models(
    # Parameter name changed for clarity
    base_data_for_dms_training: pd.DataFrame,
    max_forecast_horizon_hours: int
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
        model_for_h = algo.train_xgboost_model(
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


# --- Main Execution Example (How to run the training with FILTERED data) ---
if __name__ == "__main__":
    print("Fetching and preparing base historical data...")
    electricity_raw, temperature_raw = data.fetch_elec_temp()
    merged_data_complete = data.prepare_data(electricity_raw, temperature_raw)
    print(f"Full base historical data prepared. Shape: {merged_data_complete.shape}")
    if train:
        if merged_data_complete.empty:
            print("Error: Base historical data is empty. Cannot proceed with training.")
        else:
            # --- FILTERING FOR TESTING ---
            # Let's define a period for testing, e.g., 30 days of data
            # Ensure this period is long enough for lags + horizon shifts + some training samples.
            # Max lag is 168 hours (7 days). Let's say max horizon for test is 3 days (72 hours).
            # So, we need at least 7 days (lags) + 3 days (horizon) = 10 days of data that *won't* be NaN.
            # Add some more for actual training samples. Let's take 30 days.
            
            num_days_for_testing = 30
            required_hours_for_testing = num_days_for_testing * 24

            if len(merged_data_complete) >= required_hours_for_testing:
                # Take a slice, e.g., the most recent `required_hours_for_testing`
                # Or a slice from the middle, or the beginning.
                # For reproducibility, taking from the start might be easier if data changes.
                # test_data_slice = merged_data_complete.head(required_hours_for_testing).copy()
                test_data_slice = merged_data_complete.tail(required_hours_for_testing).copy() # More realistic for recent data
                print(f"Using a filtered slice for testing. Shape: {test_data_slice.shape}")
            else:
                print(f"Full dataset is too small ({len(merged_data_complete)} hours) for the desired test slice of {required_hours_for_testing} hours.")
                print("Using the full dataset for testing instead (if it's not too large for a quick test).")
                test_data_slice = merged_data_complete.copy()
                if len(test_data_slice) > 5000: # Arbitrary limit for a "quick" test
                    print("Warning: Full dataset is large, testing might be slow. Consider a smaller slice if possible.")


            # Define the maximum forecast horizon for this TEST run
            MAX_HORIZON_TEST = 3 * 24  # Test for 3 days (72 hours)
            # MAX_HORIZON_TEST = 24 # Or just 1 day for a very quick test

            # Call the training function with the filtered data slice
            train_all_dms_horizon_models(
                base_data_for_dms_training=test_data_slice, # Pass the filtered data
                max_forecast_horizon_hours=MAX_HORIZON_TEST
            )

            print("\n--- DMS Training Script (TEST RUN) Finished ---")

    history_slice_for_pred = merged_data_complete.iloc[-200:].copy()
    
    # Define how many hours to forecast
    forecast_horizon = 24 # Predict next 24 hours

    # --- Optional: Prepare future exogenous variables ---
    # If your models were trained with future temperature (e.g., temp at t+h),
    # you MUST provide these future values.
    # For this example, let's simulate having perfect future temperature forecasts
    # by taking them from our `merged_data_complete` (cheating for demo purposes).
    
    # Get the last timestamp from our history slice
    last_hist_time = history_slice_for_pred.index[-1]
    
    # Define the forecast period timestamps
    future_timestamps = pd.date_range(
        start=last_hist_time + pd.Timedelta(hours=1),
        periods=forecast_horizon,
        freq=history_slice_for_pred.index.freq # Use same frequency as history
    )

    # Create a dummy future exogenous series (e.g., for 'temperature')
    # In a real scenario, this would come from a weather forecast API or model.
    simulated_future_temps_data = {}
    if 'temperature' in features_list: # Only if temperature is a feature
        # Try to get actual future temperatures if they exist in our full dataset beyond history_slice_for_pred
        # This is for example purposes; in reality, these would be true forecasts.
        actual_future_data_slice = merged_data_complete[merged_data_complete.index.isin(future_timestamps)]
        if not actual_future_data_slice.empty and 'temperature' in actual_future_data_slice:
            simulated_future_temps_data['temperature'] = actual_future_data_slice['temperature'].values
        else:
            # Fallback: just repeat the last known temperature (very basic)
            last_known_temp = history_slice_for_pred['temperature'].iloc[-1]
            simulated_future_temps_data['temperature'] = [last_known_temp] * forecast_horizon
            print("Warning: Could not get actual future temperatures for example; using last known temp.")

    future_exog_df = pd.DataFrame(simulated_future_temps_data, index=future_timestamps)
    # --- End of Optional Future Exogenous Variables ---

    try:
        print(f"\nCalling predict_dms with history ending: {history_slice_for_pred.index[-1]}")
        print(f"Future exogenous data for prediction (first 5 rows if available):\n{future_exog_df.head()}")

        dms_predictions = algo.predict_dms(
            history_df=history_slice_for_pred,
            max_horizon_hours=forecast_horizon,
            models_base_path='models/', # Ensure your models are here
            future_exog_series=future_exog_df if not future_exog_df.empty else None
        )

        print("\n--- DMS Forecast Results ---")
        if not dms_predictions.empty:
            print(dms_predictions)

            # You could then call your visualization function here if you have actuals for this period
            # data.visualize_dms_forecast(dms_predictions, actuals_for_this_period, ...)
        else:
            print("Prediction returned an empty series.")

    except Exception as e:
        print(f"An error occurred during the predict_dms example: {e}")
        import traceback
        traceback.print_exc()