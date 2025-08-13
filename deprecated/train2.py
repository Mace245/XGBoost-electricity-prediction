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
    history_df: pd.DataFrame, 
    max_horizon_hours: int,         
    models_base_path: str = 'models/',
    future_exog_series: pd.DataFrame = None 
):
    global features_list, target_col 


    if not isinstance(history_df.index, pd.DatetimeIndex):
        raise ValueError("history_df must have a DatetimeIndex.")
    if history_df.empty:
        raise ValueError("history_df cannot be empty.")

    last_known_time = history_df.index[-1]
    print(f"Generating DMS forecast for {max_horizon_hours} hours, starting after: {last_known_time}")

    all_predictions = []
    forecast_timestamps = []

    try:
        base_features_at_t = data.create_dms_feature_set_for_prediction(
            history_df_slice=history_df, 
            features_list=features_list,
            target_col=target_col
        )
    except Exception as e:
        print(f"Error creating base feature set for prediction at {last_known_time}: {e}")
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
        except Exception as e:
            print(f"  Error loading model for h={h} ({model_filename}): {e}. Appending NaN.")
            all_predictions.append(np.nan)
            continue

        features_for_model_h = base_features_at_t.copy()

        if future_exog_series is not None and not future_exog_series.empty:
            for exog_col_name in future_exog_series.columns:
                if exog_col_name in features_for_model_h.columns: 
                    try:
                        future_val = future_exog_series.loc[current_forecast_dt, exog_col_name]
                        features_for_model_h[exog_col_name] = future_val
                        # print(f"    Updated {exog_col_name} for h={h} to {future_val} for time {current_forecast_dt}")
                    except KeyError:
                        print(f"    Warning: Future value for '{exog_col_name}' at {current_forecast_dt} not found in future_exog_series.")
                        pass
                    except Exception as e_exog:
                        print(f"    Error accessing future exog '{exog_col_name}' for {current_forecast_dt}: {e_exog}")


        # Make the prediction using model_h
        try:
            prediction_value = model_h.predict(features_for_model_h)[0]
            prediction_value = max(0.0, float(prediction_value))
            all_predictions.append(prediction_value)
            # print(f"  h={h}: Predicted {prediction_value:.2f} for {current_forecast_dt}")
        except Exception as e_pred:
            print(f"  Error during prediction with model_h for h={h} ({current_forecast_dt}): {e_pred}. Appending NaN.")
            all_predictions.append(np.nan)

    if not forecast_timestamps:
        return pd.Series(dtype=float)

    final_forecast_series = pd.Series(all_predictions, index=pd.DatetimeIndex(forecast_timestamps, name='DateTime'))
    print(f"DMS forecast generation complete for {max_horizon_hours} hours.")
    return final_forecast_series


if __name__ == "__main__":
    print("Fetching and preparing base historical data...")
    try:
        electricity_raw, temperature_raw = data.fetch_elec_temp()
        if electricity_raw.index.tz is None: electricity_raw.index = electricity_raw.index.tz_localize('Asia/Kuala_Lumpur', ambiguous='infer')
        if temperature_raw.index.tz is None: temperature_raw.index = temperature_raw.index.tz_localize('Asia/Kuala_Lumpur', ambiguous='infer') 

        if str(electricity_raw.index.tz).lower() != 'utc': electricity_raw.index = electricity_raw.index.tz_convert('UTC')
        if str(temperature_raw.index.tz).lower() != 'utc': temperature_raw.index = temperature_raw.index.tz_convert('UTC')
        
        merged_data_complete = data.prepare_data(electricity_raw, temperature_raw)
        print(f"Base historical data prepared. Shape: {merged_data_complete.shape}, TZ: {merged_data_complete.index.tz}")
    except Exception as e:
        print(f"Failed to load or prepare data: {e}")
        import traceback; traceback.print_exc(); exit()

    if merged_data_complete.empty:
        print("Error: Base historical data is empty. Cannot proceed.")
        exit()


    # --- Define Train/Test Split Parameters ---
    MAX_HORIZON_MODELS_TO_TRAIN = TEST_SET_HOURS
    max_lag_hours = 0
    for f in features_list:
        if f.startswith("lag_"):
            try: max_lag_hours = max(max_lag_hours, int(f.split("_")[1]))
            except: pass
    
    min_data_needed = max_lag_hours + MAX_HORIZON_MODELS_TO_TRAIN + TEST_SET_HOURS 

    if len(merged_data_complete) < min_data_needed:
        print(f"Not enough data ({len(merged_data_complete)} hours) for the desired setup.")
        print(f"Need at least {min_data_needed} hours (max_lag: {max_lag_hours} + training_horizon_shift: {MAX_HORIZON_MODELS_TO_TRAIN} + test_set: {TEST_SET_HOURS}).")
        exit()

    # --- Split Data into Training and Test Sets ---
    train_data_end_index = len(merged_data_complete) - TEST_SET_HOURS
    training_data_for_dms = merged_data_complete.iloc[:train_data_end_index].copy()
    
    test_actuals_full_df = merged_data_complete.iloc[train_data_end_index:].copy()
    test_actuals_series = test_actuals_full_df[target_col]

    print(f"Training data shape: {training_data_for_dms.shape}")
    print(f"Test actuals shape: {test_actuals_series.shape}")

    # --- 1. Train DMS Models on the Training Data ---
    if train:
        FORCE_RETRAIN_MODELS = True 
        if FORCE_RETRAIN_MODELS:
            print("\n--- Training DMS Models ---")
            algo.train_all_dms_horizon_models(
                base_data_for_dms_training=training_data_for_dms,
                max_forecast_horizon_hours=MAX_HORIZON_MODELS_TO_TRAIN,
                features_list=features_list,
                target_col=target_col
            )
        else:
            print("\n--- Skipping Model Training (assuming models exist) ---")


    # --- 2. Make Predictions for the Test Set Period ---
    print("\n--- Generating Forecast for Test Period ---")
    history_for_test_prediction = training_data_for_dms.copy() 

    future_exog_for_test_period = None
    if 'temperature' in features_list: 
        future_exog_for_test_period = test_actuals_full_df[['temperature']].copy() 

    dms_test_predictions = predict_dms(
        history_df=history_for_test_prediction,
        max_horizon_hours=TEST_SET_HOURS,
        models_base_path='models/',
        future_exog_series=future_exog_for_test_period
    )

    print(f"\n--- Evaluating Forecast for Test Period (First 5 predictions vs actuals) ---")
    comparison_df_head = pd.DataFrame({'Actual': test_actuals_series, 'Forecast': dms_test_predictions}).dropna().head()

    models_to_inspect_for_viz = {}
    horizons_to_load_for_viz = [1, TEST_SET_HOURS // 2 if TEST_SET_HOURS > 1 else 1, TEST_SET_HOURS]
    horizons_to_load_for_viz = sorted(list(set(h for h in horizons_to_load_for_viz if h > 0)))


    for h_inspect in horizons_to_load_for_viz:
        if h_inspect > MAX_HORIZON_MODELS_TO_TRAIN:
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
    )


    print("\n--- Script Finished ---")