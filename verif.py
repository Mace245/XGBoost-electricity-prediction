import os
import pickle
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import sys

# --- Configuration (Should match app.py and training setup) ---
# Adjust this path if the script is not in the same directory as lib/
# Or ensure lib is in the Python path
try:
    from lib.data import temp_fetch
except ImportError as e:
    print(f"ERROR: Could not import temp_fetch from lib.data: {e}")
    sys.exit(1)

# --- Configuration ---
# --- Files and Paths ---
HISTORICAL_DATA_CSV = 'Data/processed_hourly_Wh_data.csv' # Path to your historical ground truth
MODEL_FILE = 'model.pkl'
OUTPUT_PLOT_FILE = 'verification_plot.png' # Optional plot output

# --- Model/Data Parameters (MUST match app.py/train.py) ---
MODEL_FEATURES = ['hour', 'day_of_week', 'day_of_month', 'is_weekend',
                  'lag_1', 'lag_24', 'lag_168', 'temperature']
TARGET_VARIABLE = 'Wh' # The actual target column name in your CSV
MAX_LAG = 168 # The maximum lag hours needed as input by the model
API_TIMEZONE = "Asia/Jakarta" # Timezone used during training and in app.py

# --- Verification Parameters ---
# How many hours *after* the initial MAX_LAG period do you want to verify?
# Set to None to verify all possible data points after the initial lag period.
VERIFICATION_HOURS = None # Or set to e.g., 24*7 for one week

# Simulate the recursive forecast for this many hours at a time
# This mimics how app.py predicts (e.g., 1 day, 3 days, 1 week)
# Choose a value like 24, 72, or 168
FORECAST_HORIZON = 24

# --- Approximate Location (for fetching historical temperature) ---
# Should be the same as used in training / app.py
LATITUDE = 14.5833
LONGITUDE = 121.0

# --- Helper Functions ---

def create_time_features(df, index_col='DateTime'):
    """Creates time-based features from the DataFrame's index."""
    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
         df.index = pd.to_datetime(df.index)

    data = df.copy()
    # Ensure index is timezone-aware consistent with API_TIMEZONE
    if data.index.tz is None:
        data.index = data.index.tz_localize('UTC').tz_convert(API_TIMEZONE)
    else:
        data.index = data.index.tz_convert(API_TIMEZONE)

    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['day_of_month'] = data.index.day
    data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
    return data

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    non_zero_mask = y_true != 0
    if np.sum(non_zero_mask) == 0:
        return np.inf # Or 0, depending on definition if all true values are zero
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100


# --- Main Verification Script ---

print("--- Starting Verification Script ---")

# 1. Load Historical Ground Truth Data
print(f"Loading historical data from: {HISTORICAL_DATA_CSV}")
if not os.path.exists(HISTORICAL_DATA_CSV):
    print(f"ERROR: Historical data file not found at {HISTORICAL_DATA_CSV}")
    sys.exit(1)
try:
    history_df = pd.read_csv(HISTORICAL_DATA_CSV, parse_dates=['DateTime'])
    history_df.set_index('DateTime', inplace=True)
    # Ensure the target variable column exists
    if TARGET_VARIABLE not in history_df.columns:
        print(f"ERROR: Target variable '{TARGET_VARIABLE}' not found in columns of {HISTORICAL_DATA_CSV}")
        print(f"Available columns: {history_df.columns.tolist()}")
        sys.exit(1)
    print(f"Loaded {len(history_df)} historical records.")
    # Ensure data is sorted by time
    history_df.sort_index(inplace=True)
    # Localize to UTC first (assuming CSV is naive or UTC), then convert to target TZ
    if history_df.index.tz is None:
        history_df = history_df.tz_localize('UTC')
    history_df = history_df.tz_convert(API_TIMEZONE)

except Exception as e:
    print(f"ERROR: Failed to load or process historical data: {e}")
    sys.exit(1)

# Check if enough data exists for the initial lag period
if len(history_df) < MAX_LAG:
    print(f"ERROR: Not enough historical data ({len(history_df)} points) to satisfy the required MAX_LAG ({MAX_LAG}).")
    sys.exit(1)

# 2. Load the Pre-trained Model
print(f"Loading model from: {MODEL_FILE}")
if not os.path.exists(MODEL_FILE):
    print(f"ERROR: Model file not found at {MODEL_FILE}")
    sys.exit(1)
try:
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    # Verify model type if possible (optional)
    if not isinstance(model, xgb.XGBRegressor):
         print(f"Warning: Loaded model might not be an XGBRegressor ({type(model)}).")
    print("Model loaded successfully.")
except Exception as e:
    print(f"ERROR: Failed to load the model: {e}")
    sys.exit(1)

# 3. Fetch Corresponding Historical Temperature Data
print("Fetching historical temperature data...")
try:
    start_date = history_df.index.min().strftime('%Y-%m-%d')
    end_date = history_df.index.max().strftime('%Y-%m-%d')
    print(f"Requesting temperatures from {start_date} to {end_date}...")
    temp_df = temp_fetch(start_date, end_date, LATITUDE, LONGITUDE, historical=True)

    if temp_df is None or temp_df.empty:
         raise ValueError("Temperature data fetch returned empty results.")

    # Convert temp index to the target timezone for merging
    temp_df.index = temp_df.index.tz_convert(API_TIMEZONE)

    # Merge temperature data - Use 'nearest' interpolation for potentially slightly offset timestamps
    # Or ffill/bfill if exact hour alignment is expected
    history_df = pd.merge_asof(history_df.sort_index(), temp_df[['temperature']].sort_index(),
                               left_index=True, right_index=True,
                               direction='nearest', tolerance=pd.Timedelta('30minutes'))

    # Check for missing temperatures after merge and fill (e.g., forward fill)
    missing_temps = history_df['temperature'].isnull().sum()
    if missing_temps > 0:
        print(f"Warning: Found {missing_temps} missing temperature values after merge. Forward filling...")
        history_df['temperature'].ffill(inplace=True)
        # Check again if any remain (e.g., at the very beginning)
        if history_df['temperature'].isnull().any():
             print(f"Warning: Still missing temperatures after ffill. Filling remaining with average.")
             avg_temp = history_df['temperature'].mean()
             history_df['temperature'].fillna(avg_temp, inplace=True)

    print("Temperature data fetched and merged.")

except Exception as e:
    print(f"ERROR: Failed to fetch or merge temperature data: {e}")
    # Optionally, decide if you want to proceed without temperature or exit
    # For verification, it's best to exit if temps are crucial and missing.
    sys.exit(1)


# 4. Prepare Full Dataset with Features (for lag calculation)
print("Preparing data with time features...")
try:
    full_featured_df = create_time_features(history_df[[TARGET_VARIABLE, 'temperature']])
except Exception as e:
    print(f"ERROR: Failed during feature creation: {e}")
    sys.exit(1)

# 5. Simulate Recursive Prediction (Mimicking app.py)
print(f"Simulating recursive prediction with a horizon of {FORECAST_HORIZON} hours...")

all_predictions = []
all_actuals = []
all_timestamps = []

# Determine the range of data points to predict
start_index = MAX_LAG # Start predicting after the initial history needed for lags
end_index = len(full_featured_df)
if VERIFICATION_HOURS is not None:
    end_index = min(start_index + VERIFICATION_HOURS, len(full_featured_df))

print(f"Will attempt to predict from index {start_index} to {end_index -1}.")

current_sim_index = start_index
while current_sim_index < end_index:
    print(f"  Predicting block starting at {full_featured_df.index[current_sim_index]} (Index: {current_sim_index})...")

    # --- Data Setup for this forecast block ---
    # History known *before* this block starts (includes MAX_LAG points)
    known_history_end_index = current_sim_index
    known_history_start_index = max(0, known_history_end_index - MAX_LAG)
    # Use .iloc for integer-based slicing, extract target and temp
    current_history = full_featured_df.iloc[known_history_start_index:known_history_end_index][[TARGET_VARIABLE, 'temperature']].copy()

    # --- Recursive Prediction Loop for this block ---
    predictions_this_block = []
    actuals_this_block = []
    timestamps_this_block = []

    # Determine how many steps to predict in this block
    steps_in_block = min(FORECAST_HORIZON, end_index - current_sim_index)

    for i in range(steps_in_block):
        target_prediction_index = current_sim_index + i
        current_pred_time = full_featured_df.index[target_prediction_index]
        timestamps_this_block.append(current_pred_time)
        actuals_this_block.append(full_featured_df.iloc[target_prediction_index][TARGET_VARIABLE])

        # --- Feature Engineering for the current step ---
        features = {}
        features['hour'] = current_pred_time.hour
        features['day_of_week'] = current_pred_time.dayofweek
        features['day_of_month'] = current_pred_time.day
        features['is_weekend'] = 1 if current_pred_time.dayofweek >= 5 else 0
        # Get the actual temperature for this historical timestamp
        features['temperature'] = full_featured_df.iloc[target_prediction_index]['temperature']
        if pd.isna(features['temperature']):
             # Fallback if temp somehow still missing (should have been handled)
             features['temperature'] = current_history['temperature'].iloc[-1] if not current_history.empty else 25.0


        # Calculate lags using combined known history + previous predictions in *this block*
        temp_combined_target = pd.concat([current_history[TARGET_VARIABLE], pd.Series(predictions_this_block)])

        for lag in [1, 24, 168]:
            lag_time = current_pred_time - timedelta(hours=lag)
            # Find the value at lag_time in the combined history/predictions
            # Use index lookup for timestamp alignment
            try:
                 # Prioritize looking up in the DataFrame index first
                 val = current_history[TARGET_VARIABLE].get(lag_time)
                 if val is None and lag == 1 and predictions_this_block:
                      # If lag 1 is not in history, try the last prediction
                      val = predictions_this_block[-1]
                 # Add other lag logic if needed (e.g., looking up lag 24 in predictions list)
                 # For simplicity here, we primarily use the original history for lags > 1 in verification,
                 # which tests the model slightly differently than pure recursive but is simpler.
                 # To exactly match app.py, you'd need more complex lookup in the `predictions_this_block` list.

                 features[f'lag_{lag}'] = val if pd.notna(val) else 0 # Use 0 if lag value not found/NaN
            except KeyError:
                 features[f'lag_{lag}'] = 0 # Lag falls before start of data


        # Create feature vector in the correct order
        try:
            feature_vector_df = pd.DataFrame([features], columns=MODEL_FEATURES).fillna(0)
        except Exception as e:
             print(f"\nERROR creating feature vector at step {i} for time {current_pred_time}")
             print(f"Features dict: {features}")
             print(f"Error: {e}")
             # Decide how to handle: skip step, break block, exit?
             break # Break current block prediction on error

        # --- Predict ---
        try:
            prediction = max(0.0, float(model.predict(feature_vector_df)[0]))
            predictions_this_block.append(prediction)
        except Exception as e:
            print(f"\nERROR during model.predict at step {i} for time {current_pred_time}")
            print(f"Feature vector:\n{feature_vector_df}")
            print(f"Error: {e}")
            # Append a placeholder like NaN or break
            predictions_this_block.append(np.nan) # Add NaN if prediction fails
            break # Stop predicting this block

    # --- Store results for this block ---
    all_predictions.extend(predictions_this_block)
    all_actuals.extend(actuals_this_block)
    all_timestamps.extend(timestamps_this_block)

    # Move to the next block start index
    current_sim_index += steps_in_block

    # Stop if errors occurred within the block and resulted in early exit
    if len(predictions_this_block) != steps_in_block:
         print("WARN: Prediction block terminated early due to errors.")
         # break # Uncomment to stop entire simulation on first block error

print(f"Simulation finished. Compared {len(all_predictions)} predictions against actual values.")

# 6. Calculate and Print Accuracy Metrics
if not all_predictions or len(all_predictions) != len(all_actuals):
    print("ERROR: Mismatch in prediction/actual counts or no predictions made. Cannot calculate metrics.")
else:
    # Filter out potential NaN predictions if errors occurred
    valid_indices = [i for i, p in enumerate(all_predictions) if pd.notna(p)]
    if len(valid_indices) != len(all_predictions):
        print(f"Warning: {len(all_predictions) - len(valid_indices)} prediction steps failed and were excluded from metrics.")

    actuals_filtered = [all_actuals[i] for i in valid_indices]
    predictions_filtered = [all_predictions[i] for i in valid_indices]
    timestamps_filtered = [all_timestamps[i] for i in valid_indices]

    if not predictions_filtered:
         print("No valid predictions available to calculate metrics.")
    else:
        print("\n--- Accuracy Metrics ---")
        mae = mean_absolute_error(actuals_filtered, predictions_filtered)
        mse = mean_squared_error(actuals_filtered, predictions_filtered)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(actuals_filtered, predictions_filtered)

        print(f"Mean Absolute Error (MAE): {mae:.4f} {TARGET_VARIABLE}")
        print(f"Mean Squared Error (MSE):  {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f} {TARGET_VARIABLE}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print("------------------------")

        # 7. Optional: Plot Actual vs. Predicted
        try:
            plt.figure(figsize=(15, 7))
            plt.plot(timestamps_filtered, actuals_filtered, label='Actual Historical Data', marker='.', linestyle='-', alpha=0.7)
            plt.plot(timestamps_filtered, predictions_filtered, label=f'Simulated Forecast (Horizon={FORECAST_HORIZON}h)', marker='.', linestyle='--', alpha=0.7)
            plt.xlabel("DateTime")
            plt.ylabel(f"Energy ({TARGET_VARIABLE})")
            plt.title(f"Verification: Actual vs. Simulated Forecast (RMSE: {rmse:.2f})")
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(OUTPUT_PLOT_FILE)
            print(f"Plot saved to {OUTPUT_PLOT_FILE}")
            # plt.show() # Uncomment to display plot interactively
        except Exception as e:
            print(f"\nWarning: Could not generate or save plot: {e}")


print("--- Verification Script Finished ---")