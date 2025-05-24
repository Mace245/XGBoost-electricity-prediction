# app.py
import os
import time
import threading
import pickle
from datetime import datetime, timedelta, timezone as dt_timezone

import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from antares_http import antares
from lib import data 
from lib import algo 

# --- Configuration ---
ANTARES_ACCESS_KEY = '5cd4cda046471a89:75f9e1c6b34bf41a'
ANTARES_PROJECT_NAME = 'UjiCoba_TA'
ANTARES_DEVICE_NAME = 'TA_DKT1'

DATABASE_FILE = 'energy_app_dms_simplified.db'

BACKGROUND_FETCH_INTERVAL_SECONDS = 15 * 60

LATITUDE_CONFIG = 14.5833
LONGITUDE_CONFIG = 121.0
APP_DISPLAY_TIMEZONE = "Asia/Kuala_Lumpur"
# All internal processing and DB storage will be UTC.

DMS_MODELS_BASE_PATH = 'models_dms/'
MAX_FORECAST_HORIZON_APP = 24 * 7

DMS_FEATURES_LIST = ['hour', 'day_of_week', 'day_of_month', 'is_weekend',
                     'lag_1', 'lag_24', 'lag_168', 'temperature']
DMS_TARGET_COL_DATAFRAME = 'Wh'    # Name of the target column in DataFrames for algo/data modules
DB_TARGET_COL_NAME = 'EnergyWh'    # Name of the target column in the DB Model
DB_TEMP_COL_NAME = 'TemperatureCelsius' # Name of the temperature column in the DB Model

MAX_LAG_HOURS = 0
for f_name in DMS_FEATURES_LIST:
    if f_name.startswith("lag_"):
        try: MAX_LAG_HOURS = max(MAX_LAG_HOURS, int(f_name.split("_")[1]))
        except: pass

# Retraining Configuration
RETRAIN_CHECK_INTERVAL_SECONDS = 3600 * 6 # Check for retraining eligibility every 6 hours
RETRAIN_TRIGGER_DAY = 6 # Sunday (0=Monday, 6=Sunday)
RETRAIN_TRIGGER_HOUR_UTC = 2 # 2 AM UTC on Sunday (time for potentially lower server load)

# --- Initialize App, DB ---
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DATABASE_FILE}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.urandom(24) # For flash messages
db = SQLAlchemy(app)
# Make datetime.now(dt_timezone.utc) available in templates
app.jinja_env.globals['utc_now'] = lambda: datetime.now(dt_timezone.utc)


# --- Database Model ---
class HourlyReading(db.Model): # Renamed for clarity
    id = db.Column(db.Integer, primary_key=True)
    timestamp_utc = db.Column(db.String(20), nullable=False, unique=True, index=True)
    EnergyWh = db.Column(db.Float)
    TemperatureCelsius = db.Column(db.Float, nullable=True)

    def __repr__(self):
        return f"<HourlyReading {self.timestamp_utc} - Energy: {self.EnergyWh}>"

# --- Background Data Collection ---
last_fetched_hour_utc = None # Keep track of the last hour we successfully fetched data for


def background_data_collector():
    """Periodically fetches data for the CURRENT hour from Antares and temperature forecast."""
    global last_fetched_hour_utc # If you still want to track the last processed hour to avoid re-processing within the same interval
    print("Simplified Background Data Collector started (no catch-up)...")

    # Initialize last_fetched_hour_utc based on DB, but only to prevent immediate re-fetch of current hour if app restarts
    with app.app_context():
        latest_db_entry = HourlyReading.query.order_by(HourlyReading.timestamp_utc.desc()).first()
        if latest_db_entry:
            try:
                # Parse the timestamp and floor it to the hour
                parsed_ts = pd.to_datetime(latest_db_entry.timestamp_utc).tz_convert('UTC') # Ensure UTC
                last_fetched_hour_utc = parsed_ts.replace(minute=0, second=0, microsecond=0)
                print(f"Data Collector: Last stored hour in DB: {last_fetched_hour_utc.isoformat()}")
            except Exception as e:
                print(f"Data Collector: Error parsing last DB timestamp: {e}. Will proceed as if no prior data for this session.")
                last_fetched_hour_utc = None # Reset if parsing fails
        else:
            print("Data Collector: No existing data in DB.")
            last_fetched_hour_utc = None

    while True:
        current_time_utc = datetime.now(dt_timezone.utc)
        # Target the beginning of the current hour
        target_hour_to_process_utc = current_time_utc.replace(minute=0, second=0, microsecond=0)
        target_hour_iso = target_hour_to_process_utc.isoformat()

        print(f"Data Collector: Current target hour is {target_hour_iso}")

        # Check if this hour was already processed in a *very recent* previous iteration of this loop
        # This 'last_fetched_hour_utc' primarily prevents re-processing if the script loops faster than an hour.
        if last_fetched_hour_utc is not None and target_hour_to_process_utc <= last_fetched_hour_utc:
            print(f"Data Collector: Hour {target_hour_iso} already processed or not yet new. Last was {last_fetched_hour_utc.isoformat()}. Sleeping.")
            time.sleep(BACKGROUND_FETCH_INTERVAL_SECONDS)
            continue

        with app.app_context(): # Ensure DB operations are within app context
            # Check if data for this specific hour ALREADY EXISTS in the database
            existing_reading_for_target_hour = HourlyReading.query.filter_by(timestamp_utc=target_hour_iso).first()

            if existing_reading_for_target_hour:
                print(f"  Data for {target_hour_iso} already exists in DB. Checking for updates...")
                # --- Logic to potentially update existing record (Optional) ---
                # You might want to update if, for example, Antares provides a more accurate cumulative value later in the hour.
                # For simplicity here, we'll assume if it exists, we might only update if a field was NULL.
                # Or, if your 'Energy' is cumulative, you might want to fetch latest and update.
                # This part depends heavily on your data's nature.

                # Example: Update if energy was NULL or if new energy is higher (if EnergyWh is cumulative hourly max)
                # Fetch fresh Antares data for potential update
                new_energy_wh_value = None
                try:
                    antares.setAccessKey(ANTARES_ACCESS_KEY)
                    latest_antares_data = antares.get(ANTARES_PROJECT_NAME, ANTARES_DEVICE_NAME)
                    if latest_antares_data and 'content' in latest_antares_data:
                        new_energy_wh_value = latest_antares_data['content'].get('Energy')
                except Exception as e_ant_update:
                    print(f"  Error fetching Antares for update check on {target_hour_iso}: {e_ant_update}")

                commit_update_needed = False
                if new_energy_wh_value is not None: # Only proceed if we got new energy data
                    try:
                        new_energy_wh_value_float = float(new_energy_wh_value) # Attempt conversion
                        if existing_reading_for_target_hour.EnergyWh is None or \
                           new_energy_wh_value_float > existing_reading_for_target_hour.EnergyWh: # Example update condition
                            print(f"  Updating Energy for {target_hour_iso}. Stored: {existing_reading_for_target_hour.EnergyWh}, New: {new_energy_wh_value_float}")
                            existing_reading_for_target_hour.EnergyWh = new_energy_wh_value_float
                            commit_update_needed = True
                    except (ValueError, TypeError) as e_conv:
                         print(f"  Antares energy value '{new_energy_wh_value}' for update check on {target_hour_iso} is not a valid number: {e_conv}")


                # Example: Update temperature if it was NULL
                if existing_reading_for_target_hour.TemperatureCelsius is None:
                    print(f"  Temperature for {target_hour_iso} was NULL, attempting to fetch and update.")
                    new_temperature_value = None
                    try:
                        date_str_utc = target_hour_to_process_utc.strftime('%Y-%m-%d')
                        temp_df = data.temp_fetch(date_str_utc, date_str_utc, LATITUDE_CONFIG, LONGITUDE_CONFIG, historical=False) # Assuming historical=False gets current/forecast
                        if temp_df is not None and 'temperature' in temp_df:
                            temp_series = temp_df['temperature']
                            new_temperature_value = temp_series.get(target_hour_to_process_utc) # Get temp for the specific target hour
                            if pd.notna(new_temperature_value):
                                existing_reading_for_target_hour.TemperatureCelsius = float(new_temperature_value)
                                commit_update_needed = True
                            else:
                                print(f"  Fetched temperature for {target_hour_iso} update was NaN/None.")
                    except Exception as e_temp_update:
                        print(f"  Error fetching temperature for update on {target_hour_iso}: {e_temp_update}")

                if commit_update_needed:
                    try:
                        db.session.commit()
                        print(f"  Updated DB for {target_hour_iso}. Energy: {existing_reading_for_target_hour.EnergyWh}, Temp: {existing_reading_for_target_hour.TemperatureCelsius}")
                    except Exception as e_db_update:
                        db.session.rollback()
                        print(f"  Error committing update for {target_hour_iso}: {e_db_update}")
                else:
                    print(f"  No updates needed for existing record {target_hour_iso}.")

                last_fetched_hour_utc = target_hour_to_process_utc # Mark this hour as processed for this cycle
                time.sleep(BACKGROUND_FETCH_INTERVAL_SECONDS)
                continue # Move to next cycle of the main while loop

            # --- If data for target_hour_iso does NOT exist, create new record ---
            print(f"  No existing record for {target_hour_iso}. Fetching new data...")
            energy_wh_value = None
            # 1. Fetch from Antares
            try:
                antares.setAccessKey(ANTARES_ACCESS_KEY)
                latest_antares_data = antares.get(ANTARES_PROJECT_NAME, ANTARES_DEVICE_NAME)
                if latest_antares_data and 'content' in latest_antares_data:
                    raw_energy = latest_antares_data['content'].get('Energy')
                    if raw_energy is not None:
                        try:
                            energy_wh_value = float(raw_energy) # Convert to float immediately
                        except (ValueError, TypeError) as e_conv:
                            print(f"  Antares 'Energy' value '{raw_energy}' for {target_hour_iso} is not a valid number: {e_conv}")
                            energy_wh_value = None # Ensure it's None if conversion fails
                    else:
                        print(f"  Antares data for {target_hour_iso} missing 'Energy' key or value is None.")
                else:
                    print(f"  No valid content from Antares for {target_hour_iso}.")
            except Exception as e_ant:
                print(f"  Error fetching from Antares for {target_hour_iso}: {e_ant}")

            if energy_wh_value is None: # Check after attempting conversion
                print(f"  Skipping DB store for {target_hour_iso} due to missing or invalid Antares energy data.")
                # Not updating last_fetched_hour_utc here, so it will be retried in the next interval
                time.sleep(BACKGROUND_FETCH_INTERVAL_SECONDS)
                continue

            # 2. Fetch Temperature Forecast for this specific hour_to_process
            temperature_value_float = None # Initialize as float or None
            try:
                date_str_utc = target_hour_to_process_utc.strftime('%Y-%m-%d')
                # Assuming temp_fetch is robust and returns a DataFrame with a 'temperature' Series (DatetimeIndexed) or None
                temp_df = data.temp_fetch(date_str_utc, date_str_utc, LATITUDE_CONFIG, LONGITUDE_CONFIG, historical=False)
                if temp_df is not None and 'temperature' in temp_df:
                    temp_series = temp_df['temperature']
                    # Ensure target_hour_to_process_utc is timezone-aware if temp_series.index is
                    # (already done by datetime.now(dt_timezone.utc))
                    raw_temp = temp_series.get(target_hour_to_process_utc) # Get temp for the specific target hour
                    if pd.notna(raw_temp): # Check if it's a valid number (not NaN)
                        temperature_value_float = float(raw_temp)
                    else:
                        print(f"  Temperature for {target_hour_iso} from API was NaN/None.")
                else:
                    print(f"  Temperature data not available or in unexpected format from temp_fetch for {target_hour_iso}.")
            except Exception as e_temp:
                print(f"  Error fetching/processing temperature for {target_hour_iso}: {e_temp}")
                # temperature_value_float remains None

            # 3. Store in Database
            new_entry = HourlyReading(
                timestamp_utc=target_hour_iso,
                EnergyWh=energy_wh_value, # Already a float or this point wouldn't be reached
                TemperatureCelsius=temperature_value_float # Already a float or None
            )
            db.session.add(new_entry)

            try:
                db.session.commit()
                # Safe printing, assuming EnergyWh and TemperatureCelsius are now guaranteed to be float or None
                energy_print_val = f"{new_entry.EnergyWh:.2f}" if new_entry.EnergyWh is not None else "N/A"
                temp_print_val = f"{new_entry.TemperatureCelsius:.2f}" if new_entry.TemperatureCelsius is not None else "N/A"
                print(f"  Stored new record: {new_entry.timestamp_utc} - Energy: {energy_print_val}, Temp: {temp_print_val}")
                last_fetched_hour_utc = target_hour_to_process_utc # Successfully stored and processed
            except Exception as e_db_commit:
                db.session.rollback()
                # Re-construct the print string carefully for the error message if new_entry might be in a weird state
                # or simply print the raw error and basic info.
                energy_val_on_fail = new_entry.EnergyWh if hasattr(new_entry, 'EnergyWh') else 'UNKNOWN_ENERGY'
                temp_val_on_fail = new_entry.TemperatureCelsius if hasattr(new_entry, 'TemperatureCelsius') else 'UNKNOWN_TEMP'

                print(f"  DB Commit Error for new record {target_hour_iso}: {e_db_commit}.")
                print(f"    Attempted to store: Energy={repr(energy_val_on_fail)}, Temp={repr(temp_val_on_fail)}")
                # Not updating last_fetched_hour_utc, so it will be retried
        
        time.sleep(BACKGROUND_FETCH_INTERVAL_SECONDS)


# --- Model Retraining ---
last_retrain_completed_utc = datetime.now(dt_timezone.utc) - timedelta(days=8) # Ensure first check can trigger
retraining_active = False # Flag to prevent concurrent retraining

def schedule_dms_retraining():
    """Initiates the DMS model retraining process."""
    global retraining_active, last_retrain_completed_utc
    if retraining_active:
        print("Retraining is already active. New trigger ignored.")
        return

    retraining_active = True
    print("\n--- INITIATING DMS MODEL RETRAINING ---")
    try:
        with app.app_context(): # Required for database access within the thread
            # 1. Fetch all data from the app's database for training
            training_df_utc = data.get_all_data_from_db_for_training(
                db.session, HourlyReading,
                output_df_target_col=DMS_TARGET_COL_DATAFRAME, # 'Wh'
                output_df_temp_col='temperature',           # 'temperature'
                model_actual_target_attr=DB_TARGET_COL_NAME,    # 'EnergyWh'
                model_actual_temp_attr=DB_TEMP_COL_NAME     # 'TemperatureCelsius'
            )

            if training_df_utc.empty or len(training_df_utc) < (MAX_LAG_HOURS + MAX_FORECAST_HORIZON_APP + 24): # Min data
                print(f"  Insufficient data for retraining ({len(training_df_utc)} records). "
                      f"Need > {MAX_LAG_HOURS + MAX_FORECAST_HORIZON_APP + 24}. Retraining aborted.")
                retraining_active = False
                return

            print(f"  Retraining DMS models with {len(training_df_utc)} data points.")
            # 2. Call the main training function from algo.py
            algo.train_all_dms_horizon_models(
                base_data_for_dms_training=training_df_utc,
                max_forecast_horizon_hours=MAX_FORECAST_HORIZON_APP,
                features_list=DMS_FEATURES_LIST,
                target_col=DMS_TARGET_COL_DATAFRAME,
                models_save_path=DMS_MODELS_BASE_PATH
            )
            last_retrain_completed_utc = datetime.now(dt_timezone.utc)
            print(f"--- DMS MODEL RETRAINING COMPLETED: {last_retrain_completed_utc.isoformat()} ---")
            flash("DMS models have been successfully retrained.", "success") # Won't show unless a request context is active

    except Exception as e:
        print(f"--- FATAL ERROR DURING DMS RETRAINING: {e} ---")
        import traceback; traceback.print_exc()
        flash(f"Error during model retraining: {e}", "error")
    finally:
        retraining_active = False

def background_retraining_scheduler():
    """Checks periodically if it's time to retrain models."""
    print("Background Retraining Scheduler started...")
    global last_retrain_completed_utc
    while True:
        now_utc = datetime.now(dt_timezone.utc)
        if (now_utc.weekday() == RETRAIN_TRIGGER_DAY and
            now_utc.hour == RETRAIN_TRIGGER_HOUR_UTC and
            (now_utc - last_retrain_completed_utc).days >= 7): # Check if at least 7 days passed
            print(f"Retraining condition met (Day: {now_utc.weekday()}, Hour: {now_utc.hour} UTC). Last: {last_retrain_completed_utc.date()}")
            schedule_dms_retraining() # This will run in the current thread.
                                     # For long retraining, consider starting a new thread for schedule_dms_retraining itself.
        time.sleep(RETRAIN_CHECK_INTERVAL_SECONDS)


# --- Flask Routes ---
@app.route('/')
def home():
    return redirect(url_for('forecast_view'))

@app.route('/database_log')
def database_log_view():
    page = request.args.get('page', 1, type=int)
    per_page = 50 # Show 50 records per page
    pagination = HourlyReading.query.order_by(HourlyReading.timestamp_utc.desc()).paginate(page=page, per_page=per_page, error_out=False)
    readings_for_page = pagination.items
    # Convert UTC strings to display timezone for the template
    readings_display = []
    for r in readings_for_page:
        dt_utc = pd.to_datetime(r.timestamp_utc)
        dt_display = dt_utc.tz_convert(APP_DISPLAY_TIMEZONE)
        readings_display.append({
            'id': r.id,
            'timestamp_display': dt_display.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'EnergyWh': r.EnergyWh,
            'TemperatureCelsius': r.TemperatureCelsius
        })
    return render_template('database_log.html', readings=readings_display, pagination=pagination, APP_DISPLAY_TIMEZONE=APP_DISPLAY_TIMEZONE)

@app.route('/fetch_graph_data_db_log') # New specific endpoint for this graph
def fetch_graph_data_db_log_api():
    start_date_str = request.args.get('start_date') # Expected YYYY-MM-DD
    end_date_str = request.args.get('end_date')     # Expected YYYY-MM-DD

    if not (start_date_str and end_date_str):
        return jsonify({"error": "Start and end dates are required."}), 400
    try:
        # Assume input dates are 'local' to the app's display, convert to UTC for query
        start_dt_local = pd.to_datetime(start_date_str).replace(hour=0, minute=0, second=0)
        end_dt_local = pd.to_datetime(end_date_str).replace(hour=23, minute=59, second=59)

        start_dt_utc = start_dt_local.tz_localize(APP_DISPLAY_TIMEZONE, ambiguous='infer').tz_convert('UTC')
        end_dt_utc = end_dt_local.tz_localize(APP_DISPLAY_TIMEZONE, ambiguous='infer').tz_convert('UTC')

        query_start_utc_iso = start_dt_utc.isoformat().replace('+00:00', 'Z')
        query_end_utc_iso = end_dt_utc.isoformat().replace('+00:00', 'Z')

    except Exception as e:
        return jsonify({"error": f"Invalid date format: {e}"}), 400

    readings = HourlyReading.query.filter(
        HourlyReading.timestamp_utc >= query_start_utc_iso,
        HourlyReading.timestamp_utc <= query_end_utc_iso
    ).order_by(HourlyReading.timestamp_utc).all()

    if not readings:
        return jsonify({"labels": [], "data": []}) # Return empty if no data for range

    # Prepare data for chart, converting timestamps to APP_DISPLAY_TIMEZONE for labels
    labels = []
    data_values = []
    for r in readings:
        dt_utc = pd.to_datetime(r.timestamp_utc)
        dt_display = dt_utc.tz_convert(APP_DISPLAY_TIMEZONE)
        labels.append(dt_display.strftime('%Y-%m-%d %H:%M')) # Format for chart label
        data_values.append(r.EnergyWh if r.EnergyWh is not None else 0)

    return jsonify({"labels": labels, "data": data_values})


@app.route('/forecast')
def forecast_view():
    latest_reading_db = HourlyReading.query.order_by(HourlyReading.timestamp_utc.desc()).first()
    latest_reading_display = None
    if latest_reading_db:
        dt_utc = pd.to_datetime(latest_reading_db.timestamp_utc)
        dt_display = dt_utc.tz_convert(APP_DISPLAY_TIMEZONE)
        latest_reading_display = {
            "timestamp_display": dt_display.strftime('%Y-%m-%d %H:%M:%S %Z'),
            "EnergyWh": f"{latest_reading_db.EnergyWh:.2f}" if latest_reading_db.EnergyWh is not None else "N/A",
            "TemperatureCelsius": f"{latest_reading_db.TemperatureCelsius:.2f}" if latest_reading_db.TemperatureCelsius is not None else "N/A"
        }
    return render_template('forecast.html',
                           latest_reading=latest_reading_display,
                           max_forecast_hours=MAX_FORECAST_HORIZON_APP)


@app.route('/run_forecast_dms', methods=['POST'])
def run_forecast_dms_api():
    req_data = request.get_json()
    timeframe_selected_str = req_data.get('timeframe') # e.g., "24h"
    try:
        hours_to_forecast = int(timeframe_selected_str.replace('h', ''))
        if not (0 < hours_to_forecast <= MAX_FORECAST_HORIZON_APP):
            raise ValueError("Requested forecast hours exceed maximum trained horizon or is invalid.")
    except (ValueError, TypeError, AttributeError):
        return jsonify({"error": f"Invalid timeframe. Max is {MAX_FORECAST_HORIZON_APP}h."}), 400

    # --- Fetch Historical Data ---
    # We need:
    # 1. History for lag feature creation (MAX_LAG_HOURS) ending at T-1 (where T is forecast start)
    # 2. History for display (same length as hours_to_forecast) ending at T-1
    
    # Total historical points needed from DB: MAX_LAG_HOURS (for features) + hours_to_forecast (for display context)
    # Ensure we get enough for the *oldest* point needed for lags of the display history.
    # More simply: get MAX_LAG_HOURS + hours_to_forecast, all ending at the most recent data point.
    
    num_records_to_fetch = MAX_LAG_HOURS + hours_to_forecast + 5 # +5 buffer for safety

    history_query_results = HourlyReading.query.order_by(HourlyReading.timestamp_utc.desc()).limit(num_records_to_fetch).all()

    if len(history_query_results) < MAX_LAG_HOURS : # Need at least enough for lags for the first forecast point
        return jsonify({"error": f"Insufficient historical data in DB ({len(history_query_results)} records). "
                                 f"Need at least {MAX_LAG_HOURS} for lags."}), 400
    
    if len(history_query_results) < hours_to_forecast + 1 and hours_to_forecast > 0 : # +1 because one point is the T-1 for lags
        print(f"Warning: Not enough historical data ({len(history_query_results)}) to show full {hours_to_forecast}h preceding context. "
              f"Will show what's available.")


    # Convert query results to DataFrame, oldest first, with UTC DatetimeIndex
    history_list_for_df = []
    for r_hist in history_query_results:
        history_list_for_df.append({
            'DateTime': pd.to_datetime(r_hist.timestamp_utc, utc=True),
            DMS_TARGET_COL_DATAFRAME: r_hist.EnergyWh,
            'temperature': r_hist.TemperatureCelsius
        })
    full_history_df_utc = pd.DataFrame(history_list_for_df).set_index('DateTime').iloc[::-1].sort_index()

    if full_history_df_utc.empty:
         return jsonify({"error": "Failed to construct historical DataFrame for prediction."}), 500

    # history_df_for_prediction_features will be the part used by algo.predict_dms
    # It needs to end at the last actual data point to generate features for T+1
    history_df_for_prediction_features = full_history_df_utc.copy() # Or full_history_df_utc.tail(MAX_LAG_HOURS + few)

    # history_for_display will be the 'hours_to_forecast' period *before* the forecast starts
    # It ends at the same time as history_df_for_prediction_features.index[-1]
    history_for_display_utc = full_history_df_utc.tail(hours_to_forecast).copy() if hours_to_forecast > 0 else pd.DataFrame()


    # --- Fetch Future Temperatures (Forecast) ---
    last_known_history_time_utc = history_df_for_prediction_features.index[-1]
    forecast_period_start_utc = last_known_history_time_utc + timedelta(hours=1)
    forecast_period_end_utc = last_known_history_time_utc + timedelta(hours=hours_to_forecast)
    future_temperatures_df = None 
    try:
        start_date_utc = forecast_period_start_utc.strftime('%Y-%m-%d')
        end_date_utc = forecast_period_end_utc.strftime('%Y-%m-%d')
        temp_fcst_utc = data.temp_fetch(start_date_utc, end_date_utc, LATITUDE_CONFIG, LONGITUDE_CONFIG, historical=False)
        if temp_fcst_utc is not None and not temp_fcst_utc.empty:
            required_future_idx_utc = pd.date_range(start=forecast_period_start_utc,
                                                    end=forecast_period_end_utc,
                                                    freq='h', tz='UTC')
            aligned_temps = temp_fcst_utc['temperature'].reindex(required_future_idx_utc, method='ffill').fillna(method='bfill')
            future_temperatures_df = pd.DataFrame({'temperature': aligned_temps})
            if future_temperatures_df['temperature'].isna().any():
                last_hist_temp = history_df_for_prediction_features['temperature'].dropna().iloc[-1] if not history_df_for_prediction_features['temperature'].dropna().empty else 25.0
                future_temperatures_df['temperature'] = future_temperatures_df['temperature'].fillna(last_hist_temp)
    except Exception as e:
        print(f"Warning: Exception during future temperature fetch: {e}. Using fallback.")
    if future_temperatures_df is None or future_temperatures_df.empty:
        print("  Executing temperature fallback for forecast period.")
        last_hist_temp = history_df_for_prediction_features['temperature'].dropna().iloc[-1] if not history_df_for_prediction_features['temperature'].dropna().empty else 25.0
        required_future_idx_utc = pd.date_range(start=forecast_period_start_utc,
                                                end=forecast_period_end_utc,
                                                freq='h', tz='UTC')
        future_temperatures_df = pd.DataFrame({'temperature': last_hist_temp}, index=required_future_idx_utc)


    # --- Call DMS Prediction ---
    try:
        dms_predictions_series_utc = algo.predict_dms(
            history_df=history_df_for_prediction_features, # Use the history DataFrame
            max_horizon_hours=hours_to_forecast,
            features_list=DMS_FEATURES_LIST,
            target_col=DMS_TARGET_COL_DATAFRAME,
            models_base_path=DMS_MODELS_BASE_PATH,
            future_exog_series=future_temperatures_df
        )
    # ... (Error handling for prediction remains the same) ...
    except Exception as e_pred_dms:
        print(f"Error during DMS prediction call: {e_pred_dms}")
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Forecast generation error: {str(e_pred_dms)}"}), 500

    if dms_predictions_series_utc.empty:
        return jsonify({"error": "Forecast generation resulted in no prediction data."}), 500

    # --- Prepare data for the chart ---
    # 1. Historical data for display
    history_labels_display = []
    history_data_display = []
    if not history_for_display_utc.empty:
        history_display_local = history_for_display_utc.tz_convert(APP_DISPLAY_TIMEZONE)
        history_labels_display = [dt.strftime('%Y-%m-%d %H:%M') for dt in history_display_local.index]
        history_data_display = [round(val, 2) if pd.notna(val) else None for val in history_display_local[DMS_TARGET_COL_DATAFRAME].values]

    # 2. Forecast data for display
    forecast_labels_display = []
    forecast_data_display = []
    if not dms_predictions_series_utc.empty:
        predictions_display_local = dms_predictions_series_utc.tz_convert(APP_DISPLAY_TIMEZONE)
        forecast_labels_display = [dt.strftime('%Y-%m-%d %H:%M') for dt in predictions_display_local.index]
        forecast_data_display = [round(p_val, 2) if pd.notna(p_val) else None for p_val in predictions_display_local.values]

    return jsonify({
        "history_labels": history_labels_display,
        "history_data": history_data_display,
        "forecast_labels": forecast_labels_display, # Renamed from "labels"
        "forecast_data": forecast_data_display    # Renamed from "data"
    })


@app.route('/trigger_manual_retrain', methods=['POST'])
def trigger_manual_retrain_route():
    global retraining_active # Ensure global flag is used
    if retraining_active:
        flash('Retraining is already in progress. Please wait.', 'warning')
    else:
        flash('Manual model retraining has been triggered. This may take several minutes. Check server logs for progress and completion.', 'info')
        # Run retraining in a new thread to avoid blocking the web request
        retrain_thread = threading.Thread(target=schedule_dms_retraining)
        retrain_thread.start()
    return redirect(url_for('forecast_view'))


@app.route('/model_performance')
def model_performance_view():
    # Pass any necessary data for initial page load if needed
    return render_template('model_performance.html', max_eval_days=MAX_FORECAST_HORIZON_APP // 24)

@app.route('/calculate_model_performance', methods=['POST'])
def calculate_model_performance_api():
    req_data = request.get_json()
    eval_period_days = req_data.get('eval_period_days', 7)

    if not isinstance(eval_period_days, int) or not (0 < eval_period_days <= (MAX_FORECAST_HORIZON_APP // 24)):
        return jsonify({"error": f"Evaluation period (days) must be a positive integer up to {MAX_FORECAST_HORIZON_APP // 24}."}), 400
    
    eval_period_hours = eval_period_days * 24

    print(f"Calculating model performance for the last {eval_period_days} days ({eval_period_hours} hours).")

    # 1. Fetch data: `eval_period_hours` for actuals + `MAX_LAG_HOURS` for history features
    required_total_data_points = eval_period_hours + MAX_LAG_HOURS + 5 # +5 buffer
    all_db_readings_query = HourlyReading.query.order_by(HourlyReading.timestamp_utc.desc()).limit(required_total_data_points).all()
    
    if len(all_db_readings_query) < required_total_data_points:
        return jsonify({"error": f"Not enough data in DB for {eval_period_days}-day evaluation. "
                                 f"Need {required_total_data_points} hourly records, found {len(all_db_readings_query)}."}), 400

    # Convert to DataFrame (oldest first for processing, UTC index)
    db_data_list_eval = []
    for r_eval in all_db_readings_query:
        db_data_list_eval.append({
            'DateTime': pd.to_datetime(r_eval.timestamp_utc, utc=True),
            DMS_TARGET_COL_DATAFRAME: r_eval.EnergyWh,
            'temperature': r_eval.TemperatureCelsius
        })
    full_eval_period_df_utc = pd.DataFrame(db_data_list_eval).set_index('DateTime').iloc[::-1].sort_index()

    # 2. Split into history (for prediction input) and actuals (for comparison)
    # The 'actuals' part is the most recent `eval_period_hours` of data.
    # The 'history' part is the data immediately preceding these actuals, used for lag features.
    actuals_for_evaluation_df = full_eval_period_df_utc.tail(eval_period_hours).copy()
    history_end_time_for_pred = actuals_for_evaluation_df.index[0] - pd.Timedelta(hours=1)
    history_for_prediction_input = full_eval_period_df_utc.loc[full_eval_period_df_utc.index <= history_end_time_for_pred].tail(MAX_LAG_HOURS + 5).copy() #Sufficient history for lags
    
    if history_for_prediction_input.empty or len(actuals_for_evaluation_df) != eval_period_hours :
         return jsonify({"error": f"Data splitting error for performance evaluation. "
                                  f"Actuals length: {len(actuals_for_evaluation_df)}, Expected: {eval_period_hours}"}), 500

    actuals_series_for_metrics = actuals_for_evaluation_df[DMS_TARGET_COL_DATAFRAME]

    # 3. Get future exogenous variables for the evaluation period (these are the *actual* temperatures during that period)
    future_exog_for_eval_period = actuals_for_evaluation_df[['temperature']].copy() if 'temperature' in DMS_FEATURES_LIST else None

    # 4. Generate DMS predictions for this evaluation period
    try:
        predictions_for_eval_series = algo.predict_dms(
            history_df=history_for_prediction_input, # The history right before the actuals start
            max_horizon_hours=eval_period_hours,
            features_list=DMS_FEATURES_LIST,
            target_col=DMS_TARGET_COL_DATAFRAME,
            models_base_path=DMS_MODELS_BASE_PATH,
            future_exog_series=future_exog_for_eval_period # Actual temperatures for the eval period
        )
    except Exception as e_eval_pred:
        print(f"Error during performance evaluation prediction: {e_eval_pred}")
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Error generating predictions for evaluation: {str(e_eval_pred)}"}), 500

    if predictions_for_eval_series.empty:
        return jsonify({"error": "Performance evaluation: Prediction generation resulted in no data."}), 500

    # 5. Calculate metrics (ensure alignment before calculation)
    comparison_df_eval = pd.DataFrame({'Actual': actuals_series_for_metrics, 'Forecast': predictions_for_eval_series}).dropna()
    if comparison_df_eval.empty:
        return jsonify({"error": "No overlapping data between actuals and forecast for performance metrics calculation."}), 500

    actuals_aligned_eval = comparison_df_eval['Actual']
    forecast_aligned_eval = comparison_df_eval['Forecast']

    metrics_results = {}
    metrics_results['rmse'] = np.sqrt(np.mean((actuals_aligned_eval - forecast_aligned_eval)**2))
    metrics_results['mae'] = np.mean(np.abs(actuals_aligned_eval - forecast_aligned_eval))
    # MAPE calculation robust to zeros in actuals
    valid_actuals_for_mape = actuals_aligned_eval[actuals_aligned_eval != 0]
    if not valid_actuals_for_mape.empty:
        metrics_results['mape'] = np.mean(np.abs((valid_actuals_for_mape - forecast_aligned_eval.loc[valid_actuals_for_mape.index]) / valid_actuals_for_mape)) * 100
    else:
        metrics_results['mape'] = float('inf') # Or 'N/A' if all actuals are zero
    
    # Prepare data for chart (convert to APP_DISPLAY_TIMEZONE for display)
    actuals_display_local = actuals_series_for_metrics.tz_convert(APP_DISPLAY_TIMEZONE)
    predictions_display_local = predictions_for_eval_series.tz_convert(APP_DISPLAY_TIMEZONE)

    chart_labels_eval = [dt_loc.strftime('%Y-%m-%d %H:%M') for dt_loc in actuals_display_local.index]
    chart_actual_data_eval = [round(val, 2) if pd.notna(val) else None for val in actuals_display_local.values]
    # Ensure forecast data aligns with actuals' index for the chart
    chart_forecast_data_eval = [round(predictions_display_local.get(dt_idx_loc), 2) if pd.notna(predictions_display_local.get(dt_idx_loc)) else None for dt_idx_loc in actuals_display_local.index]


    return jsonify({
        "metrics": {k: round(v, 2) if isinstance(v, (float, np.floating)) and pd.notna(v) else (v if pd.notna(v) else 'N/A') for k, v in metrics_results.items()},
        "chart_data": {
            "labels": chart_labels_eval,
            "actuals": chart_actual_data_eval,
            "forecasts": chart_forecast_data_eval
        },
        "eval_period_days": eval_period_days,
        "eval_start_time_display": actuals_display_local.index[0].strftime('%Y-%m-%d %H:%M %Z'),
        "eval_end_time_display": actuals_display_local.index[-1].strftime('%Y-%m-%d %H:%M %Z')
    })

# --- Main Execution ---
if __name__ == "__main__":
    with app.app_context():
        db.create_all() # Ensure tables are created
        print(f"Database '{DATABASE_FILE}' ensured/created.")

    # Start background threads
    data_collector_thread = threading.Thread(target=background_data_collector, daemon=True)
    data_collector_thread.start()

    retraining_scheduler_thread = threading.Thread(target=background_retraining_scheduler, daemon=True)
    retraining_scheduler_thread.start()
    
    # Optional: Trigger an initial training if no models exist or DB is freshly populated
    # This is a simplified check; a more robust check would see if all DMS_MODELS_BASE_PATH/dms_model_horizon_h.pkl exist
    time.sleep(5) # Give app a moment to fully initialize before checking models
    if not os.path.exists(os.path.join(DMS_MODELS_BASE_PATH, f"dms_model_horizon_{MAX_FORECAST_HORIZON_APP}h.pkl")):
        print("Initial models not found. Will attempt initial training if data is available...")
        with app.app_context():
            if HourlyReading.query.first(): # Check if there's any data at all
                print("Data found in DB. Triggering initial training...")
                # Run initial training in a separate thread so it doesn't block app startup
                initial_train_thread = threading.Thread(target=schedule_dms_retraining)
                initial_train_thread.start()
            else:
                print("No data in DB for initial training. Please wait for data collection or add data manually if needed.")

    # For production, use a proper WSGI server like Gunicorn or uWSGI
    # threaded=True is okay for development with background tasks
    # use_reloader=False is important when using threads to avoid issues with the reloader
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)