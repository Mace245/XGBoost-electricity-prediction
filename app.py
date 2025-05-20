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
from antares_http import antares # Assuming this is correctly installed and configured

# Assuming data.py and algo.py are in the same directory or accessible via PYTHONPATH
from lib import data # Your data.py module
from lib import algo # Your algo.py module

# --- Configuration ---
ANTARES_ACCESS_KEY = os.getenv('ANTARES_ACCESS_KEY', 'YOUR_ANTARES_ACCESS_KEY') # Use environment variables
ANTARES_PROJECT_NAME = os.getenv('ANTARES_PROJECT_NAME', 'YOUR_ANTARES_PROJECT_NAME')
ANTARES_DEVICE_NAME = os.getenv('ANTARES_DEVICE_NAME', 'YOUR_ANTARES_DEVICE_NAME')

DATABASE_FILE = 'energy_app_dms_simplified.db'
# For background data fetching (how often to check if a new hour has started)
# For hourly data, 60 seconds is too frequent. Check every 5-15 minutes.
BACKGROUND_FETCH_INTERVAL_SECONDS = 15 * 60

LATITUDE_CONFIG = 14.5833 # Consider making these configurable if they change
LONGITUDE_CONFIG = 121.0
# Timezone for displaying data to the user if different from UTC
APP_DISPLAY_TIMEZONE = "Asia/Kuala_Lumpur"
# All internal processing and DB storage will be UTC.

DMS_MODELS_BASE_PATH = 'models_dms/'
# Max hours the app will offer forecasts for. All models from h=1 to this must be trained.
MAX_FORECAST_HORIZON_APP = 24 * 7 # 7 days

# Features and target for DMS models (must match what models were trained with)
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


# --- OpenMeteo Temperature Fetch Utility ---
# This could also live in data.py if preferred
import openmeteo_requests
import requests_cache
from retry_requests import retry

def fetch_temperature_forecast_openmeteo(start_date_str_utc, end_date_str_utc, latitude, longitude):
    """Fetches hourly temperature forecast from OpenMeteo for a UTC date range."""
    cache_session = requests_cache.CachedSession('.cache_openmeteo_app', expire_after=1800) # Cache for 30 mins
    retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    url = "https://api.open-meteo.com/v1/forecast" # Using the forecast API
    params = {
        "latitude": latitude, "longitude": longitude,
        "hourly": "temperature_2m", "timezone": "UTC", # Explicitly request UTC
        "start_date": start_date_str_utc, "end_date": end_date_str_utc
    }
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0] # Process first location
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

        hourly_data_index = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
        # Create a Series for easier reindexing later
        temp_series = pd.Series(data=hourly_temperature_2m, index=hourly_data_index, name='temperature')
        return temp_series
    except Exception as e:
        print(f"Error in fetch_temperature_forecast_openmeteo: {e}")
        return None

# --- Database Model ---
class HourlyReading(db.Model): # Renamed for clarity
    id = db.Column(db.Integer, primary_key=True)
    # Store DateTime as ISO 8601 UTC string (e.g., "2023-10-27T10:00:00Z")
    # This is a standard and unambiguous way to store datetimes.
    timestamp_utc = db.Column(db.String(20), nullable=False, unique=True, index=True)
    # Column names match constants for clarity
    EnergyWh = db.Column(db.Float)
    TemperatureCelsius = db.Column(db.Float, nullable=True)

    def __repr__(self):
        return f"<HourlyReading {self.timestamp_utc} - Energy: {self.EnergyWh}>"

# --- Background Data Collection ---
last_fetched_hour_utc = None # Keep track of the last hour we successfully fetched data for

def background_data_collector():
    """Periodically fetches the latest hourly data from Antares and temperature forecast."""
    global last_fetched_hour_utc
    print("Background Data Collector started...")

    # On first run, try to determine the last stored hour or start "fresh"
    with app.app_context():
        latest_db_entry = HourlyReading.query.order_by(HourlyReading.timestamp_utc.desc()).first()
        if latest_db_entry:
            try:
                last_fetched_hour_utc = pd.to_datetime(latest_db_entry.timestamp_utc).replace(minute=0, second=0, microsecond=0)
                print(f"Data Collector: Resuming. Last known stored hour: {last_fetched_hour_utc.isoformat()}")
            except Exception as e:
                print(f"Data Collector: Error parsing last DB timestamp: {e}. Will start fresh.")
                last_fetched_hour_utc = None
        else:
            print("Data Collector: No existing data in DB. Will start fresh.")


    while True:
        current_time_utc = datetime.now(dt_timezone.utc)
        # Target the beginning of the current hour or the next hour if we just passed it
        target_hour_utc = current_time_utc.replace(minute=0, second=0, microsecond=0)

        if last_fetched_hour_utc is None: # First run after startup or if DB was empty
            print(f"Data Collector: First run or no previous data, targeting {target_hour_utc.isoformat()} for fetch.")
        elif target_hour_utc <= last_fetched_hour_utc:
            # print(f"Data Collector: Current hour {target_hour_utc.isoformat()} already processed or not yet new. Last was {last_fetched_hour_utc.isoformat()}.")
            time.sleep(BACKGROUND_FETCH_INTERVAL_SECONDS)
            continue # Not a new hour yet based on our last fetch

        # If there's a gap (e.g., app was down), fetch hour by hour up to current
        hour_to_process = last_fetched_hour_utc + timedelta(hours=1) if last_fetched_hour_utc else target_hour_utc
        
        processed_in_cycle = False
        while hour_to_process <= target_hour_utc:
            print(f"Data Collector: Processing for hour {hour_to_process.isoformat()}...")
            with app.app_context(): # Ensure DB operations are within app context
                # Check if data for this hour already exists
                if HourlyReading.query.filter_by(timestamp_utc=hour_to_process.isoformat() + "Z").first():
                    print(f"  Data for {hour_to_process.isoformat()} already in DB. Skipping Antares/Temp fetch.")
                    last_fetched_hour_utc = hour_to_process # Mark as processed
                    hour_to_process += timedelta(hours=1)
                    processed_in_cycle = True
                    continue

                energy_wh_value = None
                # 1. Fetch from Antares (assuming it gives the latest reading)
                try:
                    antares.setAccessKey(ANTARES_ACCESS_KEY)
                    latest_antares_data = antares.get(ANTARES_PROJECT_NAME, ANTARES_DEVICE_NAME)
                    if latest_antares_data and 'content' in latest_antares_data:
                        # IMPORTANT: You need to align Antares data to the correct hour.
                        # Antares 'latest' might not be for 'hour_to_process'.
                        # This example assumes 'Energy' is the cumulative or hourly value you need.
                        # A more robust solution would be to fetch Antares data for a specific time range if API supports it.
                        # For now, we take the latest and assume it's relevant if fetched near the hour.
                        # This part is highly dependent on how Antares structures its data and timestamps.
                        energy_wh_value = latest_antares_data['content'].get('Energy') # Adjust key if needed
                        if energy_wh_value is None:
                            print(f"  Antares data for ~{hour_to_process.isoformat()} missing 'Energy' key or value.")
                    else:
                        print(f"  No valid content from Antares for ~{hour_to_process.isoformat()}.")
                except Exception as e:
                    print(f"  Error fetching from Antares for ~{hour_to_process.isoformat()}: {e}")

                if energy_wh_value is None:
                    print(f"  Skipping DB store for {hour_to_process.isoformat()} due to missing Antares energy data.")
                    # Decide if you want to advance last_fetched_hour_utc even if Antares fails
                    # For now, we don't, to retry this hour later.
                    # To prevent continuous retries on persistent Antares failure, add a retry limit or error flag.
                    break # Break inner loop, wait for next BACKGROUND_FETCH_INTERVAL_SECONDS

                # 2. Fetch Temperature Forecast for this specific hour_to_process
                temperature_value = None
                try:
                    date_str_utc = hour_to_process.strftime('%Y-%m-%d')
                    temp_series_utc = fetch_temperature_forecast_openmeteo(date_str_utc, date_str_utc, LATITUDE_CONFIG, LONGITUDE_CONFIG)
                    if temp_series_utc is not None:
                        temperature_value = temp_series_utc.get(hour_to_process) # Get temp for the exact UTC hour
                        if pd.isna(temperature_value):
                            print(f"  Temperature for {hour_to_process.isoformat()} was NaN after fetch, trying ffill.")
                            # Attempt to fill from earlier in the day if exact hour is missing
                            filled_temp = temp_series_utc.reindex(pd.date_range(start=hour_to_process.replace(hour=0), end=hour_to_process, freq='h', tz='UTC'), method='ffill')
                            if not filled_temp.empty:
                                temperature_value = filled_temp.iloc[-1]

                except Exception as e:
                    print(f"  Error fetching temperature for {hour_to_process.isoformat()}: {e}")

                # 3. Store in Database
                new_entry = HourlyReading(
                    timestamp_utc=hour_to_process.isoformat() + "Z", # ISO 8601 format
                    EnergyWh=float(energy_wh_value),
                    TemperatureCelsius=float(temperature_value) if pd.notna(temperature_value) else None
                )
                db.session.add(new_entry)
                try:
                    db.session.commit()
                    print(f"  Stored: {new_entry.timestamp_utc} - Energy: {new_entry.EnergyWh:.2f}, Temp: {new_entry.TemperatureCelsius:.2f if new_entry.TemperatureCelsius else 'N/A'}")
                    last_fetched_hour_utc = hour_to_process # Successfully stored
                    processed_in_cycle = True
                except Exception as e_commit:
                    db.session.rollback()
                    print(f"  DB Commit Error for {hour_to_process.isoformat()}: {e_commit}. Will retry this hour later.")
                    break # Break inner loop to retry this hour after interval

            hour_to_process += timedelta(hours=1) # Move to next hour if current was processed
        
        if not processed_in_cycle and last_fetched_hour_utc is not None : # No new hour processed, but we are past the last fetched hour
             if target_hour_utc > last_fetched_hour_utc: # ensure we are not going backward
                  last_fetched_hour_utc = target_hour_utc - timedelta(hours=1) # Update to avoid re-fetching same current hour immediately

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

    # 1. Fetch necessary historical data from DB for feature creation
    # We need at least MAX_LAG_HOURS records ending at the most recent time.
    history_query_results = HourlyReading.query.order_by(HourlyReading.timestamp_utc.desc()).limit(MAX_LAG_HOURS + 5).all() # +5 buffer

    if len(history_query_results) < MAX_LAG_HOURS:
        return jsonify({"error": f"Insufficient historical data in DB ({len(history_query_results)} records). "
                                 f"Need at least {MAX_LAG_HOURS} for lags."}), 400

    # Convert query results to DataFrame, oldest first, with UTC DatetimeIndex
    history_list_for_df = []
    for r_hist in history_query_results:
        history_list_for_df.append({
            'DateTime': pd.to_datetime(r_hist.timestamp_utc, utc=True),
            DMS_TARGET_COL_DATAFRAME: r_hist.EnergyWh, # Use 'Wh' for DataFrame
            'temperature': r_hist.TemperatureCelsius   # Use 'temperature' for DataFrame
        })
    history_df_utc = pd.DataFrame(history_list_for_df).set_index('DateTime').iloc[::-1].sort_index()

    if history_df_utc.empty:
         return jsonify({"error": "Failed to construct historical DataFrame for prediction."}), 500

    # 2. Fetch future temperature forecasts
    last_known_history_time_utc = history_df_utc.index[-1]
    forecast_period_start_utc = last_known_history_time_utc + timedelta(hours=1)
    forecast_period_end_utc = last_known_history_time_utc + timedelta(hours=hours_to_forecast)
    future_temperatures_df = None # This will be a DataFrame with 'temperature' column

    try:
        start_date_api_utc = forecast_period_start_utc.strftime('%Y-%m-%d')
        end_date_api_utc = forecast_period_end_utc.strftime('%Y-%m-%d')
        temp_series_fcst_utc = fetch_temperature_forecast_openmeteo(
            start_date_api_utc, end_date_api_utc, LATITUDE_CONFIG, LONGITUDE_CONFIG
        )

        if temp_series_fcst_utc is not None and not temp_series_fcst_utc.empty:
            # Ensure we have a continuous series for the exact forecast period hours
            required_future_idx_utc = pd.date_range(start=forecast_period_start_utc,
                                                    end=forecast_period_end_utc,
                                                    freq='h', tz='UTC')
            # Reindex and fill, then convert to DataFrame
            aligned_temps = temp_series_fcst_utc.reindex(required_future_idx_utc, method='ffill').fillna(method='bfill')
            future_temperatures_df = pd.DataFrame({'temperature': aligned_temps})

            # Fallback for any remaining NaNs (e.g., if API didn't cover full range)
            if future_temperatures_df['temperature'].isna().any():
                last_hist_temp = history_df_utc['temperature'].dropna().iloc[-1] if not history_df_utc['temperature'].dropna().empty else 25.0 # Default temp
                future_temperatures_df['temperature'] = future_temperatures_df['temperature'].fillna(last_hist_temp)
        else:
            print("Warning: Future temperature fetch returned no data. Using fallback.")
    except Exception as e:
        print(f"Warning: Exception during future temperature fetch: {e}. Using fallback.")

    if future_temperatures_df is None or future_temperatures_df.empty: # Fallback if fetch failed
        print("  Executing temperature fallback for forecast period.")
        last_hist_temp = history_df_utc['temperature'].dropna().iloc[-1] if not history_df_utc['temperature'].dropna().empty else 25.0
        required_future_idx_utc = pd.date_range(start=forecast_period_start_utc,
                                                end=forecast_period_end_utc,
                                                freq='h', tz='UTC')
        future_temperatures_df = pd.DataFrame({'temperature': last_hist_temp}, index=required_future_idx_utc)

    # 3. Call the DMS prediction function from algo.py
    try:
        dms_predictions_series_utc = algo.predict_dms(
            history_df=history_df_utc, # Pass UTC history
            max_horizon_hours=hours_to_forecast,
            features_list=DMS_FEATURES_LIST,
            target_col=DMS_TARGET_COL_DATAFRAME, # 'Wh' for DataFrame
            models_base_path=DMS_MODELS_BASE_PATH,
            future_exog_series=future_temperatures_df # Pass UTC future temps DataFrame
        )
    except Exception as e_pred_dms:
        print(f"Error during DMS prediction call: {e_pred_dms}")
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Forecast generation error: {str(e_pred_dms)}"}), 500

    if dms_predictions_series_utc.empty:
        return jsonify({"error": "Forecast generation resulted in no prediction data."}), 500

    # 4. Prepare response: Convert UTC prediction timestamps to APP_DISPLAY_TIMEZONE for chart labels
    try:
        dms_predictions_display_local = dms_predictions_series_utc.tz_convert(APP_DISPLAY_TIMEZONE)
    except TypeError as e_tz: # Handles case where series might be tz-naive unexpectedly
        print(f"Timezone conversion error for display: {e_tz}. Assuming UTC for display.")
        dms_predictions_display_local = dms_predictions_series_utc # Display as is (should be UTC)


    timestamps_for_chart = [dt.strftime('%Y-%m-%d %H:%M') for dt in dms_predictions_display_local.index]
    data_for_chart = [round(p_val, 2) if pd.notna(p_val) else None for p_val in dms_predictions_display_local.values]

    return jsonify({"labels": timestamps_for_chart, "data": data_for_chart})


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