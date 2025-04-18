import os
import time
import threading
import pickle
from datetime import datetime, timedelta, timezone
import socket
import struct
import pandas as pd
import numpy as np
import xgboost as xgb
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from antares_http import antares

# --- Import temp_fetch from lib.data ---
try:
    from lib.data import temp_fetch
except ImportError as e:
    print(f"ERROR: Could not import temp_fetch from lib.data: {e}")
    temp_fetch = None

# --- Configuration ---
ANTARES_ACCESS_KEY = '5cd4cda046471a89:75f9e1c6b34bf41a'
ANTARES_PROJECT_NAME = 'UjiCoba_TA'
ANTARES_DEVICE_NAME = 'TA_DKT1'
DATABASE_FILE = 'ntp_compare_hour_app.db' # New DB name
CHECK_INTERVAL_SECONDS = 60 # How often to check NTP time
LATITUDE = 14.5833
LONGITUDE = 121.0
API_TIMEZONE = "Asia/Jakarta"
NTP_SERVER = 'pool.ntp.org'
MODEL_FILE = 'model.pkl'
MODEL_FEATURES = ['hour', 'day_of_week', 'day_of_month', 'is_weekend',
                  'lag_1', 'lag_24', 'lag_168', 'temperature']
TARGET_VARIABLE = 'DailyEnergy' # Assumed Wh
MAX_LAG = 168

# --- Initialize App, DB, and Load Model ---
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DATABASE_FILE}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
app.jinja_env.globals['now'] = datetime.utcnow

model = None
try:
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    print(f"Model '{MODEL_FILE}' loaded successfully.")
except Exception as e:
    print(f"ERROR loading model '{MODEL_FILE}': {e}. Forecasting disabled.")

# --- Database Model ---
class EnergyTempReading(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    DateTime = db.Column(db.String(19), nullable=False, unique=True, index=True) # YYYY-MM-DD HH:00:00 UTC
    DailyEnergy = db.Column(db.Float) # Assumed Wh
    Temperature = db.Column(db.Float, nullable=True)

# --- NTP Time Fetch Function (using socket - unchanged) ---
def get_ntp_time(server="pool.ntp.org"):
    """Gets current UTC time from an NTP server using sockets."""
    NTP_PORT = 123
    NTP_PACKET_FORMAT = "!12I"
    NTP_DELTA = 2208988800
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.settimeout(5)
        data = b'\x1b' + 47 * b'\0'
        client.sendto(data, (server, NTP_PORT))
        data, _ = client.recvfrom(1024)
        if data:
            secs = struct.unpack('!I', data[40:44])[0]
            timestamp = secs - NTP_DELTA
            return datetime.fromtimestamp(timestamp, timezone.utc)
        else:
            print("NTP Error: No data received")
            return None
    except socket.timeout:
        print("NTP Error: Request timed out")
        return None
    except Exception as e:
        print(f"NTP Error: {e}")
        return None
    finally:
        if 'client' in locals():
            client.close()


# --- Background Data Fetching (Function itself unchanged) ---
def fetch_and_store_hourly_data(target_hour_dt_utc: datetime):
    """Fetches Antares data and stores the record for the specified UTC hour."""
    with app.app_context():
        temp_val = None
        formatted_hourly_timestamp = target_hour_dt_utc.strftime('%Y-%m-%d %H:%M:00')
        try:
            existing = EnergyTempReading.query.filter_by(DateTime=formatted_hourly_timestamp).first()
            if existing: return

            antares.setAccessKey(ANTARES_ACCESS_KEY)
            latest_data = antares.get(ANTARES_PROJECT_NAME, ANTARES_DEVICE_NAME)
            if not latest_data or 'content' not in latest_data: return
            content = latest_data['content']
            energy_wh = content.get('Energy')
            if energy_wh is None: return

            if temp_fetch:
                try:
                    date_str = target_hour_dt_utc.strftime('%Y-%m-%d')
                    # --- ALWAYS historical=False ---
                    temp_df = temp_fetch(date_str, date_str, LATITUDE, LONGITUDE, historical=False)
                    # ---
                    if temp_df is not None and not temp_df.empty:
                        temp_df.index = temp_df.index.tz_convert(timezone.utc) # Ensure UTC index for lookup
                        target_ts_pd = pd.Timestamp(target_hour_dt_utc) # Already UTC
                        temp_val = temp_df['temperature'].get(target_ts_pd)
                        # if temp_val is None: print(f"Hourly Check: Target hour {formatted_hourly_timestamp} UTC not in temp forecast.")
                except Exception as e:
                    print(f"Hourly Check: Error getting temp forecast for {formatted_hourly_timestamp} UTC: {e}")

            new_reading = EnergyTempReading(DateTime=formatted_hourly_timestamp, DailyEnergy=energy_wh, Temperature=temp_val)
            db.session.add(new_reading)
            db.session.commit()
            print(f"Stored Hourly (NTP Hour Change): {formatted_hourly_timestamp} UTC - Energy(Wh): {energy_wh}, Temp: {temp_val}")

        except Exception as e:
            print(f"Hourly Check Fetch/Store Error: {e}")
            db.session.rollback()

# --- MODIFIED Background Checker ---
def background_ntp_checker_compare_hour():
    """Periodically checks NTP time and triggers hourly fetch IF the hour component changes."""
    print("Background NTP Checker (Compare Hour) started...")
    previous_ntp_time_utc = None # Store the PREVIOUS exact NTP time

    while True:
        current_ntp_time_utc = get_ntp_time(NTP_SERVER)
        print("Current UTC Time is:", current_ntp_time_utc)

        if current_ntp_time_utc:
            trigger_fetch = False
            target_hour_to_fetch = None

            if previous_ntp_time_utc is None:
                # First run: fetch data for the current hour
                trigger_fetch = True
                # Round down CURRENT time to get the hour to fetch for
                target_hour_to_fetch = current_ntp_time_utc.replace(minute=0, second=0, microsecond=0)
                print(f"NTP Check: First run. Targeting hour {target_hour_to_fetch.strftime('%H')}:00 UTC.")
            elif current_ntp_time_utc.hour != previous_ntp_time_utc.hour:
                # Hour component has changed since the last check
                trigger_fetch = True
                 # Round down CURRENT time to get the hour that just STARTED
                target_hour_to_fetch = current_ntp_time_utc.replace(minute=0, second=0, microsecond=0)
                print(f"NTP Check: Hour changed from {previous_ntp_time_utc.hour} to {current_ntp_time_utc.hour}. Targeting {target_hour_to_fetch.strftime('%H')}:00 UTC.")
            # else:
                # Hour hasn't changed, do nothing
                # print(f"NTP Check: Hour ({current_ntp_time_utc.hour}) unchanged since last check.") # Debug

            # If an hour change was detected, trigger the fetch
            if trigger_fetch and target_hour_to_fetch:
                fetch_and_store_hourly_data(target_hour_dt_utc=target_hour_to_fetch)

            # IMPORTANT: Update the previous time for the next comparison
            previous_ntp_time_utc = current_ntp_time_utc

        else:
            print("NTP Check: Failed to get NTP time, skipping check cycle.")

        # Wait before checking again
        time.sleep(CHECK_INTERVAL_SECONDS)


# --- Flask Routes (Unchanged) ---
@app.route('/')
def home(): return redirect(url_for('forecast_view'))

@app.route('/database')
def database_view():
    readings = EnergyTempReading.query.order_by(EnergyTempReading.DateTime.desc()).all()
    return render_template('database.html', readings=readings)

@app.route('/forecast')
def forecast_view():
    latest_reading = EnergyTempReading.query.order_by(EnergyTempReading.DateTime.desc()).first()
    return render_template('forecast.html', latest_reading=latest_reading)


# --- Forecasting API (Unchanged logic, uses historical=False) ---
@app.route('/run_forecast', methods=['POST'])
def run_forecast_api():
    # (Keep this function exactly as in the previous "direct NTP" version)
    # It correctly reads UTC strings, converts to API_TIMEZONE, fetches temp with historical=False, and forecasts.
    if model is None: return jsonify({"error": "Forecasting model not loaded."}), 500
    try:
        data = request.get_json()
        timeframe = data.get('timeframe')
        hours_map = {'1day': 24, '3days': 72, '1week': 168}
        if timeframe not in hours_map: return jsonify({"error": "Invalid timeframe"}), 400
        hours_to_forecast = hours_map[timeframe]

        print(f"Fetching last {MAX_LAG} hourly readings (UTC)...")
        history_query = EnergyTempReading.query.order_by(EnergyTempReading.DateTime.desc()).limit(MAX_LAG).all()
        if len(history_query) < MAX_LAG: return jsonify({"error": f"Need {MAX_LAG} history points, found {len(history_query)}."}), 400
        history_df = pd.DataFrame([{'DateTime': r.DateTime, TARGET_VARIABLE: r.DailyEnergy, 'Temperature': r.Temperature} for r in history_query]).set_index('DateTime').iloc[::-1]
        history_df.index = pd.to_datetime(history_df.index + '+00:00') # Parse UTC
        history_df = history_df.tz_convert(API_TIMEZONE) # Convert to processing TZ

        print("Fetching future temperatures (historical=False)...")
        last_known_time_local = history_df.index[-1]
        forecast_start_dt_local = last_known_time_local + timedelta(hours=1)
        forecast_end_dt_local = last_known_time_local + timedelta(hours=hours_to_forecast)
        future_temps = None
        if temp_fetch:
            try:
                start_date_api = forecast_start_dt_local.strftime('%Y-%m-%d')
                end_date_api = forecast_end_dt_local.strftime('%Y-%m-%d')
                temp_df = temp_fetch(start_date_api, end_date_api, LATITUDE, LONGITUDE, historical=False) # historical=False
                if temp_df is not None and not temp_df.empty:
                    temp_df.index = temp_df.index.tz_convert(API_TIMEZONE)
                    future_index = pd.date_range(start=forecast_start_dt_local, periods=hours_to_forecast, freq='h', tz=API_TIMEZONE)
                    future_temps = temp_df['temperature'].reindex(future_index, method='ffill')
            except Exception as e: print(f"Warning: Failed future temp fetch: {e}")
        if future_temps is None: # Fallback
            last_temp = history_df['Temperature'].iloc[-1] if pd.notna(history_df['Temperature'].iloc[-1]) else 25.0
            future_temps = pd.Series([last_temp] * hours_to_forecast, index=pd.date_range(start=forecast_start_dt_local, periods=hours_to_forecast, freq='h', tz=API_TIMEZONE))

        print("Starting recursive forecast...")
        predictions, timestamps = [], []
        current_data = history_df[[TARGET_VARIABLE]].copy()
        for h in range(hours_to_forecast):
            current_pred_time_local = forecast_start_dt_local + timedelta(hours=h)
            timestamps.append(current_pred_time_local.strftime('%Y-%m-%d %H:%M:00'))
            features = {'hour': current_pred_time_local.hour, 'day_of_week': current_pred_time_local.dayofweek,
                        'day_of_month': current_pred_time_local.day, 'is_weekend': 1 if current_pred_time_local.dayofweek >= 5 else 0,
                        'temperature': future_temps.get(current_pred_time_local, future_temps.iloc[-1])}
            for lag in [1, 24, 168]:
                lag_time_local = current_pred_time_local - timedelta(hours=lag)
                features[f'lag_{lag}'] = current_data[TARGET_VARIABLE].get(lag_time_local, np.nan)
            feature_vector_df = pd.DataFrame([features], columns=MODEL_FEATURES).fillna(0)
            prediction = max(0.0, float(model.predict(feature_vector_df)[0]))
            predictions.append(prediction)
            new_row = pd.DataFrame({TARGET_VARIABLE: [prediction]}, index=[current_pred_time_local])
            current_data = pd.concat([current_data, new_row])
        print("Forecast complete.")
        return jsonify({"labels": timestamps, "data": predictions})

    except Exception as e:
        print(f"Error in /run_forecast: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred during forecasting."}), 500


# --- Main Execution ---
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print(f"Database '{DATABASE_FILE}' ensured.")

    # Start background thread using the new logic
    fetch_thread = threading.Thread(target=background_ntp_checker_compare_hour, daemon=True)
    fetch_thread.start()

    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
