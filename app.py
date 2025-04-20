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

# try:
#     from lib.data import temp_fetch
# except ImportError as e:
#     print(f"ERROR: Could not import temp_fetch from lib.data: {e}")

import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

def temp_fetch(start_date, end_date, latitude:float, longitude:float, historical:bool):
	# Setup the Open-Meteo API client with cache and retry on error
	cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	openmeteo = openmeteo_requests.Client(session = retry_session)

	if historical == True:
		url = "https://archive-api.open-meteo.com/v1/archive"
	else:
		url = "https://api.open-meteo.com/v1/forecast"
	params = {
		"latitude": latitude,
		"longitude": longitude,
		"hourly": "temperature_2m",
		"timezone": "GMT+8",
		"start_date": start_date,
		"end_date": end_date
	}
	responses = openmeteo.weather_api(url, params=params)

	# Process first location. Add a for-loop for multiple locations or weather models
	response = responses[0]
	# print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
	# print(f"Elevation {response.Elevation()} m asl")
	# print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
	# print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

	# Process hourly data. The order of variables needs to be the same as requested.
	hourly = response.Hourly()
	hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

	hourly_data = {"DateTime": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	)}

	hourly_data["temperature"] = hourly_temperature_2m

	hourly_dataframe = pd.DataFrame(data = hourly_data)
	hourly_dataframe.set_index('DateTime', inplace=True)
	print(f"Start Date: {start_date}")
	print(f"End Date: {end_date}")
	# print(hourly_dataframe)
	return hourly_dataframe

# --- Configuration ---
ANTARES_ACCESS_KEY = '5cd4cda046471a89:75f9e1c6b34bf41a'
ANTARES_PROJECT_NAME = 'UjiCoba_TA'
ANTARES_DEVICE_NAME = 'TA_DKT1'
DATABASE_FILE = 'ntp_compare_hour_app.db'
CHECK_INTERVAL_SECONDS = 60
LATITUDE = 14.5833
LONGITUDE = 121.0
API_TIMEZONE = "Asia/Jakarta" 
NTP_SERVER = 'pool.ntp.org'
MODEL_FILE = 'model.pkl'
MODEL_FEATURES = ['hour', 'day_of_week', 'day_of_month', 'is_weekend',
                  'lag_1', 'lag_24', 'lag_168', 'temperature']
TARGET_VARIABLE = 'DailyEnergy' 
MAX_LAG = 168

# --- Initialize App, DB ---
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DATABASE_FILE}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
app.jinja_env.globals['now'] = datetime.utcnow # For footer

# --- Load Model ---
model = None
try:
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    print(f"Model '{MODEL_FILE}' loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Model file '{MODEL_FILE}' not found. Forecasting will be unavailable.")
except Exception as e:
    print(f"ERROR loading model '{MODEL_FILE}': {e}. Forecasting disabled.")

# --- Database Model ---
class EnergyTempReading(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    DateTime = db.Column(db.String(19), nullable=False, unique=True, index=True) 
    DailyEnergy = db.Column(db.Float) # Assumed Wh
    Temperature = db.Column(db.Float, nullable=True)

# --- NTP Time Fetch ---
def get_ntp_time(server="pool.ntp.org"):
    """Gets current UTC time from an NTP server using sockets."""
    NTP_PORT, NTP_PACKET_FORMAT, NTP_DELTA = 123, "!12I", 2208988800
    client = None # Initialize client to None
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client.settimeout(5)
        data = b'\x1b' + 47 * b'\0'
        client.sendto(data, (server, NTP_PORT))
        data, _ = client.recvfrom(1024)
        if data:
            secs = struct.unpack(NTP_PACKET_FORMAT, data)[10]
            timestamp = secs - NTP_DELTA
            return datetime.fromtimestamp(timestamp, timezone.utc)
        return None
    except socket.timeout:
        print("NTP Error: Request timed out")
        return None
    except Exception as e:
        print(f"NTP Error: {e}")
        return None
    finally:
        if client: client.close()

# --- Background Hourly Data Fetch ---
def fetch_and_store_hourly_data(target_hour_dt_utc: datetime):
    """Fetches Antares data and temperature forecast, stores hourly record if new."""
    with app.app_context():
        formatted_ts = target_hour_dt_utc.strftime('%Y-%m-%d %H:%M:00')
        if EnergyTempReading.query.filter_by(DateTime=formatted_ts).first():
            return

        try:
            antares.setAccessKey(ANTARES_ACCESS_KEY)
            latest_data = antares.get(ANTARES_PROJECT_NAME, ANTARES_DEVICE_NAME)
            if not latest_data or 'content' not in latest_data:
                print(f"Hourly Store: Invalid Antares data near {formatted_ts} UTC.")
                return

            content = latest_data['content']
            energy_wh = content.get('Energy')
            if energy_wh is None:
                print(f"Hourly Store: Energy value missing near {formatted_ts} UTC.")
                return

            temp_val = None
            if temp_fetch:
                try:
                    date_str = target_hour_dt_utc.strftime('%Y-%m-%d')
                    temp_df = temp_fetch(date_str, date_str, LATITUDE, LONGITUDE, historical=False)
                    if temp_df is not None and not temp_df.empty:
                        temp_df.index = temp_df.index.tz_convert(timezone.utc)
                        temp_val = temp_df['temperature'].get(pd.Timestamp(target_hour_dt_utc))
                except Exception as e:
                    print(f"Hourly Store: Error getting temp forecast for {formatted_ts} UTC: {e}")

            new_reading = EnergyTempReading(DateTime=formatted_ts, DailyEnergy=energy_wh, Temperature=temp_val)
            db.session.add(new_reading)
            db.session.commit()
            print(f"Stored Hourly: {formatted_ts} UTC - Energy(Wh): {energy_wh:.2f}, Temp: {temp_val:.2f if temp_val is not None else 'N/A'}")

        except Exception as e:
            db.session.rollback()
            print(f"Hourly Store Error for {formatted_ts} UTC: {e}")

def background_ntp_checker():
    """Periodically checks NTP time and triggers hourly fetch on hour change."""
    print("Background NTP Checker started...")
    previous_ntp_time_utc = None
    while True:
        current_ntp_time_utc = get_ntp_time(NTP_SERVER)
        print("Current UTC Time is:", current_ntp_time_utc)
        if current_ntp_time_utc:
            target_hour_to_fetch = None
            # Determine if the hour is different from the last check, or if it's the first run
            if previous_ntp_time_utc is None or current_ntp_time_utc.hour != previous_ntp_time_utc.hour:
                # Hour changed or first run: define the target hour (start of current hour)
                target_hour_to_fetch = current_ntp_time_utc.replace(minute=0, second=0, microsecond=0)
                print(f"NTP Check: New hour detected ({target_hour_to_fetch.strftime('%H:%M')} UTC). Triggering store.")
                fetch_and_store_hourly_data(target_hour_dt_utc=target_hour_to_fetch)

            previous_ntp_time_utc = current_ntp_time_utc # Update time for next comparison
        else:
            print("NTP Check: Failed to get NTP time, skipping cycle.")

        time.sleep(CHECK_INTERVAL_SECONDS) # Wait before next check

# --- Flask Routes ---
@app.route('/')
def home():
    # Redirect to forecast page as the primary view
    return redirect(url_for('forecast_view'))

@app.route('/database')
def database_view():
    """Displays graph controls and full data table."""
    all_readings = EnergyTempReading.query.order_by(EnergyTempReading.DateTime.desc()).all()
    return render_template('database.html', readings=all_readings)

@app.route('/get_range_data')
def get_range_data():
    """API endpoint for the database graph."""
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    if not (start_date and end_date):
        return jsonify({"error": "Start and end dates required."}), 400

    start_dt_str = f"{start_date} 00:00:00"
    end_dt_str = f"{end_date} 23:59:59"
    readings = EnergyTempReading.query.filter(
        EnergyTempReading.DateTime >= start_dt_str,
        EnergyTempReading.DateTime <= end_dt_str
    ).order_by(EnergyTempReading.DateTime).all()

    labels = [r.DateTime for r in readings]
    data = [r.DailyEnergy if r.DailyEnergy is not None else 0 for r in readings]
    return jsonify({"labels": labels, "data": data})

@app.route('/forecast')
def forecast_view():
    """Displays the forecast controls and graph."""
    latest = EnergyTempReading.query.order_by(EnergyTempReading.DateTime.desc()).first()
    return render_template('forecast.html', latest_reading=latest)

@app.route('/run_forecast', methods=['POST'])
def run_forecast_api():
    """API endpoint to generate forecast using recursive method."""
    if model is None:
        return jsonify({"error": "Forecasting model is not loaded."}), 500

    # --- Input Validation ---
    req_data = request.get_json()
    timeframe = req_data.get('timeframe')
    hours_map = {'1day': 24, '3days': 72, '1week': 168}
    if timeframe not in hours_map:
        return jsonify({"error": "Invalid timeframe selected."}), 400
    hours_to_forecast = hours_map[timeframe]

    # --- Fetch History ---
    history_query = EnergyTempReading.query.order_by(EnergyTempReading.DateTime.desc()).limit(MAX_LAG).all()
    if len(history_query) < MAX_LAG:
        return jsonify({"error": f"Insufficient history: need {MAX_LAG} hours, found {len(history_query)}."}), 400

    # Convert UTC strings from DB to localized DateTimeIndex for processing
    history_df = pd.DataFrame(
        [{'DateTime': r.DateTime, TARGET_VARIABLE: r.DailyEnergy, 'Temperature': r.Temperature} for r in history_query]
    ).set_index('DateTime').iloc[::-1] # Oldest first
    history_df.index = pd.to_datetime(history_df.index + '+00:00').tz_convert(API_TIMEZONE) # UTC -> Local Processing TZ

    # --- Fetch Future Temperatures (Forecast) ---
    last_known_local = history_df.index[-1]
    forecast_start_local = last_known_local + timedelta(hours=1)
    forecast_end_local = last_known_local + timedelta(hours=hours_to_forecast)
    future_temps_series = None # Initialize

    if temp_fetch:
        try:
            start_api_date = forecast_start_local.strftime('%Y-%m-%d')
            end_api_date = forecast_end_local.strftime('%Y-%m-%d')
            temp_df = temp_fetch(start_api_date, end_api_date, LATITUDE, LONGITUDE, historical=False) # Always forecast
            if temp_df is not None and not temp_df.empty:
                temp_df.index = temp_df.index.tz_convert(API_TIMEZONE) # Convert UTC result to local processing TZ
                future_idx_local = pd.date_range(start=forecast_start_local, periods=hours_to_forecast, freq='h', tz=API_TIMEZONE)
                future_temps_series = temp_df['temperature'].reindex(future_idx_local, method='ffill') # Align to needed hours
        except Exception as e:
            print(f"Warning: Failed future temperature fetch: {e}")

    # Fallback if fetch failed or temp_fetch unavailable
    if future_temps_series is None:
        last_temp = history_df['Temperature'].iloc[-1] if pd.notna(history_df['Temperature'].iloc[-1]) else 25.0
        future_temps_series = pd.Series(last_temp, index=pd.date_range(start=forecast_start_local, periods=hours_to_forecast, freq='h', tz=API_TIMEZONE))

    # --- Recursive Prediction Loop ---
    predictions, timestamps = [], []
    current_data = history_df[[TARGET_VARIABLE]].copy() # Holds target values (history + predictions)

    for h in range(hours_to_forecast):
        pred_time_local = forecast_start_local + timedelta(hours=h)
        timestamps.append(pred_time_local.strftime('%Y-%m-%d %H:%M:00')) # Labels for chart

        # Build features for this step
        features = {
            'hour': pred_time_local.hour,
            'day_of_week': pred_time_local.dayofweek,
            'day_of_month': pred_time_local.day,
            'is_weekend': 1 if pred_time_local.dayofweek >= 5 else 0,
            'temperature': future_temps_series.get(pred_time_local, future_temps_series.iloc[-1]) # Get temp, fallback to last if somehow missing
        }
        for lag in [1, 24, 168]:
            lag_time = pred_time_local - timedelta(hours=lag)
            features[f'lag_{lag}'] = current_data[TARGET_VARIABLE].get(lag_time, np.nan) # Look up lag in current_data

        # Prepare for model and predict
        feature_vector = pd.DataFrame([features], columns=MODEL_FEATURES).fillna(0) # Ensure order and handle missing lags
        prediction = max(0.0, float(model.predict(feature_vector)[0])) # Predict and ensure non-negative
        predictions.append(prediction)

        # Add prediction back to current_data for subsequent lag calculations
        current_data.loc[pred_time_local] = prediction

    return jsonify({"labels": timestamps, "data": predictions})

# --- Main Execution ---
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print(f"Database '{DATABASE_FILE}' ensured.")

    # Start background checker thread
    fetch_thread = threading.Thread(target=background_ntp_checker, daemon=True)
    fetch_thread.start()

    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

