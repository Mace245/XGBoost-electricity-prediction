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
# Make sure request is imported from flask
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from antares_http import antares

# --- Import temp_fetch from lib.data ---
try:
    from lib.data import temp_fetch
except ImportError as e:
    print(f"ERROR: Could not import temp_fetch from lib.data: {e}")
    print("Ensure 'lib/data.py' exists and 'lib/__init__.py' is present.")
    temp_fetch = None

# --- Configuration (Keep as before) ---
ANTARES_ACCESS_KEY = '5cd4cda046471a89:75f9e1c6b34bf41a'
ANTARES_PROJECT_NAME = 'UjiCoba_TA'
ANTARES_DEVICE_NAME = 'TA_DKT1'
DATABASE_FILE = 'ntp_compare_hour_app.db' # Slightly new DB name for clean run
CHECK_INTERVAL_SECONDS = 60
LATITUDE = 14.5833
LONGITUDE = 121.0
API_TIMEZONE = "Asia/Jakarta"
NTP_SERVER = 'pool.ntp.org'
MODEL_FILE = 'model.pkl'
MODEL_FEATURES = ['hour', 'day_of_week', 'day_of_month', 'is_weekend',
                  'lag_1', 'lag_24', 'lag_168', 'temperature']
TARGET_VARIABLE = 'DailyEnergy' # Assumed Wh
MAX_LAG = 168

# --- Initialize App, DB, and Load Model (Keep as before) ---
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

# --- Database Model (Keep as before) ---
class EnergyTempReading(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    DateTime = db.Column(db.String(19), nullable=False, unique=True, index=True) # YYYY-MM-DD HH:00:00 UTC
    DailyEnergy = db.Column(db.Float) # Assumed Wh
    Temperature = db.Column(db.Float, nullable=True)

# --- NTP Time Fetch Function (Keep as before) ---
def get_ntp_time(server="pool.ntp.org"):
    # (Keep this function exactly as before)
    NTP_PORT = 123; NTP_PACKET_FORMAT = "!12I"; NTP_DELTA = 2208988800
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); client.settimeout(5); data = b'\x1b' + 47 * b'\0'; client.sendto(data, (server, NTP_PORT)); data, _ = client.recvfrom(1024)
        if data: secs = struct.unpack('!I', data[40:44])[0]; timestamp = secs - NTP_DELTA; return datetime.fromtimestamp(timestamp, timezone.utc)
        else: return None
    except socket.timeout: return None
    except Exception: return None
    finally:
        if 'client' in locals(): client.close()

# --- Background Data Fetching (Keep as before) ---
def fetch_and_store_hourly_data(target_hour_dt_utc: datetime):
    # (Keep this function exactly as before)
    with app.app_context():
        temp_val = None; formatted_hourly_timestamp = target_hour_dt_utc.strftime('%Y-%m-%d %H:%M:00')
        try:
            existing = EnergyTempReading.query.filter_by(DateTime=formatted_hourly_timestamp).first();
            if existing: return
            antares.setAccessKey(ANTARES_ACCESS_KEY); latest_data = antares.get(ANTARES_PROJECT_NAME, ANTARES_DEVICE_NAME)
            if not latest_data or 'content' not in latest_data: return
            content = latest_data['content']; energy_wh = content.get('Energy')
            if energy_wh is None: return
            if temp_fetch:
                try:
                    date_str = target_hour_dt_utc.strftime('%Y-%m-%d'); temp_df = temp_fetch(date_str, date_str, LATITUDE, LONGITUDE, historical=False) # historical=False
                    if temp_df is not None and not temp_df.empty:
                        temp_df.index = temp_df.index.tz_convert(timezone.utc); target_ts_pd = pd.Timestamp(target_hour_dt_utc); temp_val = temp_df['temperature'].get(target_ts_pd)
                except Exception as e: print(f"Hourly Check: Error getting temp forecast for {formatted_hourly_timestamp} UTC: {e}")
            new_reading = EnergyTempReading(DateTime=formatted_hourly_timestamp, DailyEnergy=energy_wh, Temperature=temp_val); db.session.add(new_reading); db.session.commit()
            print(f"Stored Hourly (NTP Check): {formatted_hourly_timestamp} UTC - Energy(Wh): {energy_wh}, Temp: {temp_val}")
        except Exception as e: print(f"Hourly Check Fetch/Store Error: {e}"); db.session.rollback()

# --- Background Checker (Keep as before) ---
def background_ntp_checker_compare_hour():
    # (Keep this function exactly as before)
    print("Background NTP Checker (Compare Hour) started...")
    previous_ntp_time_utc = None
    while True:
        current_ntp_time_utc = get_ntp_time(NTP_SERVER)
        if current_ntp_time_utc:
            trigger_fetch = False; target_hour_to_fetch = None
            if previous_ntp_time_utc is None: trigger_fetch = True; target_hour_to_fetch = current_ntp_time_utc.replace(minute=0, second=0, microsecond=0); print(f"NTP Check: First run. Targeting hour {target_hour_to_fetch.strftime('%H')}:00 UTC.")
            elif current_ntp_time_utc.hour != previous_ntp_time_utc.hour: trigger_fetch = True; target_hour_to_fetch = current_ntp_time_utc.replace(minute=0, second=0, microsecond=0); print(f"NTP Check: Hour changed from {previous_ntp_time_utc.hour} to {current_ntp_time_utc.hour}. Targeting {target_hour_to_fetch.strftime('%H')}:00 UTC.")
            if trigger_fetch and target_hour_to_fetch: fetch_and_store_hourly_data(target_hour_dt_utc=target_hour_to_fetch)
            previous_ntp_time_utc = current_ntp_time_utc
        else: print("NTP Check: Failed to get NTP time, skipping check cycle.")
        time.sleep(CHECK_INTERVAL_SECONDS)

# --- Flask Routes ---
@app.route('/')
def home(): return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard(): return render_template('dashboard.html')

# --- MODIFIED /database Route ---
@app.route('/database')
def database_view():
    # Fetch all readings to pass to the template for the table
    # Graph data will still be fetched via AJAX
    all_readings = EnergyTempReading.query.order_by(EnergyTempReading.DateTime.desc()).all()
    return render_template('database.html', readings=all_readings) # Pass readings here

# --- API Route for Graph Data (Keep as before) ---
@app.route('/get_range_data')
def get_range_data():
    # (Keep this function exactly as before)
    start_date_str = request.args.get('start_date'); end_date_str = request.args.get('end_date')
    if not start_date_str or not end_date_str: return jsonify({"error": "Provide start/end dates."}), 400
    try:
        start_datetime_str = f"{start_date_str} 00:00:00"; end_datetime_str = f"{end_date_str} 23:59:59"
        readings_in_range = EnergyTempReading.query.filter( EnergyTempReading.DateTime >= start_datetime_str, EnergyTempReading.DateTime <= end_datetime_str ).order_by(EnergyTempReading.DateTime).all()
        if not readings_in_range: return jsonify({"labels": [], "data": []})
        labels = [r.DateTime for r in readings_in_range]; data = [r.DailyEnergy if r.DailyEnergy is not None else 0 for r in readings_in_range]
        return jsonify({"labels": labels, "data": data})
    except Exception as e: print(f"Error in /get_range_data: {e}"); return jsonify({"error": "Internal error."}), 500

# --- Forecast Routes (Keep as before) ---
@app.route('/forecast')
def forecast_view():
    latest_reading = EnergyTempReading.query.order_by(EnergyTempReading.DateTime.desc()).first()
    return render_template('forecast.html', latest_reading=latest_reading)

@app.route('/run_forecast', methods=['POST'])
def run_forecast_api():
    # (Keep this function exactly as before)
    if model is None: return jsonify({"error": "Forecasting model not loaded."}), 500
    try:
        data = request.get_json(); timeframe = data.get('timeframe'); hours_map = {'1day': 24, '3days': 72, '1week': 168}
        if timeframe not in hours_map: return jsonify({"error": "Invalid timeframe"}), 400
        hours_to_forecast = hours_map[timeframe]; history_query = EnergyTempReading.query.order_by(EnergyTempReading.DateTime.desc()).limit(MAX_LAG).all()
        if len(history_query) < MAX_LAG: return jsonify({"error": f"Need {MAX_LAG} history points, found {len(history_query)}."}), 400
        history_df = pd.DataFrame([{'DateTime': r.DateTime, TARGET_VARIABLE: r.DailyEnergy, 'Temperature': r.Temperature} for r in history_query]).set_index('DateTime').iloc[::-1]; history_df.index = pd.to_datetime(history_df.index + '+00:00').tz_convert(API_TIMEZONE)
        last_known_time_local = history_df.index[-1]; forecast_start_dt_local = last_known_time_local + timedelta(hours=1); forecast_end_dt_local = last_known_time_local + timedelta(hours=hours_to_forecast); future_temps = None
        if temp_fetch:
            try:
                start_date_api = forecast_start_dt_local.strftime('%Y-%m-%d'); end_date_api = forecast_end_dt_local.strftime('%Y-%m-%d'); temp_df = temp_fetch(start_date_api, end_date_api, LATITUDE, LONGITUDE, historical=False) # historical=False
                if temp_df is not None and not temp_df.empty: temp_df.index = temp_df.index.tz_convert(API_TIMEZONE); future_index = pd.date_range(start=forecast_start_dt_local, periods=hours_to_forecast, freq='h', tz=API_TIMEZONE); future_temps = temp_df['temperature'].reindex(future_index, method='ffill')
            except Exception as e: print(f"Warning: Failed future temp fetch: {e}")
        if future_temps is None: last_temp = history_df['Temperature'].iloc[-1] if pd.notna(history_df['Temperature'].iloc[-1]) else 25.0; future_temps = pd.Series([last_temp] * hours_to_forecast, index=pd.date_range(start=forecast_start_dt_local, periods=hours_to_forecast, freq='h', tz=API_TIMEZONE))
        predictions, timestamps = [], []; current_data = history_df[[TARGET_VARIABLE]].copy()
        for h in range(hours_to_forecast):
            current_pred_time_local = forecast_start_dt_local + timedelta(hours=h); timestamps.append(current_pred_time_local.strftime('%Y-%m-%d %H:%M:00'))
            features = {'hour': current_pred_time_local.hour, 'day_of_week': current_pred_time_local.dayofweek,'day_of_month': current_pred_time_local.day, 'is_weekend': 1 if current_pred_time_local.dayofweek >= 5 else 0,'temperature': future_temps.get(current_pred_time_local, future_temps.iloc[-1])}
            for lag in [1, 24, 168]: features[f'lag_{lag}'] = current_data[TARGET_VARIABLE].get(current_pred_time_local - timedelta(hours=lag), np.nan)
            feature_vector_df = pd.DataFrame([features], columns=MODEL_FEATURES).fillna(0); prediction = max(0.0, float(model.predict(feature_vector_df)[0]))
            predictions.append(prediction); new_row = pd.DataFrame({TARGET_VARIABLE: [prediction]}, index=[current_pred_time_local]); current_data = pd.concat([current_data, new_row])
        return jsonify({"labels": timestamps, "data": predictions})
    except Exception as e: print(f"Error in /run_forecast: {e}"); import traceback; traceback.print_exc(); return jsonify({"error": "Internal forecast error."}), 500

# --- Main Execution (Keep as before) ---
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print(f"Database '{DATABASE_FILE}' ensured.")
    fetch_thread = threading.Thread(target=background_ntp_checker_compare_hour, daemon=True)
    fetch_thread.start()
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

