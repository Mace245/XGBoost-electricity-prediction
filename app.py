import os
import time
import threading
from flask import Flask, render_template, jsonify, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import pandas as pd
import numpy as np # For dummy forecast data
from antares_http import antares

# --- Import the updated temp_fetch function ---
try:
    from lib.data import temp_fetch
except ImportError:
    print("ERROR: temp_api.py not found or temp_fetch function missing. Temperature fetching will fail.")
    temp_fetch = None

# --- Configuration ---
ANTARES_ACCESS_KEY = '5cd4cda046471a89:75f9e1c6b34bf41a' # Replace with your Access Key[1]
ANTARES_PROJECT_NAME = 'UjiCoba_TA'                   # Replace with your Project Name[1]
ANTARES_DEVICE_NAME = 'TA_DKT1'                     # Replace with your Device Name[1]
DATABASE_FILE = 'dashboard_app.db'                  # New DB filename
FETCH_INTERVAL_SECONDS = 10                         # Data fetching interval
LATITUDE = 14.5833
LONGITUDE = 121.0
API_TIMEZONE = "Asia/Jakarta"

# --- Initialize Flask App & SQLAlchemy ---
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DATABASE_FILE}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

app.jinja_env.globals['now'] = datetime.now

# --- SQLAlchemy Database Model ---
class EnergyTempReading(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    DateTime = db.Column(db.String(19), nullable=False, unique=True, index=True)
    DailyEnergy = db.Column(db.Float) # Assuming this corresponds to 'Wh' in the request context
    Temperature = db.Column(db.Float, nullable=True)

    def __repr__(self):
        return f'<Reading {self.DateTime}: Energy={self.DailyEnergy}, Temp={self.Temperature}>'

# --- Temperature Fetching (Adapted from previous version) ---
def get_temperature_for_timestamp(dt_object, lat, lon):
    # (Code from previous version - slightly simplified for brevity)
    try:
        date_str = dt_object.strftime('%Y-%m-%d')
        temp_df = temp_fetch(start_date=date_str, end_date=date_str, latitude=lat, longitude=lon, historical=False)
        if temp_df is None or temp_df.empty: return None
        try:
            temp_df.index = temp_df.index.tz_convert(API_TIMEZONE)
        except Exception:
            pass # Use UTC index if conversion fails
        lookup_ts = dt_object.replace(minute=0, second=0, microsecond=0)
        target_timestamp = pd.Timestamp(lookup_ts, tz=temp_df.index.tz) # Use DataFrame's actual timezone
        if target_timestamp in temp_df.index:
            return temp_df.loc[target_timestamp, 'temperature']
        else:
            # Try nearest if exact hour isn't found (optional)
            nearest_idx = temp_df.index.get_indexer([target_timestamp], method='nearest')
            if nearest_idx.size > 0 and nearest_idx[0] != -1:
                return temp_df.iloc[nearest_idx[0]]['temperature']
            return None
    except Exception as e:
        print(f"Error fetching/processing temperature: {e}")
        return None


# --- Antares Fetch and Store Function (Modified to use Energy/Wh) ---
def fetch_and_store_antares_data():
    """Fetches latest Antares data & temperature, stores if new."""
    with app.app_context():
        # print(f"{datetime.now()}: Attempting to fetch Antares data...") # Less verbose logging
        temperature = None
        try:
            antares.setAccessKey(ANTARES_ACCESS_KEY)
            latest_data = antares.get(ANTARES_PROJECT_NAME, ANTARES_DEVICE_NAME)

            if not latest_data or 'content' not in latest_data or 'last_modified_time' not in latest_data:
                # print("Error: Invalid data received from Antares.")
                return

            content = latest_data['content']
            timestamp_str = latest_data['last_modified_time']
            try:
                dt_object = datetime.strptime(timestamp_str, '%Y%m%dT%H%M%S')
                formatted_timestamp = dt_object.strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                return # Skip if timestamp invalid

            energy_wh = content.get('Energy') # GET THE CORRECT FIELD FOR WH

            # Fetch temperature
            temperature = get_temperature_for_timestamp(dt_object, LATITUDE, LONGITUDE)

            # Check for Duplicates
            existing_reading = EnergyTempReading.query.filter_by(DateTime=formatted_timestamp).first()

            if not existing_reading:
                # Store New Data
                if energy_wh is not None: # Make sure we have the value
                    new_reading = EnergyTempReading(
                        DateTime=formatted_timestamp,
                        DailyEnergy=energy_wh, # Storing the Wh value here
                        Temperature=temperature
                    )
                    db.session.add(new_reading)
                    db.session.commit()
                    print(f"Stored: {formatted_timestamp} - Energy(Wh): {energy_wh}, Temp: {temperature}")
                # else: print(f"Warning: Energy(Wh) missing for {formatted_timestamp}. Not storing.")

        except Exception as e:
            print(f"An error occurred during fetch/store: {e}")
            db.session.rollback()

# --- Background Thread Function (Unchanged) ---
def background_fetcher():
    """Function that runs in the background to fetch data periodically."""
    print("Background fetcher started...")
    while True:
        fetch_and_store_antares_data()
        time.sleep(FETCH_INTERVAL_SECONDS)

# --- Placeholder Forecast Function ---
def generate_dummy_forecast(timeframe_hours):
    """Generates dummy forecast data for plotting."""
    start_time = datetime.now()
    timestamps = [(start_time + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S') for i in range(timeframe_hours)]
    # Simple sine wave + noise for dummy data
    dummy_values = 150 + 50 * np.sin(np.linspace(0, 4 * np.pi, timeframe_hours)) + np.random.rand(timeframe_hours) * 20
    return {"labels": timestamps, "data": dummy_values.tolist()}

# --- Flask Routes ---
@app.route('/')
def home():
    """Redirect base URL to the dashboard."""
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    """Displays the main dashboard page."""
    return render_template('dashboard.html')

@app.route('/database')
def database_view():
    """Displays the stored energy and temperature readings."""
    readings = EnergyTempReading.query.order_by(EnergyTempReading.DateTime.desc()).all()
    return render_template('database.html', readings=readings)

@app.route('/forecast')
def forecast_view():
    """Displays the forecasting interface."""
    # Get the latest reading to show current usage
    latest_reading = EnergyTempReading.query.order_by(EnergyTempReading.DateTime.desc()).first()
    return render_template('forecast.html', latest_reading=latest_reading)

@app.route('/run_forecast', methods=['POST'])
def run_forecast_api():
    """API endpoint to generate forecast data (currently placeholder)."""
    try:
        data = request.get_json()
        timeframe = data.get('timeframe') # '1day', '3days', '1week'

        if timeframe == '1day':
            hours = 24
        elif timeframe == '3days':
            hours = 72
        elif timeframe == '1week':
            hours = 168
        else:
            return jsonify({"error": "Invalid timeframe"}), 400

        print(f"Generating dummy forecast for {timeframe} ({hours} hours)")
        forecast_data = generate_dummy_forecast(hours)
        return jsonify(forecast_data)

    except Exception as e:
        print(f"Error in /run_forecast: {e}")
        return jsonify({"error": str(e)}), 500


# --- Main Execution Block ---
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print(f"Database '{DATABASE_FILE}' and table 'energy_temp_reading' ensured.")

    fetch_thread = threading.Thread(target=background_fetcher, daemon=True)
    fetch_thread.start()

    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

