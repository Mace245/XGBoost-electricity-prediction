import time
import random
from datetime import datetime, timedelta
from sqlalchemy.exc import IntegrityError
import os
from app import get_ntp_time

# --- IMPORTANT: Assumes app.py is in the same directory or Python path ---
try:
    # Import the necessary Flask app instance, db object, and Model
    # Make sure your main Flask app file is named 'app.py'
    from app import app, db, EnergyTempReading
except ImportError as e:
    print(f"Error: Could not import 'app', 'db', or 'EnergyTempReading' from app.py.")
    print(f"Ensure this script is in the same directory as app.py or that app.py is in the Python path.")
    print(f"Details: {e}")
    exit()

# --- Configuration ---
INSERT_INTERVAL_SECONDS = 0.1 # How often to insert data (make it different from Antares sender)
START_DATETIME_STR = get_ntp_time("pool.ntp.org") # Initial timestamp for dummy data

# --- Main Insertion Loop ---
print(f"Starting dummy data inserter for database '{app.config['SQLALCHEMY_DATABASE_URI']}'...")
print(f"Inserting data every {INSERT_INTERVAL_SECONDS} seconds. Press Ctrl+C to stop.")

# Initialize the starting datetime
try:
    current_dummy_time = START_DATETIME_STR.replace(minute=0, second=0, microsecond=0)
except ValueError:
    print(f"Error: Invalid START_DATETIME_STR format. Use 'YYYY-MM-DD HH:MM:SS'.")
    exit()

while True:
    # Use the Flask app context to interact with the database
    with app.app_context():
        try:
            # 1. Generate Dummy Data
            # Adjust ranges as needed
            dummy_daily_energy = round(random.uniform(0.0, 50.0), 4) # Smaller increment for hourly simulation
            dummy_temperature = round(random.uniform(20.0, 35.0), 2)
            formatted_timestamp = current_dummy_time.strftime('%Y-%m-%d %H:%M:%S')

            # 2. Create DB Record Object
            new_reading = EnergyTempReading(
                DateTime=formatted_timestamp,
                DailyEnergy=dummy_daily_energy,
                Temperature=dummy_temperature
            )

            # 3. Add and Commit (with duplicate check via exception)
            db.session.add(new_reading)
            try:
                db.session.commit()
                print(f"[{time.strftime('%H:%M:%S')}] Inserted: {formatted_timestamp} - Energy: {dummy_daily_energy:.4f}, Temp: {dummy_temperature:.2f}")
            except IntegrityError:
                # This error occurs if the DateTime (unique key) already exists
                db.session.rollback() # Rollback the failed transaction
                print(f"[{time.strftime('%H:%M:%S')}] Skipped: Data for {formatted_timestamp} already exists.")
            except Exception as commit_err:
                # Catch other potential commit errors
                db.session.rollback()
                print(f"[{time.strftime('%H:%M:%S')}] Error committing to DB for {formatted_timestamp}: {commit_err}")

        except Exception as outer_err:
            print(f"An outer error occurred: {outer_err}")
            # Ensure rollback if error happens before commit attempt
            db.session.rollback()

    # 4. Increment Timestamp for next iteration (e.g., by 1 hour)
    current_dummy_time += timedelta(hours=1)

    # 5. Wait for the next interval
    try:
        time.sleep(INSERT_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nStopping inserter.")
        break
