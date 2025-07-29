import time
import random
from datetime import datetime, timedelta, timezone
from sqlalchemy.exc import IntegrityError
import os
import pandas as pd
import sys

# --- IMPORTANT: Assumes app.py is in the same directory or Python path ---
try:
    # Import the necessary Flask app instance, db object, and Model
    # Make sure your main Flask app file is named 'app.py'
    from app import app, db, EnergyTempReading, get_ntp_time
    print("Successfully imported Flask app, db, EnergyTempReading, get_ntp_time.")
except ImportError as e:
    print(f"Error: Could not import from app.py.")
    print(f"Ensure this script is in the same directory as app.py or that app.py is in the Python path.")
    print(f"Details: {e}")
    sys.exit(1) # Use sys.exit for clearer termination

# --- Configuration ---
INSERT_INTERVAL_SECONDS = 0.1 # How often to insert data (controls simulation speed)
CSV_FILE_PATH = 'Data/processed_hourly_Wh_data.csv' # Path to your historical data
# **** IMPORTANT: Verify this column name matches your CSV file ****
ENERGY_COLUMN_NAME = 'Wh' # Column containing the hourly energy values
# Define how far back from the *current* time the historical data should start
# e.g., timedelta(days=7) means the first record from the CSV will correspond
# to the timestamp 7 days before the current time (rounded to the hour).
# Set to timedelta(0) to start the historical data at the current hour.
START_OFFSET_FROM_NOW = timedelta(days=14) # Example: Start data 2 weeks ago

# --- Load Historical Data ---
print(f"Loading historical data from: {CSV_FILE_PATH}")
if not os.path.exists(CSV_FILE_PATH):
    print(f"ERROR: Historical data file not found at '{CSV_FILE_PATH}'")
    sys.exit(1)

try:
    historical_df = pd.read_csv(CSV_FILE_PATH, parse_dates=['DateTime'])
    # Ensure the energy column exists
    if ENERGY_COLUMN_NAME not in historical_df.columns:
        print(f"ERROR: Energy column '{ENERGY_COLUMN_NAME}' not found in CSV.")
        print(f"Available columns: {historical_df.columns.tolist()}")
        sys.exit(1)
    # Ensure DateTime column exists and is parsed
    if 'DateTime' not in historical_df.columns:
         print(f"ERROR: 'DateTime' column not found in CSV.")
         sys.exit(1)

    historical_df.set_index('DateTime', inplace=True)
    historical_df.sort_index(inplace=True) # Ensure data is chronological
    print(f"Loaded {len(historical_df)} records from CSV.")
    print(f"CSV Data Range: {historical_df.index.min()} to {historical_df.index.max()}")

    # Make CSV timestamps timezone-aware (assuming they are UTC or naive UTC)
    if historical_df.index.tz is None:
        historical_df.index = historical_df.index.tz_localize('UTC')
    else:
        historical_df.index = historical_df.index.tz_convert('UTC') # Ensure UTC


except Exception as e:
    print(f"ERROR: Failed to load or process CSV file: {e}")
    sys.exit(1)

# --- Calculate Timestamp Offset ---
try:
    # Get the current time (UTC) and round down to the start of the hour
    now_utc = get_ntp_time("pool.ntp.org")
    if now_utc is None:
        print("ERROR: Failed to get current time from NTP server.")
        sys.exit(1)

    current_target_start_time = (now_utc - START_OFFSET_FROM_NOW).replace(minute=0, second=0, microsecond=0)

    # Get the first timestamp from the historical data
    first_csv_timestamp = historical_df.index[0]

    # Calculate the difference needed to shift the historical data
    time_offset = current_target_start_time - first_csv_timestamp

    print(f"Current UTC time (rounded): {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Target start time for DB insert: {current_target_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"First timestamp in CSV: {first_csv_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Calculated time offset: {time_offset}")

except Exception as e:
    print(f"ERROR: Failed during timestamp calculation: {e}")
    sys.exit(1)


# --- Main Insertion Loop ---
print(f"\nStarting data insertion into database '{app.config['SQLALCHEMY_DATABASE_URI']}'...")
print(f"Inserting data every {INSERT_INTERVAL_SECONDS} seconds. Press Ctrl+C to stop.")
print("Data from CSV will be timestamp-shifted to appear recent.")

inserted_count = 0
skipped_count = 0

# Loop through the historical data loaded from the CSV
for original_csv_timestamp, row in historical_df.iterrows():
    # Use the Flask app context to interact with the database
    with app.app_context():
        try:
            # 1. Get Actual Energy Value
            actual_energy = row[ENERGY_COLUMN_NAME]

            # Handle potential NaN values in the energy column
            if pd.isna(actual_energy):
                print(f"[{time.strftime('%H:%M:%S')}] Skipped: NaN value found for energy at original timestamp {original_csv_timestamp.strftime('%Y-%m-%d %H:%M:%S')}.")
                skipped_count += 1
                continue # Skip this row

            # 2. Calculate New Timestamp (Shifted to be current)
            current_db_timestamp = original_csv_timestamp + time_offset
            formatted_timestamp = current_db_timestamp.strftime('%Y-%m-%d %H:%M:00') # Match app.py format (HH:MM:00)

            # 3. Generate Dummy Temperature (as in original script)
            #    Alternatively, you could fetch *actual* historical temperature
            #    corresponding to 'original_csv_timestamp' if needed for accuracy.
            dummy_temperature = round(random.uniform(20.0, 35.0), 2)

            # 4. Create DB Record Object
            new_reading = EnergyTempReading(
                DateTime=formatted_timestamp, # Use shifted, formatted timestamp
                DailyEnergy=actual_energy,    # Use actual energy from CSV
                Temperature=dummy_temperature # Use dummy temperature
            )

            # 5. Add and Commit (with duplicate check via exception)
            db.session.add(new_reading)
            try:
                db.session.commit()
                print(f"[{time.strftime('%H:%M:%S')}] Inserted: {formatted_timestamp} - Energy: {actual_energy:.4f}, Temp: {dummy_temperature:.2f} (Original: {original_csv_timestamp.strftime('%Y-%m-%d %H:%M')})")
                inserted_count += 1
            except IntegrityError:
                # This error occurs if the DateTime (unique key) already exists
                db.session.rollback() # Rollback the failed transaction
                print(f"[{time.strftime('%H:%M:%S')}] Skipped: Data for {formatted_timestamp} already exists.")
                skipped_count += 1
            except Exception as commit_err:
                # Catch other potential commit errors
                db.session.rollback()
                print(f"[{time.strftime('%H:%M:%S')}] DB Commit Error for {formatted_timestamp}: {commit_err}")
                skipped_count += 1

        except Exception as outer_err:
            print(f"[{time.strftime('%H:%M:%S')}] Outer error processing row for {original_csv_timestamp}: {outer_err}")
            # Ensure rollback if error happens before commit attempt
            db.session.rollback()
            skipped_count += 1

    # 6. Wait for the next interval to control insertion speed
    try:
        time.sleep(INSERT_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nStopping inserter...")
        break # Exit the loop if Ctrl+C is pressed

print("\n--- Insertion Finished ---")
print(f"Total records inserted: {inserted_count}")
print(f"Total records skipped (duplicate/error/NaN): {skipped_count}")
print(f"Last inserted timestamp (shifted): {formatted_timestamp if 'formatted_timestamp' in locals() else 'N/A'}")