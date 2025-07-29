# direct_db_injector.py
import sqlite3
import csv
import pandas as pd
from datetime import timezone as dt_timezone, timedelta
import os
import numpy as np

# --- OpenMeteo Fetch Utility ---
import openmeteo_requests
import requests_cache
from retry_requests import retry

LATITUDE_CONFIG = 14.5833  # Define these, or pass them as args
LONGITUDE_CONFIG = 121.0
CSV_DATETIME_INPUT_TIMEZONE = "Asia/Kuala_Lumpur"

def fetch_temperature_for_date_range(start_date_utc_str: str, end_date_utc_str: str, latitude, longitude):
    """
    Fetches all hourly temperatures from OpenMeteo for a given UTC date range.
    Returns a pandas Series with DatetimeIndex (UTC) and temperature values, or None on error.
    """
    cache_session = requests_cache.CachedSession('.cache_openmeteo_injector_range', expire_after=3600)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    # Use archive API for historical data
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude, "longitude": longitude,
        "hourly": "temperature_2m", "timezone": "UTC", # Always fetch UTC
        "start_date": start_date_utc_str, "end_date": end_date_utc_str
    }
    try:
        print(f"  Prefetching temperatures from {start_date_utc_str} to {end_date_utc_str} UTC...")
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
        if hourly.VariablesLength() <= 0 : # Check if temperature variable is available
             print("  Warning: No hourly temperature data returned by API for the range.")
             return pd.Series(dtype=float)


        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

        hourly_data_index = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
        temp_series_utc = pd.Series(data=hourly_temperature_2m, index=hourly_data_index, name='temperature')
        print(f"  Successfully prefetched {len(temp_series_utc)} temperature points.")
        return temp_series_utc
    except Exception as e:
        print(f"  Error prefetching temperatures for range {start_date_utc_str} to {end_date_utc_str}: {e}")
        return None


# --- General Configuration ---
DATABASE_FILE = 'app.db'
TABLE_NAME = 'hourly_reading'
DB_STORAGE_TIMEZONE_STR = 'UTC'

CSV_COL_DATETIME = 'DateTime'
CSV_COL_ENERGY = 'Wh' # Or your CSV header

# --- Database Functions (create_connection, create_table_if_not_exists, insert_reading) ---
# ... (These functions remain the same as in your last version) ...
def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        print(f"Connected to SQLite: {db_file} (SQLite version: {sqlite3.sqlite_version})")
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
    return conn

def create_table_if_not_exists(conn):
    # Change 'timestamp_utc' to 'timestamp' in this SQL command
    sql_create_table = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL UNIQUE,
        EnergyWh REAL,
        TemperatureCelsius REAL
    );
    """
    try:
        cursor = conn.cursor()
        cursor.execute(sql_create_table)
        conn.commit()
        print(f"Table '{TABLE_NAME}' ensured/created with 'timestamp' column.")
    except sqlite3.Error as e:
        print(f"Error creating table: {e}")

def insert_reading(conn, timestamp, energy, temperature):
    # Change 'timestamp_utc' to 'timestamp' in this SQL command
    sql_insert = f"""
    INSERT INTO {TABLE_NAME} (timestamp, EnergyWh, TemperatureCelsius)
    VALUES (?, ?, ?);
    """
    cursor = conn.cursor()
    try:
        # Pass the new 'timestamp' variable here
        cursor.execute(sql_insert, (timestamp, energy, temperature))
        return True
    except sqlite3.IntegrityError:
        return False
    except sqlite3.Error as e:
        print(f"  Error inserting data for {timestamp}: {e}")
        return False


def inject_data_from_csv_with_temp_prefetch(csv_filepath):
    conn = create_connection(DATABASE_FILE)
    if conn is None: return

    create_table_if_not_exists(conn)
    added_count, skipped_count, processed_rows, current_batch_count = 0, 0, 0, 0
    batch_size = 1000000

    try:
        # Step 1: Read CSV to determine date range
        print(f"Reading CSV to determine date range: {csv_filepath}")
        all_csv_datetimes_utc = []
        temp_df_for_dates = pd.read_csv(csv_filepath, usecols=[CSV_COL_DATETIME], encoding='utf-8-sig')
        if temp_df_for_dates.empty:
            print("CSV is empty. Exiting.")
            conn.close(); return

        for dt_str_from_csv in temp_df_for_dates[CSV_COL_DATETIME]:
            try:
                dt_naive_or_aware = pd.to_datetime(dt_str_from_csv)
                dt_localized = dt_naive_or_aware.tz_localize(CSV_DATETIME_INPUT_TIMEZONE) if dt_naive_or_aware.tzinfo is None else dt_naive_or_aware
                all_csv_datetimes_utc.append(dt_localized.tz_convert(DB_STORAGE_TIMEZONE_STR))
            except Exception as e_date_parse:
                print(f"  Warning: Could not parse DateTime '{dt_str_from_csv}'. Skipping. Error: {e_date_parse}")
                continue
        
        if not all_csv_datetimes_utc:
            print("No valid DateTimes found in CSV. Exiting.")
            conn.close(); return

        # --- NEW LOGIC: Calculate the time offset using Jakarta Time ---
        last_csv_time_utc = max(all_csv_datetimes_utc)
        
        # 1. Define the target end time in Asia/Jakarta timezone
        target_end_time_jakarta = pd.Timestamp("2025-07-27 23:00:00", tz='Asia/Jakarta')
        
        # 2. Convert target to UTC to correctly calculate the difference with our UTC data
        target_end_time_utc = target_end_time_jakarta

        time_offset = target_end_time_utc - last_csv_time_utc
        print(f"\nTime Adjustment Calculated:")
        print(f"  Last timestamp in CSV (UTC): {last_csv_time_utc}")
        print(f"  Target end timestamp (Jakarta): {target_end_time_jakarta}")
        print(f"  Applying offset of: {time_offset}")

        # Adjust the min/max dates for the temperature API call
        min_date_utc_adjusted = min(all_csv_datetimes_utc) + time_offset
        max_date_utc_adjusted = last_csv_time_utc + time_offset
        
        min_date_utc_str = min_date_utc_adjusted.strftime('%Y-%m-%d')
        max_date_utc_str = max_date_utc_adjusted.strftime('%Y-%m-%d')

        # --- Step 2: Prefetch all temperatures for the NEW, adjusted date range ---
        prefetched_temps_utc = fetch_temperature_for_date_range(
            min_date_utc_str, max_date_utc_str, LATITUDE_CONFIG, LONGITUDE_CONFIG
        )

        if prefetched_temps_utc is None:
            print("Failed to prefetch temperatures. Aborting injection.")
            conn.close(); return
        if prefetched_temps_utc.empty:
            print("Warning: Prefetched temperature data is empty. Temperatures will be NULL.")
            prefetched_temps_utc = pd.Series(dtype=float, index=pd.to_datetime([]))

        # --- Step 3: Process CSV and insert into DB with adjusted timestamps ---
        print(f"\nProcessing CSV for database injection...")
        with open(csv_filepath, mode='r', encoding='utf-8-sig') as csvfile:
            csv_reader = csv.DictReader(csvfile)
            # ... (header check remains the same) ...

            for row_num, row in enumerate(csv_reader, 1):
                # ... (error handling and parsing remains the same) ...
                try:
                    dt_str_from_csv = row.get(CSV_COL_DATETIME)
                    energy_str = row.get(CSV_COL_ENERGY)
                    
                    dt_naive_or_aware = pd.to_datetime(dt_str_from_csv)
                    dt_localized = dt_naive_or_aware.tz_localize(CSV_DATETIME_INPUT_TIMEZONE) if dt_naive_or_aware.tzinfo is None else dt_naive_or_aware
                    dt_utc_original = dt_localized.tz_convert(DB_STORAGE_TIMEZONE_STR)

                    # Apply the calculated offset to the timestamp
                    dt_utc_adjusted = dt_utc_original + time_offset
                    
                    dt_utc_for_lookup = dt_utc_adjusted.replace(minute=0, second=0, microsecond=0)
                    db_timestamp_iso_str = dt_utc_for_lookup.isoformat()

                    temp_val = prefetched_temps_utc.get(dt_utc_for_lookup)
                    if pd.isna(temp_val):
                        temp_val = None
                    
                    energy_val = float(energy_str)
                    final_temp_val = float(temp_val) if temp_val is not None else None

                    if insert_reading(conn, db_timestamp_iso_str, energy_val, final_temp_val):
                        added_count += 1; current_batch_count += 1
                    else:
                        skipped_count += 1
                    
                    if current_batch_count >= batch_size:
                        conn.commit(); print(f"  Committed batch of {current_batch_count} records."); current_batch_count = 0
                
                except Exception as e_row:
                    print(f"  Row {row_num}: Error processing. Data: {row}. Error: {e_row}")
                    skipped_count += 1
            
            if current_batch_count > 0: conn.commit(); print(f"  Committed final batch of {current_batch_count} records.")
        
        print(f"\n--- Injection Summary ---\nProcessed {processed_rows} rows.\nAdded {added_count} new records.\nSkipped {skipped_count} records.")

    except Exception as e: 
        print(f"Unexpected error: {e}"); import traceback; traceback.print_exc()
    finally:
        if conn: conn.close(); print("DB connection closed.")


# csv_file_to_inject = 'Data/processed_hourly_Wh_data.csv'
csv_file_to_inject = 'test_sidang.csv'
if not csv_file_to_inject: print("No CSV specified. Exiting.")
elif not os.path.exists(csv_file_to_inject): print(f"Error: File '{csv_file_to_inject}' does not exist.")
else:
    print(f"Starting data injection from: {csv_file_to_inject}")
    print(f"Using Location: Kuala Lumpur (Lat: {LATITUDE_CONFIG}, Lon: {LONGITUDE_CONFIG})")
    print(f"Assuming CSV DateTime input timezone: {CSV_DATETIME_INPUT_TIMEZONE}")
    inject_data_from_csv_with_temp_prefetch(csv_file_to_inject)