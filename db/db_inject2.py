import pandas as pd
import sqlite3
import sys

# --- CONFIGURATION ---
DB_FILE = 'app1.db'
CSV_FILE = 'tes_sidang_28.csv'
# IMPORTANT: Change this to the name of the table in your database.
# I've guessed 'electricity_data'. If this is wrong, the script will
# print a list of available tables for you to choose from.
TABLE_NAME = 'hourly_reading' 
# The date from which to copy the temperature data
TEMP_SOURCE_DATE = '2025-07-20'

def list_tables(conn):
    """Helper function to list all tables in the database."""
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    cursor.close()
    return [table[0] for table in tables]

def update_database():
    """
    Adds new electricity data to the database, reusing temperature data
    from a specified date.
    """
    try:
        # --- 1. Connect to the SQLite database ---
        print(f"Connecting to database '{DB_FILE}'...")
        conn = sqlite3.connect(DB_FILE)

        # Check if the specified table exists
        available_tables = list_tables(conn)
        if TABLE_NAME not in available_tables:
            print(f"\n[ERROR] Table '{TABLE_NAME}' not found in the database.")
            if available_tables:
                print("Available tables are:", ", ".join(available_tables))
                print("Please update the TABLE_NAME variable in the script and run again.")
            else:
                print("The database appears to be empty. No tables found.")
            sys.exit(1) # Exit the script

        # --- 2. Get the temperature data for the source date ---
        print(f"Fetching temperature data from {TEMP_SOURCE_DATE}...")
        query = f"""
        SELECT strftime('%H', timestamp) as hour, TemperatureCelsius
        FROM {TABLE_NAME} 
        WHERE date(timestamp) = '{TEMP_SOURCE_DATE}'
        """
        temp_df = pd.read_sql_query(query, conn)

        if temp_df.empty:
            print(f"\n[ERROR] No temperature data found for {TEMP_SOURCE_DATE} in the database.")
            print("Please ensure the database contains data for this date.")
            conn.close()
            sys.exit(1)

        # Create a mapping from hour to temperature
        # We drop duplicates in case there are multiple readings for the same hour
        temp_map = temp_df.drop_duplicates(subset='hour').set_index('hour')['temperature']
        print(f"Successfully created a 24-hour temperature map.")

        # --- 3. Read the new data from the CSV file ---
        print(f"Reading new data from '{CSV_FILE}'...")
        new_data_df = pd.read_csv(CSV_FILE, parse_dates=['DateTime'])

        # --- 4. Combine new data with the temperature map ---
        print("Applying temperature data to new records...")
        # Get the hour from the DateTime column to use for mapping
        new_data_df['hour'] = new_data_df['DateTime'].dt.strftime('%H')
        new_data_df['temperature'] = new_data_df['hour'].map(temp_map)
        
        # Check if any temperature values failed to map
        if new_data_df['temperature'].isnull().any():
            print("\n[WARNING] Some hours in the new data could not be mapped to a temperature.")
            print("These rows will be filled with the average temperature.")
            avg_temp = temp_map.mean()
            new_data_df['temperature'].fillna(avg_temp, inplace=True)

        # Prepare the final DataFrame for insertion
        df_to_insert = new_data_df[['DateTime', 'Wh', 'temperature']]

        # --- 5. Add the new records to the database ---
        print(f"Adding {len(df_to_insert)} new records to the '{TABLE_NAME}' table...")
        df_to_insert.to_sql(TABLE_NAME, conn, if_exists='append', index=False)
        
        print("\n--- Success! ---")
        print("The new data has been successfully added to the database.")

    except sqlite3.Error as e:
        print(f"\n[DATABASE ERROR] An error occurred: {e}")
    except FileNotFoundError:
        print(f"\n[FILE ERROR] Make sure '{CSV_FILE}' and '{DB_FILE}' are in the same directory.")
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] An unexpected error occurred: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()
            print("Database connection closed.")

# --- Run the main function ---
if __name__ == "__main__":
    update_database()
