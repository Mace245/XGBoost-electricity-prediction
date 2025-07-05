import csv
import os
from datetime import datetime
import glob
import pandas as pd


def csvmaker(folder_path, output_file):
    csv_files = glob.glob(os.path.join(folder_path, '*.txt'))
    csv_files.sort(key=os.path.getmtime, reverse=False) # sort files by modification time, descending

    count = 0
    if os.path.isfile(output_file):
        os.remove(output_file)

    outfile = open(output_file, 'w', newline='')

    csv_writer = csv.writer(outfile)
    # Write header
    csv_writer.writerow(["Global_active_power", "Time", "Date"])

    for file in csv_files:
        count += 1
        with open(file, 'r') as infile:
                    # Extract the date from the filename
            # Assuming the filename format is "BEMS-<date>.txt", where <date> is in the form "1_1_2023"
            basename = os.path.basename(file)
            name_no_ext = os.path.splitext(basename)[0]  # "BEMS-1_1_2023"
            parts = name_no_ext.split('-')
            date_part = parts[1]  # "1_1_2023"

            # Convert "1_1_2023" → "2023-01-01"
            parsed_date = datetime.strptime(date_part, "%d_%m_%Y").strftime("%d/%m/%Y")
            
            for line in infile:
                row = line.strip().split('\t')
                # Ensure there are at least 64 columns
                if len(row) == 64:
                    # Python indexing: column AG (33rd) is index 32, column BL (64th) is index 63
                    timeDate= row[63]

                    selected = [row[32], timeDate, parsed_date]
                    csv_writer.writerow(selected)

    outfile.close()

    # input_file = 'BEMS-15_3_2023.txt'
    print(count)

# csvmaker("Meter-26", "output26.csv")
# csvmaker("Meter-27", "output27.csv")

df26 = pd.read_csv('output26.csv')
df27 = pd.read_csv('output27.csv')

for temp_df in [df26, df27]:
    temp_df['DateTime'] = pd.to_datetime(
        temp_df['Date'] + ' ' + temp_df['Time'],
        format='%d/%m/%Y %H:%M:%S'
    )

# --- Step 2: Set 'DateTime' as the index for BOTH DataFrames ---
# This is the crucial step for aligning by time.

df26['DateTime_minute'] = df26['DateTime'].dt.floor('T')
df27['DateTime_minute'] = df27['DateTime'].dt.floor('T')

df26.set_index('DateTime_minute', inplace=True)
df27.set_index('DateTime_minute', inplace=True)

mismatched_timestamps = df26.index.symmetric_difference(df27.index)
fill_count = len(mismatched_timestamps)
print(f"The 'fill_value=0' will be used {fill_count} time(s).")
print(mismatched_timestamps)


# --- Step 3: Add the columns based on the new DateTime index ---
# We still use .add() with fill_value=0 in case a specific minute
# exists in one file but not the other.
summed_power = df26['Global_active_power'].add(df27['Global_active_power'], fill_value=0)


# --- Step 4: Create the final DataFrame ---
# 'summed_power' is currently a Series with a DateTime index.
# We convert it back into a DataFrame with regular columns.
final_df = summed_power.reset_index()

# --- You're done! ---
print("--- Final DataFrame Summed by Time ---")
final_df = final_df.rename(columns={"DateTime_minute": "DateTime"})
print(final_df)

final_df.to_csv('output.csv', index=False)

mismatched_timestamps = pd.Series(mismatched_timestamps)
mismatched_timestamps.to_csv('missed_csvmaker.csv')

