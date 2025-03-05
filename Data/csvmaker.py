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
                    time_12h = row[63]
                    try:
                        # Convert 12-hour time to 24-hour format
                        time_24h = datetime.strptime(time_12h, "%I:%M:%S %p").strftime("%H:%M:%S")
                    except ValueError:
                    # If conversion fails, default to the original value
                        time_24h = time_12h

                    selected = [row[32], time_24h, parsed_date]
                    csv_writer.writerow(selected)

    outfile.close()

    # input_file = 'BEMS-15_3_2023.txt'
    print(count)

# csvmaker("Meter-26", "output26.csv")
# csvmaker("Meter-27", "output27.csv")

df26 = pd.read_csv('output26.csv')
df27 = pd.read_csv('output27.csv')

df26['DateTime'] = pd.to_datetime(
    df26['Date'] + ' ' + df26['Time'], 
    format='%d/%m/%Y %H:%M:%S'
)

df = pd.DataFrame({
    'Global_active_power': df26['Global_active_power'] + df27['Global_active_power'],
    'DateTime': df26['DateTime']
})

df.to_csv('output.csv', index=False)

print(df)
        

# folder_path = "Meter-27"
# csv_files = glob.glob(os.path.join(folder_path, '*.txt'))
# csv_files.sort(key=os.path.getmtime, reverse=False) # sort files by modification time, descending

# count = 0
# output_file = 'output27.csv'

# if os.path.isfile(output_file):
#     os.remove(output_file)

# outfile = open(output_file, 'w', newline='')

# csv_writer = csv.writer(outfile)
# # Write header
# csv_writer.writerow(["Global_active_power", "Time", "Date"])

# for file in csv_files:
#     count += 1
#     with open(file, 'r') as infile:
#                 # Extract the date from the filename
#         # Assuming the filename format is "BEMS-<date>.txt", where <date> is in the form "1_1_2023"
#         basename = os.path.basename(file)
#         name_no_ext = os.path.splitext(basename)[0]  # "BEMS-1_1_2023"
#         parts = name_no_ext.split('-')
#         date_part = parts[1]  # "1_1_2023"

#         # Convert "1_1_2023" → "2023-01-01"
#         parsed_date = datetime.strptime(date_part, "%d_%m_%Y").strftime("%d/%m/%Y")
        
#         for line in infile:
#             row = line.strip().split('\t')
#             # Ensure there are at least 64 columns
#             if len(row) == 64:
#                 # Python indexing: column AG (33rd) is index 32, column BL (64th) is index 63
#                 time_12h = row[63]
#                 try:
#                     # Convert 12-hour time to 24-hour format
#                     time_24h = datetime.strptime(time_12h, "%I:%M:%S %p").strftime("%H:%M:%S")
#                 except ValueError:
#                 # If conversion fails, default to the original value
#                     time_24h = time_12h

#                 selected = [row[32], time_24h, parsed_date]
#                 csv_writer.writerow(selected)

# outfile.close()

# # input_file = 'BEMS-15_3_2023.txt'
# print(count)



