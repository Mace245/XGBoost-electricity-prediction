from antares_http import antares
import pandas as pd
from datetime import datetime
import os
import time
import random

antares.setDebug(False)
antares.setAccessKey('5cd4cda046471a89:75f9e1c6b34bf41a')

latestData = antares.get('UjiCoba_TA', 'TA_DKT1')
print(latestData, "\n")
for key, value in latestData['content'].items():
    print(f'{key}: {value}')

# data = {
#     'DateTime': [datetime.strptime(latestData['last_modified_time'], '%Y%m%dT%H%M%S')],
#     'Voltage': [latestData['content']['Voltage']],
#     'Current': [latestData['content']['Current']],
#     'Power': [latestData['content']['Power']],
#     'Energy': [latestData['content']['Energy']],
#     'TotalEnergy': [latestData['content']['TotalEnergy']],
#     'DailyEnergy': [latestData['content']['DailyEnergy']],
#     'limitEnergy': [latestData['content']['limitEnergy']]
# }

# df = pd.DataFrame(data)
# df.set_index('DateTime', inplace=True)

# if not os.path.isfile('log.csv'):
#     df.to_csv('log.csv', mode='w', header=True)
# else:
#     df.to_csv('log.csv', mode='a', header=False)

# print(df)

# antares.setDebug(True)

# myData = {'Voltage': 217.6999969, 'Current': 0, 'Power': 0, 'Energy': 0.542999983, 'TotalEnergy': 58.6439476, 'DailyEnergy': 58.6439476, 'limitEnergy': 0}

SEND_INTERVAL_SECONDS = 15

try:
    # 1. Generate Dummy Data
    # Adjust ranges as needed to be realistic for your device
    dummy_voltage = round(random.uniform(215.0, 235.0), 2)
    dummy_current = round(random.uniform(0.1, 5.0), 3)
    dummy_power = round(dummy_voltage * dummy_current * 0.9, 2) # Approximate Power (with PF guess)
    dummy_energy = round(dummy_power * (SEND_INTERVAL_SECONDS / 3600.0), 4) # Wh in this interval
    # These values might need more complex logic in reality
    dummy_total_energy = round(random.uniform(1000.0, 5000.0), 2)
    dummy_daily_energy = round(random.uniform(50.0, 1500.0), 2)
    dummy_limit_energy = 2.0 # Example fixed limit

    data_to_send = {
        'Voltage': dummy_voltage,
        'Current': dummy_current,
        'Power': dummy_power,
        'Energy': dummy_energy,
        'TotalEnergy': dummy_total_energy,
        'DailyEnergy': dummy_daily_energy,
        'limitEnergy': dummy_limit_energy
    }

    # 2. Send Data to Antares
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Sending data: {data_to_send}")
    response = antares.send(data_to_send, 'UjiCoba_TA', 'TA_DKT1')

    # 3. Print Response (optional)
    # Antares library might print internally or return status/response content
    print(f"Antares response: {response}") # Check library's return value

except Exception as e:
    print(f"An error occurred: {e}")

# 4. Wait for the next interval
try:
    time.sleep(SEND_INTERVAL_SECONDS)
except KeyboardInterrupt:
    print("\nStopping sender.")