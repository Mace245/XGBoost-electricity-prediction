from antares_http import antares
import pandas as pd
from datetime import datetime
import os

antares.setDebug(False)
antares.setAccessKey('5cd4cda046471a89:75f9e1c6b34bf41a')

latestData = antares.get('UjiCoba_TA', 'TA_DKT1')
print(latestData, "\n")
for key, value in latestData['content'].items():
    print(f'{key}: {value}')

data = {
    'DateTime': [datetime.strptime(latestData['last_modified_time'], '%Y%m%dT%H%M%S')],
    'Voltage': [latestData['content']['Voltage']],
    'Current': [latestData['content']['Current']],
    'Power': [latestData['content']['Power']],
    'Energy': [latestData['content']['Energy']],
    'TotalEnergy': [latestData['content']['TotalEnergy']],
    'DailyEnergy': [latestData['content']['DailyEnergy']],
    'limitEnergy': [latestData['content']['limitEnergy']]
}

df = pd.DataFrame(data)
df.set_index('DateTime', inplace=True)

if not os.path.isfile('log.csv'):
    df.to_csv('log.csv', mode='w', header=True)
else:
    df.to_csv('log.csv', mode='a', header=False)

print(df)

# antares.setDebug(True)

# myData = {'Voltage': 217.6999969, 'Current': 0, 'Power': 0, 'Energy': 0.542999983, 'TotalEnergy': 58.6439476, 'DailyEnergy': 58.6439476, 'limitEnergy': 0}

# antares.send(myData, 'UjiCoba_TA', 'TA_DKT1')