import pandas as pd

import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
electricity_data = pd.read_csv('output.csv')

electricity_data = electricity_data.set_index('DateTime')
electricity_data = electricity_data[['Global_active_power']]

print(electricity_data)

# Specify your reference date
specified_date = pd.to_datetime('2021-10-5')

# To get rows 7 days after the specified date (inclusive)
data = electricity_data.loc[specified_date: specified_date + pd.Timedelta(days=7)]
print(data)

# Plot the data
# plt.figure(figsize=(10, 5))
plt.plot(data, linestyle='-')
plt.xlabel('Date and Hour')
plt.ylabel('Data')
plt.title('Data Visualization')
plt.tight_layout()
plt.show()