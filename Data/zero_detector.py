import pandas as pd
import matplotlib.pylab as plt

pjme = pd.read_csv('Data/minute_data.csv', index_col=[0], parse_dates=[0])

condition = pjme['Global_active_power'] == 0

zero_rows = pjme[condition]

zero_dates = zero_rows.index

# print(zero_dates)
print(zero_rows)

zero_rows.to_csv('zeroes.csv')