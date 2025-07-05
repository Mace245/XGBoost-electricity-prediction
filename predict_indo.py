import pandas as pd
import xgboost as xgb
import pickle as pkl # For loading the scaler

import openmeteo_requests
import requests_cache
from retry_requests import retry
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pylab as plt
from scipy.interpolate import interp1d

def prepare_data(electricity_data:pd.DataFrame, temperature_data:pd.DataFrame):
    merged_data = electricity_data.merge(temperature_data, how='left', left_index=True, right_index=True)
    print(merged_data)
    merged_data = merged_data.ffill().dropna()
    return merged_data

def temp_fetch(start_date, end_date, latitude:float, longitude:float, historical:bool):
	cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
	retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
	openmeteo = openmeteo_requests.Client(session = retry_session)

	if historical == True:
		url = "https://archive-api.open-meteo.com/v1/archive"
	else:
		url = "https://api.open-meteo.com/v1/forecast"
	params = {
		"latitude": latitude,
		"longitude": longitude,
		"hourly": "temperature_2m",
		"timezone": "GMT+8",
		"start_date": start_date,
		"end_date": end_date
	}
	responses = openmeteo.weather_api(url, params=params)
	response = responses[0]

	hourly = response.Hourly()
	hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()

	hourly_data = {"DateTime": pd.date_range(
		start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
		end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
		freq = pd.Timedelta(seconds = hourly.Interval()),
		inclusive = "left"
	)}

	hourly_data["temperature"] = hourly_temperature_2m

	hourly_dataframe = pd.DataFrame(data = hourly_data)
	hourly_dataframe.set_index('DateTime', inplace=True)
	print(f"Start Date: {start_date}")
	print(f"End Date: {end_date}")
	return hourly_dataframe

def fetch_elec_temp():
    electricity_data = pd.read_csv('test.csv', parse_dates=['DateTime'])
    electricity_data.set_index('DateTime', inplace=True)
    # electricity_data = electricity_data.drop(columns='Global_active_power')
    
    start_date = electricity_data.index.min().strftime('%Y-%m-%d')
    end_date = electricity_data.index.max().strftime('%Y-%m-%d')
    electricity_data = electricity_data.tz_localize('Asia/Jakarta')
    print(start_date, end_date)
 
    latitude = -6.97280651149127
    longitude = 107.63017470203512

    temperature_data = temp_fetch(
        start_date=start_date,
        end_date=end_date,
        latitude=latitude,
        longitude=longitude,
        historical=True
    )
    
    return electricity_data, temperature_data

print("Loading model and scaler...")
# Load the model that was trained on scaled data
loaded_model = xgb.XGBRegressor()
loaded_model.load_model('final_xgboost_model.ubj')

# Load the scaler that was fitted on the Malaysian training features
file = open('scaler_x.pkl', 'rb')
scaler_X = pkl.load(file)
file.close()

def create_features(df):
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df['temperature'] = df['temperature']
    # df['wh_lag_24h'] = df['Wh'].shift(24)
    # df['wh_lag_72h'] = df['Wh'].shift(72)
    # df['wh_lag_168h'] = df['Wh'].shift(168)

    return df

electricity_raw, temperature_raw = fetch_elec_temp()
merged_data_complete = prepare_data(electricity_raw, temperature_raw)
merged_data_complete = merged_data_complete.tz_convert('Asia/Jakarta')

# min_actual_wh = merged_data_complete['Wh'].min()
# max_actual_wh = merged_data_complete['Wh'].max()
# print(min_actual_wh, max_actual_wh)

X_test = create_features(merged_data_complete)
X_test = X_test.drop(columns=['Wh', 'date'])
X_test['temperature'] = X_test.pop('temperature')

features =  ['hour','dayofweek','quarter','month','year',
            'dayofyear','dayofmonth','weekofyear', 'temperature',
            # 'wh_lag_24h', 'wh_lag_72h', 'wh_lag_168h'
            ]

X_predict = X_test[features]

X_indonesia_scaled = scaler_X.transform(X_test)

predictions_normalized = loaded_model.predict(X_predict)

# First, create a results dataframe
results = pd.DataFrame({
    'DateTime': merged_data_complete.index,
    'Actual_Wh': merged_data_complete['Wh'],
    'Prediction_Normalized': predictions_normalized
}).set_index('DateTime')
print(results)

m = interp1d([1.91070775, 617.1704890666666],[60.0, 151.6])
results['Prediction_Normalized'] = m(results['Prediction_Normalized'])

# --- Step 3: Display the Resulting Table ---

print("--- Predictions Mapped to Indonesian Data Scale ---")
# We select the columns to show the clear before-and-after mapping
print(results[['Actual_Wh', 'Prediction_Normalized']])


# --- Step 4 (Optional): Plot the Mapped Data ---
results[['Actual_Wh', 'Prediction_Normalized']].plot(figsize=(15,6), style=['-', '--'])
plt.title('Comparing Actual vs. Mapped Predicted Usage')
plt.ylabel('Usage (Wh)')
plt.legend()
plt.grid(True)
plt.show()

# The 'Predicted_Normalized_Usage' is your final result for the demo.
# It represents the predicted usage pattern on a 0-1 scale, relative to the Malaysian data.