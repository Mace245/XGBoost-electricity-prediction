import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pylab as plt
import warnings
import openmeteo_requests
import requests_cache
from retry_requests import retry

# --- 0. SETUP ---
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

# --- 1. DATA FETCHING AND PREPARATION FUNCTIONS ---

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

def fetch_elec_temp(filepath):
    electricity_data = pd.read_csv(filepath, parse_dates=['DateTime'])
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

electricity_raw, temperature_raw = fetch_elec_temp('test.csv')
merged_data_complete = prepare_data(electricity_raw, temperature_raw)
test_df = merged_data_complete.tz_convert('Asia/Jakarta')

electricity_raw, temperature_raw = fetch_elec_temp('test2.csv')
merged_data_complete = prepare_data(electricity_raw, temperature_raw)
test2_df = merged_data_complete.tz_convert('Asia/Jakarta')

# --- 2. FEATURE ENGINEERING FUNCTION ---
def create_features(df, label=None):
    """Creates time series features from a datetime index."""
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    
    X = df[['hour','dayofweek','quarter','month','year',
            'dayofyear','dayofmonth','weekofyear','temperature']]
    
    if label and label in df.columns:
        y = df[label]
        return X, y
    return X

# --- 3. LOAD DATA AND CREATE FEATURES ---
print("Loading and preparing data...")
# Load initial training data for Building A
from lib import data

electricity_raw, temperature_raw = data.fetch_elec_temp()
building_a_df = data.prepare_data(electricity_raw, temperature_raw)
# Load and prepare the data for Building B
print("\nLoading and splitting data for Building B from 'test.csv'...")
electricity_raw_b, temperature_raw_b = fetch_elec_temp('test.csv')
full_building_b_df = prepare_data(electricity_raw_b, temperature_raw_b)
full_building_b_df = full_building_b_df.tz_convert('Asia/Jakarta')

# Get the unique dates and split them
unique_dates = full_building_b_df.index.normalize().unique()
train_dates = unique_dates[:4]
eval_date = unique_dates[4]

# Create the training and evaluation sets for Building B
building_b_df = full_building_b_df[full_building_b_df.index.normalize().isin(train_dates)]
building_b_eval_df = full_building_b_df[full_building_b_df.index.normalize() == eval_date]

print(f"Building B Training Data: {len(building_b_df)} rows from {train_dates.min().date()} to {train_dates.max().date()}")
print(f"Building B Evaluation Data: {len(building_b_eval_df)} rows for {eval_date.date()}")

# Create feature sets for each building
X_train_a, y_train_a = create_features(building_a_df, label='Wh')
X_train_b, y_train_b = create_features(building_b_df, label='Wh')
X_eval_b, y_eval_b = create_features(building_b_eval_df, label='Wh')

# --- 4. SCALE DATA ---
# Initialize and fit the scaler ONLY on the data from Building A
scaler = MinMaxScaler()
print("\nFitting scaler on Building A's data...")
scaler.fit(X_train_a)

# Transform both datasets using the SAME scaler learned from Building A
print("Transforming data for both buildings...")
X_train_a_scaled = scaler.transform(X_train_a)
X_train_b_scaled = scaler.transform(X_train_b)
X_eval_b_scaled = scaler.transform(X_eval_b) 

# --- 5. STAGE 1: TRAIN THE BASE MODEL ON BUILDING A ---
base_model = xgb.XGBRegressor(
    objective='reg:squarederror', n_estimators=1000, learning_rate=0.01,
    max_depth=7, colsample_bytree=1.0, subsample=0.1,
    reg_alpha=5.0, reg_lambda=10.0, n_jobs=-1
)

print("\n--- Stage 1: Training Base Model on Building A ---")
base_model.fit(X_train_a_scaled, y_train_a, verbose=100)

# Save the base model
base_model_path = 'building_a_base_model.ubj'
print(f"\nSaving the base model to {base_model_path}...")
base_model.save_model(base_model_path)

# --- 6. STAGE 2: FINE-TUNE A NEW MODEL FOR BUILDING B ---
print("\n--- Stage 2: Fine-tuning a new, specialized model for Building B ---")
# Create a new model instance for Building B
building_b_model = xgb.XGBRegressor()
# Load the state of the base model as a starting point
building_b_model.load_model(base_model_path) 

# Continue training (fine-tuning) on Building B's data
building_b_model.fit(
    X_train_b_scaled, y_train_b,
    verbose=100
)

# Save the final, specialized model for Building B
final_model_path = 'building_b_specialized_model.ubj'
print(f"\nSaving the specialized Building B model to {final_model_path}...")
building_b_model.save_model(final_model_path)

# --- 7. EVALUATE AND VISUALIZE THE SPECIALIZED BUILDING B MODEL ---
print("\nLoading final model and making predictions on Building B data...")
final_model = xgb.XGBRegressor()
final_model.load_model(final_model_path)

predictions = final_model.predict(X_eval_b_scaled)

results_df = pd.DataFrame({'Actual': y_eval_b, 'Predicted': predictions}, index=y_eval_b.index)
# print("\nFirst 5 predictions from the Building B model:")
# print(results_df.head())

mape_score = mean_absolute_percentage_error(y_eval_b, predictions)
print(f"\nFinal Model MAPE on Building B Data: {mape_score * 100:.3f}%")

# Plotting
xgb.plot_importance(final_model, height=0.9, importance_type='gain')
plt.title("Feature Importance (Specialized Building B Model)")
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.grid(False)
plt.savefig('indo_feature_importance.png')
plt.show()

plt.figure(figsize=(15, 6))
results_df['Actual'].plot(label='Aktual', style='-')
results_df['Predicted'].plot(label='Prediksi', style='--')
plt.title('Building B Model: Actual vs. Predicted')
plt.xlabel('Date')
plt.ylabel('Wh')
plt.legend()
plt.grid(True)
plt.savefig('indo_actual_vs_predicted.png')
plt.show()
