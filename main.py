import data, algo

from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pickle
import numpy as np
import pandas as pd

# Load data
electricity, temperature = data.fetch_elec_temp()
merged_data = data.prepare_data(electricity, temperature)

# Feature engineering pipeline
merged_data = algo.create_time_features(merged_data)
merged_data = algo.create_lagged_features(merged_data)
merged_data = algo.add_seasonal_components(merged_data)

# Define features
features = [
    'hour', 'day_of_week', 'day_of_month', 'is_weekend',

    'lag_1', 'lag_24', 'lag_168',
    'temperature'
]
target = 'Global_active_power'

X = merged_data[features]
y = merged_data[target]

# Train model
model = algo.train_xgboost_model(X, y)

# Save artifacts
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
merged_data.to_csv('processed_data.csv', index=True)

forecast_period = pd.Timedelta(days=365*2)

# Generate forecast
last_known_data = X.loc[X.index > X.index.max() - forecast_period]  # Last week of data
forecast = algo.recursive_forecast(model, last_known_data)

mse = mean_squared_error(y.loc[y.index > y.index.max() - forecast_period], forecast)
mae = mean_absolute_error(y.loc[y.index > y.index.max() - forecast_period], forecast)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y.loc[y.index > y.index.max() - forecast_period] - forecast) / y.loc[y.index > y.index.max() - forecast_period])) * 100

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Percentage Error: {mape:.2f}%")

# Visualize
# plt.figure(figsize=(12, 6))
plt.plot(y.loc[y.index > y.index.max() - forecast_period], label='Historical')
plt.plot(forecast, label='Forecast', linestyle='--')
plt.title(forecast_period)
plt.legend()
plt.show()

# get importance
importance = model.feature_importances_
plt.bar(range(len(importance)), importance, align='center')
plt.xticks(ticks=range(len(features)), labels=features, rotation=45)
plt.ylabel("Importance Score")
plt.title("Feature Importance")
plt.show()