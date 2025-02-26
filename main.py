import data, algo

from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pickle
import numpy as np

# Load data
electricity, temperature = data.fetch_elec_temp()
merged_data = data.prepare_data(electricity, temperature)

# Feature engineering pipeline
merged_data = algo.create_time_features(merged_data)
merged_data = algo.create_lagged_features(merged_data)
merged_data = algo.add_seasonal_components(merged_data)

# Define features
features = [
    'temperature', 'hour', 'day_of_week', 'day_of_month', 'is_weekend',
    'trend', 'seasonal', 'residual', 
    'lag_1', 'lag_24', 'lag_168'
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

# Generate 1-week forecast
last_known_data = X.iloc[-168:]  # Last week of data
forecast = algo.recursive_forecast(model, last_known_data)

# Visualize
plt.figure(figsize=(12, 6))
plt.plot(y[-168:], label='Historical')
plt.plot(forecast, label='Forecast', linestyle='--')
plt.title('1-Week Load Forecast')
plt.legend()
plt.show()

mse = mean_squared_error(y[-168:], forecast)
mae = mean_absolute_error(y[-168:], forecast)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y[-168:] - forecast) / y[-168:])) * 100

print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Percentage Error: {mape:.2f}%")

# get importance
importance = model.feature_importances_
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()