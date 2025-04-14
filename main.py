from lib import algo, data
import pickle
import numpy as np
import pandas as pd

# Define features
features = [
    'hour', 'day_of_week', 'day_of_month', 'is_weekend',

    'lag_1', 'lag_24', 'lag_168',
    'temperature'
]
target = 'Wh'

training_data = data.create_training_data(features,target)
# Train model
model = algo.train_xgboost_model(training_data.X, training_data.y)

# Save artifacts
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

forecast_period = pd.Timedelta(days=365*2)

# Generate forecast
last_known_data = training_data.X.loc[training_data.X.index > training_data.X.index.max() - forecast_period]  # Last week of data
forecast = algo.predict_on_window(model, last_known_data)

data.visualize(model, features, training_data.y, forecast, forecast_period)