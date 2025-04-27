import algo, data
import pickle
import pandas as pd

LATITUDE = 14.5833
LONGITUDE = 121.0

# Define features
features = [
    'hour', 'day_of_week', 'day_of_month', 'is_weekend',
    'lag_1','lag_24','lag_168',
    'temperature'
]
target = 'Wh'

def train(predict:bool, forecast_period_days:int):
    training_data = pd.read_csv('training_data.csv', parse_dates=['DateTime'])
    training_data.set_index('DateTime', inplace=True)
    temp_col = training_data.pop('temperature')
    training_data.insert(8, 'temperature', temp_col)

    target_data=training_data.drop(columns='Wh')
    feature_data=training_data.loc[:, 'Wh']

    # Train model
    model = algo.train_xgboost_model(target_data, feature_data)

    # Save artifacts
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    if predict:
        forecast_period = pd.Timedelta(days=forecast_period_days)

        data.temp_fetch('2023-03-08','2023-03-14', LATITUDE, LONGITUDE, historical=True).to_csv("test.csv")
        # Generate forecast
        last_known_data = target_data.loc[target_data.index > target_data.index.max() - forecast_period]  # Last week of data
        forecast = algo.predict_on_window_recursive(
            model=model, 
            history_df= training_data, 
            future_temps_series=data.temp_fetch('2023-03-07','2023-03-14', LATITUDE, LONGITUDE, historical=True), hours_to_forecast=forecast_period_days*24, model_features=features, target_variable=target, max_lag=168)
        print(last_known_data)

        forecast.to_csv('forecast.csv')

        data.visualize(model, features, feature_data, forecast, forecast_period)

# data.create_training_data(features,target)
train(True, 7)