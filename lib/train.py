import algo, data
import pickle
import pandas as pd

# Define features
features = [
    'hour', 'day_of_week', 'day_of_month', 'is_weekend',
    'lag_1', 'lag_24', 'lag_168',
    'temperature'
]
target = 'Wh'


def train(data_db, forecast_period_days:int):
    cutoff_date = data_db.X.index.max() - pd.Timedelta(days=forecast_period_days)
    # Filter X to keep only the most recent data_db
    X_filtered = data_db.X[data_db.X.index >= cutoff_date]
    
    # Align y with the filtered X (same index)
    y_filtered = data_db.y.loc[X_filtered.index]
    # Train model
    model = algo.train_xgboost_model(X_filtered, y_filtered)

    # Save artifacts
    model_filename = f'models/model_{forecast_period_days}.pkl'
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)

def predict(forecast_period_days:int, model, data_db, eval: bool):
    forecast_period = pd.Timedelta(days=forecast_period_days)

    # Generate forecast
    last_known_data = data_db.X.loc[data_db.X.index > data_db.X.index.max() - forecast_period]  # Last week of data
    # last_known_data = pd.DataFrame(index=last_known_data.index)
    forecast = algo.predict_on_window(model, last_known_data)

    # print(last_known_data)x
    
    if eval:
        data.visualize(model, features, data_db.y, forecast, forecast_period)
        print(last_known_data)


db_data = data.create_training_data(features,target) 

days = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
for i in days:
    forecast_days = i
    print('days: ', i)
    train(db_data, forecast_days)

    model_filename = f'models/model_{forecast_days}.pkl'
    model_file = open(model_filename, 'rb')
    model = pickle.load(model_file)
    predict(forecast_days, model, db_data, False)