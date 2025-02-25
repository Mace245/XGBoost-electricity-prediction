import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt

def prepare_data(electricity_data, temperature_data):
    # Resample electricity data to hourly
    electricity_hourly = electricity_data.resample('h').mean()
    electricity_hourly = electricity_hourly.head(1536) # temp for invalid merge
    # print(electricity_hourly.info(), temperature_data.info())
    
    # Merge electricity and temperature data
    merged_data = electricity_hourly
    merged_data['temperature'] = temperature_data['temperature'].values
    # print(merged_data)
    
    return merged_data

def create_time_features(electricity_data):
    electricity_data['hour'] = electricity_data.index.hour
    electricity_data['day_of_week'] = electricity_data.index.dayofweek
    electricity_data['month'] = electricity_data.index.month
    electricity_data['is_weekend'] = electricity_data.index.dayofweek.isin([5, 6]).astype(int)
    return electricity_data

def create_lagged_features(electricity_data, electricity_column, lag_hours=24*7):
    # Create all lagged features at once using concat
    lags = [electricity_data[electricity_column].shift(lag)
            for lag in range(1, lag_hours + 1)]
    lagged_df = pd.concat(lags, axis=1)
    lagged_df.columns = [f'electricity_lag_{lag}'
                        for lag in range(1, lag_hours + 1)]
    electricity_data = pd.concat([electricity_data, lagged_df], axis=1)
    return electricity_data.dropna()

def train_xgboost_model(X_train, y_train):
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    return model

def predict_electricity_usage(model, X_test,Y_test):
    predictions = model.predict(X_test)
    print("X_test columns:", type(X_test))
    predictions = pd.Series(predictions, index=Y_test.index)  # Add datetime index
    return predictions

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title('Electricity Usage Prediction')
    plt.xlabel('Time')
    plt.ylabel('Electricity Usage')
    plt.legend()
    print(y_pred, y_true)
    print(type(y_pred), type(y_true))
    plt.show()

def main():
    # Load Data
    electricity_data = pd.read_csv('household_power_consumption.csv')

    electricity_data['DateTime'] = pd.to_datetime(electricity_data['Date'] + ' ' + electricity_data['Time'], format='%d/%m/%Y %H:%M:%S')
    electricity_data = electricity_data.set_index(electricity_data['DateTime']).drop(columns=['DateTime','Date', 'Time', 'Voltage', 'Global_reactive_power', 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'])
    electricity_data['Global_active_power'] = pd.to_numeric(electricity_data['Global_active_power'], errors='coerce')

    temperature_data = pd.read_csv('open-meteo-unix 20nov-22jan.csv')
    temperature_data['time'] = pd.to_datetime(temperature_data['time'], unit ='s')
    temperature_data = temperature_data.set_index(temperature_data['time']).drop(columns=['time'])
    temperature_data['temperature'] = pd.to_numeric(temperature_data['temperature'], errors='coerce')

    # print(electricity_data)
    # print(temperature_data.index.hour)
    
    # Prepare Data
    merged_data = prepare_data(electricity_data, temperature_data)
    
    # Feature Engineering
    merged_data = create_time_features(merged_data)
    merged_data = create_lagged_features(merged_data, 'Global_active_power')
    
    # Separate Features and Target
    features = ['temperature', 'hour', 'day_of_week', 'month', 'is_weekend'] + \
               [col for col in merged_data.columns if 'electricity_lag' in col]
    
    X = merged_data[features]
    y = merged_data['Global_active_power']
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    # print("X_test columns:", type(X_test))
    # print(X_test.head())
    # print("X_train columns:", type(X_train))
    # print("y_test columns:", type(y_test))
    # print("y_train columns:", type(y_train))
    
    # Train Model
    model = train_xgboost_model(X_train, y_train)
    
    # Predict
    predictions = predict_electricity_usage(model, X_test, y_test)
    
    # Evaluate
    evaluate_model(y_test, predictions)
    
    # Visualize
    plot_predictions(y_test, predictions)

if __name__ == "__main__":
    main()