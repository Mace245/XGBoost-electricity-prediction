import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
# import shap
# import argparse
import pickle
import numpy as np
from scipy.stats import zscore
from statsmodels.tsa.seasonal import seasonal_decompose

# ----------------------
# Data Loading & Preprocessing
# ----------------------
def fetch_elec_temp():
    electricity_data = pd.read_csv('household_power_consumption.csv')
    
    # Convert datetime and handle NaNs
    electricity_data['DateTime'] = pd.to_datetime(
        electricity_data['Date'] + ' ' + electricity_data['Time'], 
        format='%d/%m/%Y %H:%M:%S'
    )
    electricity_data = electricity_data.set_index('DateTime')
    electricity_data = electricity_data[['Global_active_power']]
    electricity_data['Global_active_power'] = pd.to_numeric(
        electricity_data['Global_active_power'], errors='coerce'
    )
    
    # Load temperature data
    temperature_data = pd.read_csv('open-meteo-unix 20nov-22jan.csv')
    temperature_data['time'] = pd.to_datetime(temperature_data['time'], unit='s')
    temperature_data = temperature_data.set_index('time')[['temperature']]
    
    return electricity_data, temperature_data

def handle_outliers(data, column='Global_active_power', threshold=3):
    """Remove outliers using Z-score"""
    z_scores = zscore(data[column])
    return data[(np.abs(z_scores) < threshold)]

def prepare_data(electricity_data, temperature_data):
    # Resample electricity data to hourly
    electricity_hourly = electricity_data.resample('h').mean()
    electricity_hourly = electricity_hourly.head(1536) # temp for invalid merge
    # print(electricity_hourly.info(), temperature_data.info())
    
    # Merge electricity and temperature data
    merged_data = electricity_hourly
    merged_data['temperature'] = temperature_data['temperature'].values
    # print(merged_data)
    
    # Handle missing values and outliers
    merged_data = merged_data.ffill().dropna()
    merged_data = handle_outliers(merged_data)
    
    return merged_data

# ----------------------
# Feature Engineering
# ----------------------
def create_time_features(data):
    """Add temporal features"""
    data = data.copy()
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek  # 0=Monday
    data['day_of_month'] = data.index.day
    data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)
    return data

def create_lagged_features(data, target_col='Global_active_power', lags=[1, 24, 168]):
    """Add lagged features (1h, 24h, 168h=1 week)"""
    for lag in lags:
        data[f'lag_{lag}'] = data[target_col].shift(lag)
    return data.dropna()

def add_seasonal_components(data, period=24):
    """Add seasonal decomposition features (daily cycle)"""
    decomposition = seasonal_decompose(
        data['Global_active_power'], 
        model='additive', 
        period=period
    )
    data['trend'] = decomposition.trend
    data['seasonal'] = decomposition.seasonal
    data['residual'] = decomposition.resid
    return data.dropna()

# ----------------------
# Modeling
# ----------------------
def train_xgboost_model(X_train, y_train):
    """Train with paper-inspired parameters"""
    model = xgb.XGBRegressor(
        max_depth=6,
        learning_rate=0.01,
        n_estimators=1000,
        colsample_bytree=0.7,
        subsample=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric='mape'
    )
    
    # Time-series cross-validation
    tss = TimeSeriesSplit(n_splits=3)
    scores = []
    for train_idx, val_idx in tss.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=True
        )
        scores.append(model.best_score)
    
    print(f"Avg Validation Score: {np.mean(scores):.4f}")
    return model

# ----------------------
# Feature Engineering
# ----------------------
def create_lagged_features(data, target_col='Global_active_power'):
    """Create only necessary lags: 1h, 24h, 168h"""
    lags = [1, 24, 168]  # 1h, 24h, 168h
    for lag in lags:
        data[f'lag_{lag}'] = data[target_col].shift(lag)
    return data.dropna()

# ----------------------
# Forecasting
# ----------------------
def recursive_forecast(model, last_observed_window):
    predictions = model.predict(last_observed_window)
    print("X_test columns:", type(last_observed_window))
    predictions = pd.Series(predictions, index=last_observed_window.index)  # Add datetime index
    return predictions

# ----------------------
# Main Execution
# ----------------------
if __name__ == "__main__":
    # Load data
    electricity, temperature = fetch_elec_temp()
    merged_data = prepare_data(electricity, temperature)
    
    # Feature engineering pipeline
    merged_data = create_time_features(merged_data)
    merged_data = create_lagged_features(merged_data)
    merged_data = add_seasonal_components(merged_data)
    
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
    model = train_xgboost_model(X, y)
    
    # Save artifacts
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    merged_data.to_csv('processed_data.csv', index=True)
    
    # Generate 1-week forecast
    last_known_data = X.iloc[-168:]  # Last week of data
    forecast = recursive_forecast(model, last_known_data)
    
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