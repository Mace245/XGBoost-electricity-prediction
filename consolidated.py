import pandas as pd
import pickle
import time
from datetime import datetime, timedelta
import algo
import temp_api
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb

# Load the trained model
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        print("Model not found. Please run main.py first to train the model.")
        return None
        
# Load historical data
def load_historical_data():
    try:
        # Try to load pre-processed data first
        data = pd.read_csv('processed_data.csv', parse_dates=['DateTime'])
        data.set_index('DateTime', inplace=True)
        print("Loaded processed data.")
    except FileNotFoundError:
        # If not available, load raw data
        try:
            data = pd.read_csv('Data/processed_hourly_Wh_data.csv', parse_dates=['DateTime'])
            data.set_index('DateTime', inplace=True)
            print("Loaded raw hourly data.")
        except FileNotFoundError:
            print("No data files found. Please ensure data is available.")
            return None
    return data

def simulate_realtime_prediction(data, model, interval_minutes=5, prediction_horizon=24, 
                                 simulation_days=7):
    """
    Simulate real-time prediction using historical data
    
    Parameters:
    - data: DataFrame with historical electricity data
    - model: Trained XGBoost model
    - interval_minutes: How often to make predictions (in minutes)
    - prediction_horizon: How many hours to predict ahead
    - simulation_days: How many days to simulate
    """
    # Define simulation period
    start_time = data.index.min()
    end_time = start_time + timedelta(days=simulation_days)
    
    if end_time > data.index.max():
        end_time = data.index.max() - timedelta(hours=prediction_horizon)
        print(f"Adjusted simulation end time to {end_time} due to data limitations")
    
    current_time = start_time
    
    # Store predictions for evaluation
    all_predictions = {}
    actual_values = {}
    
    print(f"Starting simulation from {start_time} to {end_time}")
    print(f"Making predictions every {interval_minutes} minutes for the next {prediction_horizon} hours")
    
    # Simulation loop
    while current_time <= end_time:
        # Get data available up to the current time
        available_data = data[data.index <= current_time].copy()
        
        # Skip if not enough data for lag features
        if len(available_data) < 168:  # Need at least a week of data for weekly lag
            print(f"Not enough historical data at {current_time}. Skipping.")
            current_time += timedelta(minutes=interval_minutes)
            continue
            
        # Prepare features
        features_data = algo.create_time_features(available_data)
        features_data = algo.create_lagged_features(features_data)
        
        # Get temperature data for the prediction period
        pred_start = current_time
        pred_end = current_time + timedelta(hours=prediction_horizon)
        
        # Ensure all required features are available
        required_features = ['hour', 'day_of_week', 'day_of_month', 'is_weekend',
                             'lag_1', 'lag_24', 'lag_168', 'temperature']
        
        # For simulation, use the known temperature values from historical data
        # In a real scenario, you would use forecasted temperatures
        
        # Make prediction for the next prediction_horizon hours
        feature_set = features_data.tail(1)
        
        # Generate forecast using recursive method
        forecast = pd.Series(index=pd.date_range(start=current_time, 
                                               periods=prediction_horizon, 
                                               freq='H'))
        
        latest_row = feature_set.iloc[0].copy()
        
        # Recursive prediction
        for i in range(prediction_horizon):
            # Calculate the timestamp for this step
            step_time = current_time + timedelta(hours=i)
            
            # Update time features
            latest_row['hour'] = step_time.hour
            latest_row['day_of_week'] = step_time.dayofweek
            latest_row['day_of_month'] = step_time.day
            latest_row['is_weekend'] = int(step_time.dayofweek >= 5)
            
            # If temperature data is available, update it
            if 'temperature' in latest_row:
                if step_time in data.index:
                    latest_row['temperature'] = data.loc[step_time, 'temperature']
            
            # Make prediction for this hour
            pred_features = pd.DataFrame([latest_row])
            prediction = model.predict(pred_features)[0]
            
            # Store prediction
            forecast[step_time] = prediction
            
            # Update lag features for next prediction
            if i < prediction_horizon - 1:
                latest_row['lag_1'] = prediction
                if i >= 24:
                    latest_row['lag_24'] = forecast[step_time - timedelta(hours=24)]
                if i >= 168:
                    latest_row['lag_168'] = forecast[step_time - timedelta(hours=168)]
        
        # Store this prediction set
        all_predictions[current_time] = forecast
        
        # Get actual values for the predicted period
        actuals = data.loc[forecast.index, 'Global_active_power']
        actual_values[current_time] = actuals
        
        # Print some information
        print(f"\nPrediction at {current_time}:")
        print(f"Next hour prediction: {forecast[0]:.4f} kW")
        print(f"Actual value: {actuals.iloc[0] if len(actuals) > 0 else 'Unknown'}")
        
        # Move to next interval
        current_time += timedelta(minutes=interval_minutes)
    
    return all_predictions, actual_values

def evaluate_simulation(predictions, actuals):
    """Evaluate the accuracy of the simulated predictions"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import numpy as np
    
    # Calculate metrics for each prediction time
    metrics = {}
    for pred_time, forecast in predictions.items():
        actual = actuals[pred_time]
        
        # Calculate metrics only for overlapping time periods
        common_idx = forecast.index.intersection(actual.index)
        if len(common_idx) == 0:
            continue
            
        pred_values = forecast[common_idx]
        actual_values = actual[common_idx]
        
        mse = mean_squared_error(actual_values, pred_values)
        mae = mean_absolute_error(actual_values, pred_values)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual_values - pred_values) / actual_values)) * 100
        
        metrics[pred_time] = {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    # Calculate average metrics
    avg_metrics = {
        'MSE': np.mean([m['MSE'] for m in metrics.values()]),
        'MAE': np.mean([m['MAE'] for m in metrics.values()]),
        'RMSE': np.mean([m['RMSE'] for m in metrics.values()]),
        'MAPE': np.mean([m['MAPE'] for m in metrics.values()])
    }
    
    return metrics, avg_metrics

def visualize_predictions(predictions, actuals, sample_count=3):
    """Visualize a sample of predictions against actual values"""
    # Select a sample of prediction times
    pred_times = list(predictions.keys())
    if len(pred_times) > sample_count:
        indices = np.linspace(0, len(pred_times)-1, sample_count, dtype=int)
        sample_times = [pred_times[i] for i in indices]
    else:
        sample_times = pred_times
    
    # Create visualization
    fig, axes = plt.subplots(len(sample_times), 1, figsize=(12, 4*len(sample_times)))
    if len(sample_times) == 1:
        axes = [axes]
    
    for i, pred_time in enumerate(sample_times):
        forecast = predictions[pred_time]
        actual = actuals[pred_time]
        
        axes[i].plot(actual, label='Actual', color='blue')
        axes[i].plot(forecast, label='Forecast', color='red', linestyle='--')
        axes[i].set_title(f'Prediction made at {pred_time}')
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.show()

def run_realtime_simulation():
    """Run the complete real-time simulation"""
    # Load model and data
    model = load_model()
    data = load_historical_data()
    
    if model is None or data is None:
        return
    
    # Run simulation
    print("Starting real-time prediction simulation...")
    predictions, actuals = simulate_realtime_prediction(
        data, 
        model,
        interval_minutes=60,  # Make predictions every 30 minutes
        prediction_horizon=24,  # Predict 24 hours ahead
        simulation_days=7      # Simulate for 7 days
    )
    
    # Evaluate results
    print("\nEvaluating prediction accuracy...")
    _, avg_metrics = evaluate_simulation(predictions, actuals)
    
    print("\nAverage Prediction Metrics:")
    print(f"Mean Squared Error: {avg_metrics['MSE']:.4f}")
    print(f"Mean Absolute Error: {avg_metrics['MAE']:.4f}")
    print(f"Root Mean Squared Error: {avg_metrics['RMSE']:.4f}")
    print(f"Mean Absolute Percentage Error: {avg_metrics['MAPE']:.2f}%")
    
    # Visualize results
    print("\nVisualizing predictions...")
    visualize_predictions(predictions, actuals)

if __name__ == "__main__":
    run_realtime_simulation()