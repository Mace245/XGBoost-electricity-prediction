import pandas as pd
import numpy as np # Import numpy for handling potential zeros
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error # <-- CHANGED
import matplotlib.pyplot as plt
from pathlib import Path

# --- Configuration ---
DATA_FILE_PATH = Path('Data/processed_hourly_Wh_data.csv')
MODEL_SAVE_PATH = Path('xgb_model.json')
TEST_SIZE_HOURS = 24 * 30  # Use the last 30 days for testing
FORECAST_HORIZON_HOURS = 48 # Predict the next 48 hours

def load_and_prepare_data(file_path):
    """Loads, parses, and performs initial preparation of the time series data."""
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Convert 'DateTime' to datetime objects and set as index
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.set_index('DateTime')
    df = df.sort_index()
    
    # Ensure a consistent hourly frequency, filling any gaps
    df = df.asfreq('H', method='ffill')
    
    print("Data loaded and prepared.")
    return df

def create_features(df):
    """Create time-series features from the datetime index."""
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    
    # Lag and rolling features
    df['lag_24h'] = df['Wh'].shift(24)
    df['lag_48h'] = df['Wh'].shift(48)
    df['lag_1w'] = df['Wh'].shift(24 * 7)
    df['rolling_mean_3h'] = df['Wh'].shift(1).rolling(window=3).mean()
    df['rolling_mean_24h'] = df['Wh'].shift(1).rolling(window=24).mean()
    
    return df

def plot_feature_importance(model, features):
    """Plots the feature importance of the trained model."""
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': features, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
    plt.title('XGBoost Feature Importance')
    plt.xlabel('Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def main():
    """Main function to run the forecasting pipeline."""
    df = load_and_prepare_data(DATA_FILE_PATH)
    df_featured = create_features(df)
    
    TARGET = 'Wh'
    df_featured = df_featured.dropna()
    FEATURES = [col for col in df_featured.columns if col != TARGET]
    
    split_point = df_featured.index.max() - pd.Timedelta(hours=TEST_SIZE_HOURS)
    train = df_featured.loc[df_featured.index <= split_point]
    test = df_featured.loc[df_featured.index > split_point]

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    # --- 4. TRAIN XGBOOST MODEL ---
    print("Training XGBoost model with MAPE evaluation...")
    reg = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mape', # <-- CHANGED from 'mae' to 'mape'
        early_stopping_rounds=50,
        n_jobs=-1
    )

    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)

    reg.save_model(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # --- 5. EVALUATE MODEL ---
    print("Evaluating model...")
    plot_feature_importance(reg, FEATURES)
    
    test['prediction'] = reg.predict(X_test)
    
    # --- Calculate MAPE --- # <-- CHANGED
    # NOTE: MAPE is undefined for actual values of zero.
    # We filter out these cases to avoid errors or infinite results.
    non_zero_mask = y_test != 0
    if not np.all(non_zero_mask):
        print("Warning: Found actual values of 0 in the test set. These will be ignored for MAPE calculation.")
    
    mape = mean_absolute_percentage_error(y_test[non_zero_mask], test['prediction'][non_zero_mask])
    print(f"Mean Absolute Percentage Error (MAPE) on Test Set: {mape * 100:.2f}%")

    test[[TARGET, 'prediction']].plot(figsize=(15, 5), style=['-', '--'])
    plt.title(f'Test Set: Actual vs. Prediction (MAPE: {mape * 100:.2f}%)')
    plt.ylabel('Wh (Watt-hours)')
    plt.grid(True)
    plt.show()
    
    # --- 6. FORECAST FUTURE USAGE ---
    print(f"Forecasting future usage for the next {FORECAST_HORIZON_HOURS} hours...")
    last_date = df_featured.index.max()
    future_dates = pd.date_range(start=last_date, periods=FORECAST_HORIZON_HOURS + 1, freq='H', inclusive='right')
    
    future_df = pd.DataFrame(index=future_dates)
    combined_df = pd.concat([df_featured.tail(24*7), future_df])
    combined_featured = create_features(combined_df)
    
    X_future = combined_featured.loc[future_dates][FEATURES]
    
    future_df['prediction'] = reg.predict(X_future)
    print("\nFuture Forecast:")
    print(future_df['prediction'])

    plt.figure(figsize=(15, 7))
    test[TARGET].plot(label='Historical Test Data')
    test['prediction'].plot(label='Test Predictions', style='--')
    future_df['prediction'].plot(label='Future Forecast', style='--')
    plt.title('Electricity Usage Forecast')
    plt.ylabel('Wh (Watt-hours)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()