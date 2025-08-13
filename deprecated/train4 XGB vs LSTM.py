import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow import keras
import xgboost as xgb
from sklearn.metrics import mean_absolute_percentage_error
import os, time
import seaborn as sns

plt.rcParams.update({'font.size': 20})

# --- CONFIGURATION ---
DATA_FILE_PATH = 'Data/processed_hourly_Wh_data.csv'
LSTM_MODEL_PATH = 'lstm_electricity_model.keras'
XGB_MODEL_PATH = 'xgb_electricity_model.ubj'
RESULTS_CSV_PATH = 'prediction_comparison_results.csv'

# --- 1. DATA PREPARATION ---
print("Loading and preparing data...")
data = pd.read_csv(DATA_FILE_PATH)
data['Date'] = pd.to_datetime(data['DateTime'])
data.set_index('Date', inplace=True)
production = data['Wh'].astype(float).values.reshape(-1, 1)

# --- 2. LSTM MODEL PREPARATION AND TRAINING ---

# 1. Split the original, unscaled data first
# We also split the dates here to keep them aligned.
train_size = int(len(production) * 0.8)
train_data = production[:train_size]
test_data = production[train_size:]
train_dates = data.index[:train_size]
test_dates_full = data.index[train_size:] # We will trim this after windowing

# 2. Fit the scaler ONLY on the training data and transform it
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data)

# 3. Use the SAME scaler to transform the test data (do NOT use .fit_transform again)
scaled_test_data = scaler.transform(test_data)

# 4. Now, create your windowed datasets separately for train and test
window_size = 24 # Note: You should still increase this to fix Issue #2
X_train_lstm, y_train_lstm = [], []
for i in range(window_size, len(scaled_train_data)):
    X_train_lstm.append(scaled_train_data[i - window_size:i, 0])
    y_train_lstm.append(scaled_train_data[i, 0])

X_test_lstm, y_test_lstm = [], []
for i in range(window_size, len(scaled_test_data)):
    X_test_lstm.append(scaled_test_data[i - window_size:i, 0])
    y_test_lstm.append(scaled_test_data[i, 0])

# Convert lists to numpy arrays
X_train_lstm, y_train_lstm = np.array(X_train_lstm), np.array(y_train_lstm)
X_test_lstm, y_test_lstm = np.array(X_test_lstm), np.array(y_test_lstm)

# Align the test dates with the y_test_lstm labels
dates_test = test_dates_full[window_size:]

# Check if a trained LSTM model exists
if os.path.exists(LSTM_MODEL_PATH) and 0==1:
    print(f"\nLoading existing LSTM model from '{LSTM_MODEL_PATH}'...")
    lstm_model = load_model(LSTM_MODEL_PATH)
else:
    print("\nNo existing LSTM model found. Training a new one...")
    start_time = time.time()
    # Build and Train the LSTM model
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))
    lstm_model.add(LSTM(units=32))
    lstm_model.add(Dense(1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    
    history = lstm_model.fit(X_train_lstm, y_train_lstm, epochs=30, batch_size=32, validation_split=0.1, verbose=1)

    end_time = time.time()
    time_lstm = end_time - start_time
    
    print(f"Saving new LSTM model to '{LSTM_MODEL_PATH}'...")
    lstm_model.save(LSTM_MODEL_PATH)

start_time = time.time()

# Make predictions with LSTM
print("Making predictions with LSTM model...")
predictions_lstm_scaled = lstm_model.predict(X_test_lstm)
predictions_lstm = scaler.inverse_transform(predictions_lstm_scaled).flatten()

end_time = time.time()
time_pred_lstm = start_time - end_time 

# --- 3. XGBOOST MODEL PREPARATION AND TRAINING ---

def create_xgb_features(df, label=None):
    """Creates time-series and lag features for the XGBoost model."""
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    
    features = ['hour','dayofweek','quarter','month','year','dayofyear','dayofmonth','weekofyear']
    
    # Drop rows with NaN values created by lag features
    df.dropna(inplace=True)
    
    X = df[features]
    if label:
        y = df[label]
        return X, y
    return X, None

# Split data for XGBoost
train_df_xgb, test_df_xgb = train_test_split(data, test_size=0.2, shuffle=False)

X_train_xgb, y_train_xgb = create_xgb_features(train_df_xgb.copy(), label='Wh')
X_test_xgb, y_test_xgb = create_xgb_features(test_df_xgb.copy(), label='Wh')

start_time = time.time()

# Build and train XGBoost model
print("\nTraining XGBoost model...")
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=3,
    colsample_bytree=1.0,
    subsample=0.3,
    reg_alpha=5.0,
    reg_lambda=10.0,
    n_jobs=-1
)
xgb_model.fit(X_train_xgb, y_train_xgb, eval_set=[(X_test_xgb, y_test_xgb)], verbose=False)

end_time = time.time()
time_xgb = end_time - start_time

# Make predictions with XGBoost
print("Making predictions with XGBoost model...")

start_time = time.time()
predictions_xgb = xgb_model.predict(X_test_xgb)
end_time = time.time()

time_pred_xgb = end_time - start_time

# --- 4. EVALUATION AND COMPARISON ---

# Align data for comparison (since XGBoost lags drop initial rows)
y_test_actual = y_test_xgb.values

# Calculate metrics for both models
mape_lstm = mean_absolute_percentage_error(y_test_actual, predictions_lstm[-len(y_test_actual):]) * 100
mape_xgb = mean_absolute_percentage_error(y_test_actual, predictions_xgb) * 100

print("\n--- Model Comparison ---")
print(f"MAPE of LSTM Model: {mape_lstm:.2f} %")
print(f"MAPE of XGBoost Model: {mape_xgb:.2f} %")

# --- 5. SAVE RESULTS AND VISUALIZE ---

# Create a results DataFrame for comparison
results_df = pd.DataFrame({
    'Date': y_test_xgb.index,
    'Actual_Production': y_test_actual,
    'LSTM_Predicted': predictions_lstm[-len(y_test_actual):],
    'XGBoost_Predicted': predictions_xgb
})

# Save the full results DataFrame to a CSV file
results_df.to_csv(RESULTS_CSV_PATH, index=False)
print(f"\nFull comparison results saved to '{RESULTS_CSV_PATH}'")

# Create a subset of the results for plotting just the first week
plot_df = results_df.head(24 * 7) # 24 hours * 7 days

# Plot the 1-week comparison
plt.figure(figsize=(15, 7))
plt.plot(plot_df['Date'], plot_df['Actual_Production'], label='Aktual', color='black', linewidth=2)
plt.plot(plot_df['Date'], plot_df['LSTM_Predicted'], label=f'LSTM ', color='blue', linestyle='--')
plt.plot(plot_df['Date'], plot_df['XGBoost_Predicted'], label=f'XGBoost', color='red', linestyle=':')
plt.title('LSTM vs. XGBoost')
plt.xlabel('Tanggal')
plt.ylabel('Daya (kWh)')
plt.legend()
plt.grid(True)
plt.savefig('XGB vs LSTM.png', bbox_inches='tight')
plt.show()

# 1. Prepare the data
model_names = ['XGBoost', 'LSTM']
training_times = [time_xgb, time_lstm]

# 2. Create the bar chart using seaborn
plt.figure(figsize=(8, 6))
ax = sns.barplot(x=model_names, y=training_times, palette="viridis")

# 3. Add labels and a title
ax.set_xlabel("Model")
ax.set_ylabel("Waktu Pelatihan (S)")
ax.set_title("Perbandingan Waktu Pelatihan Model")
ax.set_ylim(0, max(training_times) * 1.1)

# 4. (Optional but recommended) Add data labels
for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f} s',
               (p.get_x() + p.get_width() / 2., p.get_height()),
               ha='center', va='center',
               xytext=(0, 9),
               textcoords='offset points')

# 5. Display the chart
plt.savefig('Training time.png', bbox_inches='tight')
plt.show()