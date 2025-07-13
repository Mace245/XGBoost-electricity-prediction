import pandas as pd
import numpy as np
import xgboost as xgb
from lib import data
from xgboost import plot_importance
import matplotlib.pylab as plt
from sklearn.metrics import mean_absolute_percentage_error

# TEST_SET_HOURS = 24 * 7 * 4 * 5 
TEST_SET_HOURS = 24 * 7 * 7

electricity_raw, temperature_raw = data.fetch_elec_temp()
merged_data_complete = data.prepare_data(electricity_raw, temperature_raw)

train_data_end_index = len(merged_data_complete) - TEST_SET_HOURS
training_data = merged_data_complete.iloc[:train_data_end_index].copy()
test_data = merged_data_complete.iloc[train_data_end_index:].copy()

# print(f"Training Data: {training_data.shape}")
# print(training_data)
# print(f"Test Data: {test_data.shape}")
# print(test_data)

# print(training_data.tail(1))
# print(test_data.head(1))

def create_features(df, label=None):
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df['temperature'] = df['temperature']
    df['wh_lag_24h'] = df['Wh'].shift(24)
    df['wh_lag_72h'] = df['Wh'].shift(72)
    df['wh_lag_168h'] = df['Wh'].shift(168)
    
    X = df[['hour','dayofweek','quarter','month','year',
            'dayofyear','dayofmonth','weekofyear',
            'temperature',
            'wh_lag_24h', 'wh_lag_72h', 'wh_lag_168h'
            ]]
    if label:
        y = df[label]
        return X, y
    return X

X_train, y_train = create_features(training_data, label='Wh')
X_test, y_test = create_features(test_data, label='Wh')

model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,      # 3000/28.96    2000/28.644     1000/28.121
    learning_rate=0.01,     # 0.01/28.121   0.05/32.21      0.1/30.949
    max_depth=7,            # 7/28.121      5/27.139        3/25.906      
    colsample_bytree=1.0,   # 1.0/25.906    0.5/27.134      0.1/31.779
    subsample=0.1,          # 0.1/28.121    0.3/26.511      0.5/27.004      1.0/28.165
    reg_alpha=5.0,          # 0.0/26.529    0.1/26.529      0.5/26.574      1.0/26.568      5.0/26.511     10.0/28.054
    reg_lambda=10.0,        # 0.0/26.689    0.1/26.591      0.5/26.531      1.0/26.727      5.0/26.639     10.0/26.511
    n_jobs=-1 
)

print("Training the XGBoost model...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)], 
    verbose=100                    
)

print("Saving the trained model...")
model.save_model('final_xgboost_model.ubj')

print("Loading model and making predictions...")
loaded_model = xgb.XGBRegressor()
loaded_model.load_model('final_xgboost_model.ubj')

predictions = loaded_model.predict(X_test)

mape_score = mean_absolute_percentage_error(y_test, predictions)
print(f"Final Model MAPE on Test Set: {mape_score * 100:.3f}%")

plot_importance(model, height=0.9, importance_type='gain')
plt.title("Kepentingan Fitur")
plt.rcParams.update({'font.size': 20})
plt.xlabel('Skor Kepentingan')
plt.ylabel('Fitur')
plt.grid(False)
plt.savefig('importance w temp',bbox_inches='tight')
plt.show()

test_data['Wh_pred'] = predictions
plt.figure(figsize=(15, 6))
test_data['Wh'].plot(label='Actual Values', style='-')
test_data['Wh_pred'].plot(label='Predictions', style='--')
plt.title('Actual vs. Predicted Values')
plt.xlabel('Date')
plt.ylabel('Wh')
plt.legend()
plt.grid(True)
plt.show()