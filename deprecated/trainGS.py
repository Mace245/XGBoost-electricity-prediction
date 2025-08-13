import pandas as pd
import numpy as np
import xgboost as xgb
from lib import data
# The 'algo' library is no longer needed for this forecasting method.
# from lib import algo
from xgboost import plot_importance, plot_tree
import matplotlib.pylab as plt
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV


TEST_SET_HOURS = 24 * 7 * 7 

electricity_raw, temperature_raw = data.fetch_elec_temp()
merged_data_complete = data.prepare_data(electricity_raw, temperature_raw)
# print(f"Base historical data prepared. Shape: {merged_data_complete.shape}, TZ: {merged_data_complete.index.tz}")

# --- Split Data into Training and Test Sets ---
train_data_end_index = len(merged_data_complete) - TEST_SET_HOURS
training_data = merged_data_complete.iloc[:train_data_end_index].copy()

test_data = merged_data_complete.iloc[train_data_end_index:].copy()

# print(f"Training Data: {training_data.shape}")
# print(training_data)
# print(f"Test Data: {test_data.shape}")
# print(test_data)

print(training_data.tail(1))
print(test_data.head(1))

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

    # # Convert 'hour' into sine and cosine components
    # df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    # df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # # You can do the same for month, dayofweek, etc.
    # df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    # df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear', 'temperature',
           'wh_lag_24h', 'wh_lag_72h', 'wh_lag_168h']]
    if label:
        y = df[label]
        return X, y
    return X



X_train, y_train = create_features(training_data, label='Wh')
X_test, y_test = create_features(test_data, label='Wh')

parameters= {
            "learning_rate": [0.1, 0.05, 0.01],
            "max_depth": [3, 5, 7],
            "n_estimators": [1000, 2000, 3000],
            # "learning_rate": [0.01],
            # "max_depth": [7],
            # "n_estimators": [1000],
            "colsample_bytree" : [0.1, 0.5, 1], 
            "subsample":[0.1, 0.3, 0.5, 1.0], 
            "reg_alpha":[0, 0.1, 0.5, 1, 5], 
            "reg_lambda":[0, 0.1, 0.5, 1, 5, 10]
            }

xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
xgb_cv=GridSearchCV(xgb_model, parameters, cv=5, n_jobs=-1)
xgb_cv.fit(X_train, y_train,eval_set=[(X_train, y_train), (X_test, y_test)],verbose=100)

print("tuned hpyerparameters :(best parameters) ", xgb_cv.best_params_)
print("accuracy :", xgb_cv.best_score_)
print('the best estimator:',xgb_cv.best_estimator_)

xgb_cv = GridSearchCV(
    estimator=xgb_model, 
    param_grid=parameters, 
    scoring='neg_mean_absolute_percentage_error', 
    cv=5, 
    verbose=1
)

xgb_cv.fit(X_train, y_train)

best_model = xgb_cv.best_estimator_

best_model.save_model('best_model_mape.ubj')

loaded_model = xgb.XGBRegressor()
loaded_model.load_model('best_model_mape.ubj')

predictions = loaded_model.predict(X_test)

mape_score = mean_absolute_percentage_error(y_test, predictions)

print(f"Final Model MAPE: {mape_score * 100:.3f}%")

results = pd.DataFrame(xgb_cv.cv_results_)

results.to_csv('results2.csv')

params_to_plot = list(parameters.keys())

for param in params_to_plot:
    plt.figure(figsize=(6, 6))

    param_means = results.groupby(f'param_{param}')['mean_test_score'].mean() * -1
    
    plt.plot(param_means.index, param_means, marker='o', linestyle='-', color='b', label='Average MAPE')

    plt.title(f'{param} vs. MAPE')
    plt.xlabel(param)
    plt.ylabel('CV Average MAPE')
    plt.grid(True)
    plt.xticks(ticks=param_means.index)
    plt.yticks(ticks=param_means.values)
    
    # # Keep your logic for log scales
    # if param in ['learning_rate', 'reg_lambda', 'reg_alpha']:
    #     plt.xscale('log')
        
    plt.savefig(f"{param}_vs_mape.png")



# model = xgb.XGBRegressor(
#     max_depth=1, learning_rate=0.3, n_estimators=1000,
#     colsample_bytree=1, subsample=0.3, reg_alpha=0, reg_lambda=1,
#     n_jobs=None, early_stopping_rounds=50, eval_metric='rmse', 
# )

# model.fit(X_train, y_train,
#         eval_set=[(X_train, y_train), (X_test, y_test)],
#        verbose=True) # Change verbose to True if you want to see it train

# # Retrieve the evaluation results (the history of the loss)
# results = model.evals_result()
# train_rmse = results['validation_0']['rmse']
# test_rmse = results['validation_1']['rmse']

# # Plot the learning curve
# plt.figure(figsize=(12, 7))
# plt.plot(train_rmse, label='Training Loss')
# plt.plot(test_rmse, label='Validation (Test) Loss')
# plt.title('XGBoost Learning Curve', fontsize=16)
# plt.xlabel('Number of Boosting Rounds (Trees)', fontsize=12)
# plt.ylabel('RMSE Loss', fontsize=12)
# plt.legend()
# plt.grid(True)
# plt.show()

# test_data['Wh_pred'] = model.predict(X_test)

# def mean_absolute_percentage_error(y_true, y_pred): 
#     """Calculates MAPE given y_true and y_pred"""
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# mape = mean_absolute_percentage_error(y_true=test_data['Wh'],
#                    y_pred=test_data['Wh_pred'])

# print(mape)

# _ = plot_importance(model)
# plt.show()

# x = test_data.index

# plt.plot(x, test_data['Wh'])
# plt.plot(x, test_data['Wh_pred'])
# plt.show() 

# x = merged_data_complete.index

# plt.plot(test_data.index, test_data['Wh'])
# plt.plot(training_data.index, training_data['Wh'])
# plt.show() 

# # model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
# # print(f"    Validation Score for the profile model: {model.best_score:.4f}")