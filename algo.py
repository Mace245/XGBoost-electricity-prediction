import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

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