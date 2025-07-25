import pandas as pd
import xgboost as xgb
import pickle
import os

def create_time_features(data_df):
    data_df = data_df.copy()
    data_df['date'] = data_df.index
    data_df['hour'] = data_df['date'].dt.hour
    data_df['dayofweek'] = data_df['date'].dt.dayofweek
    data_df['quarter'] = data_df['date'].dt.quarter
    data_df['month'] = data_df['date'].dt.month
    data_df['year'] = data_df['date'].dt.year
    data_df['dayofyear'] = data_df['date'].dt.dayofyear
    data_df['dayofmonth'] = data_df['date'].dt.day
    data_df['weekofyear'] = data_df['date'].dt.isocalendar().week
    return data_df

def train_xgboost_model(X_train, y_train):
    model = xgb.XGBRegressor(
        max_depth=3, learning_rate=0.01, n_estimators=2000,
        colsample_bytree=1, subsample=0.3, reg_alpha=0, reg_lambda=1,
        n_jobs=None
    )
    split_point = int(len(X_train) * 0.9)
    X_train_fold, X_val_fold = X_train.iloc[:split_point], X_train.iloc[split_point:]
    y_train_fold, y_val_fold = y_train.iloc[:split_point], y_train.iloc[split_point:]
    
    model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
    # print(f"Validation Score for the profile model: {model.best_score:.4f}")
    return model

def train_profile_model(
    base_data_for_training: pd.DataFrame,
    features_list: list[str],
    target_col: str,
):
    print(f"Profile model training started. Using base data of shape: {base_data_for_training.shape}.")

    df = base_data_for_training.copy()
    df = create_time_features(df)
    
    X_train = df[features_list]
    y_train = df[target_col]
    
    if X_train.empty:
        print("Not enough data to create training set. Aborting.")
        return

    print(f"    Training XGBoost model...")
    model = train_xgboost_model(X_train=X_train, y_train=y_train)
    print("    XGBoost model training complete.")
    
    model_filename = f'models/profile_model.pkl'
    os.makedirs('models', exist_ok=True)
    with open(model_filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"    Saved profile model to {model_filename}")
    print("Profile model training complete.")

def predict_profile(
    history_df: pd.DataFrame,
    max_horizon_hours: int,
    features_list: list[str], 
    models_base_path: str = 'models/',
    future_exog_series: pd.DataFrame = None
):
    model_filename = f"{models_base_path}profile_model.pkl"
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)

    last_known_time = history_df.index[-1]
    print(f"Generating profile-based forecast for {max_horizon_hours} hours, starting after: {last_known_time}")

    future_dates = pd.date_range(start=last_known_time + pd.Timedelta(hours=1), periods=max_horizon_hours, freq='h')
    future_df = pd.DataFrame(index=future_dates)
    combined_df = pd.concat([history_df, future_df])
    features_df = create_time_features(combined_df)
    
    if future_exog_series is not None:
        features_df['temperature'].update(future_exog_series['temperature'])
        
    X_future = features_df.loc[future_dates]
    
    X_future = X_future[features_list].fillna(method='ffill').bfill()
    
    predictions = model.predict(X_future)
        
    final_forecast_series = pd.Series(predictions, index=future_dates, name='DateTime')
    print(f"Profile-based forecast generation complete.")
    return final_forecast_series