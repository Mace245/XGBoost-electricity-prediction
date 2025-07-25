import os
import time
import threading
from datetime import datetime, timedelta, timezone as dt_timezone
from apscheduler.schedulers.background import BackgroundScheduler
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from antares_http import antares
from lib import data, algo

ANTARES_ACCESS_KEY = 'fe5c7a15d8c13220:bfd764392a99a094'
ANTARES_PROJECT_NAME = 'TADKT-1'
ANTARES_DEVICE_NAME = 'PMM'

FETCH_ENABLED = False

DATABASE_FILE = 'app.db'

BACKGROUND_FETCH_INTERVAL_SECONDS = 360 * 60

LATITUDE_CONFIG = 14.5833
LONGITUDE_CONFIG = 121.0
APP_DISPLAY_TIMEZONE = "Asia/Jakarta"
MAX_FORECAST_HORIZON_APP = 24 * 7

DMS_FEATURES_LIST = ['hour', 'dayofweek', 'dayofmonth', 'temperature']
DMS_TARGET_COL_DATAFRAME = 'kWh'    
DB_TARGET_COL_NAME = 'EnergyWh'
DB_TEMP_COL_NAME = 'TemperatureCelsius' 

RETRAIN_CHECK_INTERVAL_SECONDS = 3600 * 6
RETRAIN_TRIGGER_DAY = 6 
RETRAIN_TRIGGER_HOUR = 2 

last_retrain_completed = datetime.now(dt_timezone.utc) - timedelta(days=8)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DATABASE_FILE}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.urandom(24)
db = SQLAlchemy(app)

class HourlyReading(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String(20), nullable=False, unique=True, index=True)
    EnergyWh = db.Column(db.Float)
    TemperatureCelsius = db.Column(db.Float, nullable=True)

    def __repr__(self):
        return f"<HourlyReading {self.timestamp} - Energy: {self.EnergyWh}>"

def background_data_collector():
    current_time = datetime.now()
    target_hour_to_process = current_time.replace(minute=0, second=0, microsecond=0)
    target_hour_iso = target_hour_to_process.isoformat()

    print(f"Data Collector Job started for hour: {target_hour_iso}")

    with app.app_context():        
        energy_wh_value = None
        antares.setAccessKey(ANTARES_ACCESS_KEY)
        while(True):
            try:
                latest_antares_data = antares.get(ANTARES_PROJECT_NAME, ANTARES_DEVICE_NAME)
                if 'content' in latest_antares_data and latest_antares_data['content'].get('DailyEnergy'):
                    energy_wh_value = float(latest_antares_data['content'].get('DailyEnergy'))
                    print(f'Got energy for {target_hour_iso}: {energy_wh_value}')
                    break
            except Exception:
                print(f"  Antares fetch attempt no elec data. Retrying in 2 seconds...")
                time.sleep(2)

        temperature_value_float = None
        try:
            date_str = target_hour_to_process.strftime('%Y-%m-%d')
            temp_df = data.temp_fetch(date_str, date_str, LATITUDE_CONFIG, LONGITUDE_CONFIG, historical=False)
            print(temp_df)
            local_aware_dt = target_hour_to_process.astimezone()
            temp_value = temp_df['temperature'].get(local_aware_dt)
            print(target_hour_to_process)
            print(f'Got temperature for {target_hour_iso}: {temp_value}')
        except Exception as e_temp:
            print(f"  Error fetching temperature for new record: {e_temp}")

        new_entry = HourlyReading(
            timestamp=target_hour_iso,
            EnergyWh=energy_wh_value,
            TemperatureCelsius=temp_value
        )
        db.session.add(new_entry)

        try:
            db.session.commit()
            print(f"  Successfully stored new record for {target_hour_iso}.")
        except Exception as e_db_commit:
            db.session.rollback()
            print(f"  DB Commit Error for new record {target_hour_iso}: {e_db_commit}")

    print(f"Data Collector Job finished for hour: {target_hour_iso}")

MAX_LAG_HOURS = 0

def schedule_retraining():
    global last_retrain_completed
    print("\n--- INITIATING PROFILE MODEL RETRAINING ---")
    
    try:
        with app.app_context(): 
            training_df = data.get_all_data_from_db_for_training(
                db.session, HourlyReading,
                output_df_target_col=DMS_TARGET_COL_DATAFRAME,
                output_df_temp_col='temperature',
                model_actual_target_attr=DB_TARGET_COL_NAME,
                model_actual_temp_attr=DB_TEMP_COL_NAME
            )

            if training_df.empty or len(training_df) < (MAX_LAG_HOURS + 24):
                msg = (f"  Insufficient data for retraining ({len(training_df)} records). "
                       f"Need > {MAX_LAG_HOURS + 24}. Retraining aborted.")
                print(msg)
                return 

            print(f"  Retraining profile model with {len(training_df)} data points.")
            
            algo.train_profile_model(
                base_data_for_training=training_df,
                features_list=DMS_FEATURES_LIST,
                target_col=DMS_TARGET_COL_DATAFRAME
            )
            
            last_retrain_completed = datetime.now()

    except Exception as e:
        print(f"--- FATAL ERROR DURING PROFILE MODEL RETRAINING: {e} ---")
        import traceback; traceback.print_exc()

def background_retraining_scheduler():
    print("Background Retraining Scheduler started...")
    global last_retrain_completed
    while True:
        now = datetime.now()
        if (now.weekday() == RETRAIN_TRIGGER_DAY and
            now_.hour == RETRAIN_TRIGGER_HOUR and
            (now - last_retrain_completed).days >= 7):
            print(f"Retraining condition met (Day: {now.weekday()}, Hour: {now.hour}). Last: {last_retrain_completed.date()}")
            
            retrain_job_thread = threading.Thread(target=schedule_retraining)
            retrain_job_thread.start() 
        time.sleep(RETRAIN_CHECK_INTERVAL_SECONDS)


@app.route('/')
def home():
    return redirect(url_for('forecast_view'))

@app.route('/database_log')
def database_log_view():
    page = request.args.get('page', 1, type=int)
    per_page = 50 
    pagination = HourlyReading.query.order_by(HourlyReading.timestamp.desc()).paginate(page=page, per_page=per_page, error_out=False)
    readings_for_page = pagination.items
    readings_display = []
    for r in readings_for_page:
        readings_display.append({
            'id': r.id,
            'timestamp_display': r.timestamp,
            'EnergyWh': r.EnergyWh,
            'TemperatureCelsius': r.TemperatureCelsius
        })
    return render_template('database_log.html', readings=readings_display, pagination=pagination, APP_DISPLAY_TIMEZONE=APP_DISPLAY_TIMEZONE)


@app.route('/forecast')
def forecast_view():
    latest_reading_db = HourlyReading.query.order_by(HourlyReading.timestamp.desc()).first()
    latest_reading_display = None
    if latest_reading_db:
        latest_reading_display = {
            "timestamp_display": f"{latest_reading_db.timestamp}" if latest_reading_db.EnergyWh is not None else "N/A",
            "EnergyWh": f"{latest_reading_db.EnergyWh:.2f}" if latest_reading_db.EnergyWh is not None else "N/A",
            "TemperatureCelsius": f"{latest_reading_db.TemperatureCelsius:.2f}" if latest_reading_db.TemperatureCelsius is not None else "N/A"
        }

    return render_template('forecast.html', latest_reading=latest_reading_display)

@app.route('/run_forecast', methods=['POST'])
def run_forecast_api():
    req_data = request.get_json()
    timeframe_selected_str = req_data.get('timeframe')
    try:
        hours_to_forecast = int(timeframe_selected_str.replace('h', ''))
        if not (0 < hours_to_forecast <= MAX_FORECAST_HORIZON_APP):
            raise ValueError("Requested forecast hours exceed maximum trained horizon or is invalid.")
    except (ValueError, TypeError, AttributeError):
        return jsonify({"error": f"Invalid timeframe. Max is {MAX_FORECAST_HORIZON_APP}h."}), 400
    
    num_records_to_fetch = MAX_LAG_HOURS + hours_to_forecast + 5
    history_query_results = HourlyReading.query.order_by(HourlyReading.timestamp.desc()).limit(num_records_to_fetch).all()

    if len(history_query_results) < MAX_LAG_HOURS : 
        return jsonify({"error": f"Insufficient historical data in DB ({len(history_query_results)} records). "
                                 f"Need at least {MAX_LAG_HOURS} for lags."}), 400
    
    if len(history_query_results) < hours_to_forecast + 1 and hours_to_forecast > 0 : 
        print(f"Warning: Not enough historical data ({len(history_query_results)}) to show full {hours_to_forecast}h preceding context.")

    history_list_for_df = []
    for r_hist in history_query_results:
        history_list_for_df.append({
            'DateTime': pd.to_datetime(r_hist.timestamp),
            DMS_TARGET_COL_DATAFRAME: r_hist.EnergyWh,
            'temperature': r_hist.TemperatureCelsius
        })
    full_history_df = pd.DataFrame(history_list_for_df).set_index('DateTime').iloc[::-1].sort_index()

    if full_history_df.empty:
         return jsonify({"error": "Failed to construct historical DataFrame for prediction."}), 500

    history_df_for_prediction_features = full_history_df.copy()
    history_for_display = full_history_df.tail(hours_to_forecast).copy() if hours_to_forecast > 0 else pd.DataFrame()

    last_known_history_time = history_df_for_prediction_features.index[-1]
    forecast_period_start = last_known_history_time + timedelta(hours=1)
    forecast_period_end = last_known_history_time + timedelta(hours=hours_to_forecast)
    future_temperatures_df = None 
    try:
        start_date = forecast_period_start.strftime('%Y-%m-%d')
        end_date = forecast_period_end.strftime('%Y-%m-%d')
        temp_fcst = data.temp_fetch(start_date, end_date, LATITUDE_CONFIG, LONGITUDE_CONFIG, historical=False)
        if temp_fcst is not None and not temp_fcst.empty:
            required_future_idx = pd.date_range(start=forecast_period_start, end=forecast_period_end, freq='h', tz='UTC')
            aligned_temps = temp_fcst['temperature'].reindex(required_future_idx, method='ffill').ffill()
            future_temperatures_df = pd.DataFrame({'temperature': aligned_temps})
            if future_temperatures_df['temperature'].isna().any():
                last_hist_temp = history_df_for_prediction_features['temperature'].dropna().iloc[-1] if not history_df_for_prediction_features['temperature'].dropna().empty else 25.0
                future_temperatures_df['temperature'] = future_temperatures_df['temperature'].fillna(last_hist_temp)
    except Exception as e:
        print(f"Warning: Exception during future temperature fetch: {e}. Using fallback.")
    if future_temperatures_df is None or future_temperatures_df.empty:
        print("  Executing temperature fallback for forecast period.")
        last_hist_temp = history_df_for_prediction_features['temperature'].dropna().iloc[-1] if not history_df_for_prediction_features['temperature'].dropna().empty else 25.0
        required_future_idx = pd.date_range(start=forecast_period_start, end=forecast_period_end, freq='h', tz='UTC')
        future_temperatures_df = pd.DataFrame({'temperature': last_hist_temp}, index=required_future_idx)

    try:
        predictions_series = algo.predict_profile(
            history_df=history_df_for_prediction_features, 
            max_horizon_hours=hours_to_forecast,
            features_list=DMS_FEATURES_LIST,
            # target_col=DMS_TARGET_COL_DATAFRAME,
            future_exog_series=future_temperatures_df
        )
    except Exception as e_pred_profile:
        print(f"Error during Profile prediction call: {e_pred_profile}")
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Forecast generation error: {str(e_pred_profile)}"}), 500

    if predictions_series.empty:
        return jsonify({"error": "Forecast generation resulted in no prediction data."}), 500

    history_labels_display = []
    history_data_display = []
    if not history_for_display.empty:
        history_display_local = history_for_display.tz_convert(APP_DISPLAY_TIMEZONE)
        history_labels_display = [dt.strftime('%Y-%m-%d %H:%M') for dt in history_display_local.index]
        history_data_display = history_display_local[DMS_TARGET_COL_DATAFRAME].round(2).fillna(np.nan).replace([np.nan], [None]).tolist()

    forecast_labels_display = []
    forecast_data_display = []
    if not predictions_series.empty:
        predictions_display_local = predictions_series.tz_convert(APP_DISPLAY_TIMEZONE)
        forecast_labels_display = [dt.strftime('%Y-%m-%d %H:%M') for dt in predictions_display_local.index]
        forecast_data_display = predictions_display_local.round(2).fillna(np.nan).replace([np.nan], [None]).tolist()

    return jsonify({
        "history_labels": history_labels_display,
        "history_data": history_data_display,
        "forecast_labels": forecast_labels_display, 
        "forecast_data": forecast_data_display   
    })


@app.route('/trigger_manual_retrain', methods=['POST'])
def trigger_manual_retrain_route():            
    schedule_retraining()
    return redirect(url_for('forecast_view')) 


@app.route('/model_performance')
def model_performance_view():
    return render_template('model_performance.html', max_eval_days=MAX_FORECAST_HORIZON_APP // 24)

@app.route('/calculate_model_performance', methods=['POST'])
def calculate_model_performance_api():
    req_data = request.get_json()
    eval_period_days = req_data.get('eval_period_days', 7)

    if not isinstance(eval_period_days, int) or not (0 < eval_period_days <= (MAX_FORECAST_HORIZON_APP // 24)):
        return jsonify({"error": f"Evaluation period (days) must be a positive integer up to {MAX_FORECAST_HORIZON_APP // 24}."}), 400
    
    eval_period_hours = eval_period_days * 24

    print(f"Calculating model performance for the last {eval_period_days} days ({eval_period_hours} hours).")

    required_total_data_points = eval_period_hours + MAX_LAG_HOURS + 5
    all_db_readings_query = HourlyReading.query.order_by(HourlyReading.timestamp.desc()).limit(required_total_data_points).all()
    
    if len(all_db_readings_query) < required_total_data_points:
        return jsonify({"error": f"Not enough data in DB for {eval_period_days}-day evaluation."}), 400

    db_data_list_eval = []
    for r_eval in all_db_readings_query:
        db_data_list_eval.append({
            'DateTime': pd.to_datetime(r_eval.timestamp, utc=True),
            DMS_TARGET_COL_DATAFRAME: r_eval.EnergyWh,
            'temperature': r_eval.TemperatureCelsius
        })
    full_eval_period_df = pd.DataFrame(db_data_list_eval).set_index('DateTime').iloc[::-1].sort_index()

    actuals_for_evaluation_df = full_eval_period_df.tail(eval_period_hours).copy()
    history_for_prediction_input = full_eval_period_df.loc[full_eval_period_df.index < actuals_for_evaluation_df.index[0]].copy()
    
    if history_for_prediction_input.empty or len(actuals_for_evaluation_df) != eval_period_hours :
         return jsonify({"error": f"Data splitting error for performance evaluation."}), 500

    actuals_series_for_metrics = actuals_for_evaluation_df[DMS_TARGET_COL_DATAFRAME]
    future_exog_for_eval_period = actuals_for_evaluation_df[['temperature']].copy() if 'temperature' in DMS_FEATURES_LIST else None

    try:
        predictions_for_eval_series = algo.predict_profile(
            history_df=history_for_prediction_input,
            max_horizon_hours=eval_period_hours,
            features_list=DMS_FEATURES_LIST,
            future_exog_series=future_exog_for_eval_period
        )
    except Exception as e_eval_pred:
        print(f"Error during performance evaluation prediction: {e_eval_pred}")
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Error generating predictions for evaluation: {str(e_eval_pred)}"}), 500

    if predictions_for_eval_series.empty:
        return jsonify({"error": "Performance evaluation: Prediction generation resulted in no data."}), 500

    comparison_df_eval = pd.DataFrame({'Actual': actuals_series_for_metrics, 'Forecast': predictions_for_eval_series}).dropna()
    if comparison_df_eval.empty:
        return jsonify({"error": "No overlapping data between actuals and forecast for performance metrics calculation."}), 500

    actuals_aligned_eval = comparison_df_eval['Actual']
    forecast_aligned_eval = comparison_df_eval['Forecast']

    metrics_results = {}
    metrics_results['rmse'] = np.sqrt(np.mean((actuals_aligned_eval - forecast_aligned_eval)**2))
    metrics_results['mae'] = np.mean(np.abs(actuals_aligned_eval - forecast_aligned_eval))
    valid_actuals_for_mape = actuals_aligned_eval[actuals_aligned_eval != 0]
    if not valid_actuals_for_mape.empty:
        metrics_results['mape'] = np.mean(np.abs((valid_actuals_for_mape - forecast_aligned_eval.loc[valid_actuals_for_mape.index]) / valid_actuals_for_mape)) * 100
    else:
        metrics_results['mape'] = float('inf')
    
    actuals_display_local = actuals_series_for_metrics.tz_convert(APP_DISPLAY_TIMEZONE)
    predictions_display_local = predictions_for_eval_series.tz_convert(APP_DISPLAY_TIMEZONE)

    chart_labels_eval = [dt_loc.strftime('%Y-%m-%d %H:%M') for dt_loc in actuals_display_local.index]

    chart_actual_data_eval = actuals_display_local.round(2).fillna(np.nan).replace([np.nan], [None]).tolist()
    
    aligned_forecast_for_chart = predictions_display_local.reindex(actuals_display_local.index)
    chart_forecast_data_eval = aligned_forecast_for_chart.round(2).fillna(np.nan).replace([np.nan], [None]).tolist()

    return jsonify({
        "metrics": {k: round(v, 2) if isinstance(v, (float, np.floating)) and pd.notna(v) else (v if pd.notna(v) else 'N/A') for k, v in metrics_results.items()},
        "chart_data": {
            "labels": chart_labels_eval,
            "actuals": chart_actual_data_eval,
            "forecasts": chart_forecast_data_eval
        },
        "eval_period_days": eval_period_days,
        "eval_start_time_display": actuals_display_local.index[0].strftime('%Y-%m-%d %H:%M %Z'),
        "eval_end_time_display": actuals_display_local.index[-1].strftime('%Y-%m-%d %H:%M %Z')
    })


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        print(f"Database '{DATABASE_FILE}' ensured/created.")

    if FETCH_ENABLED:
        background_data_collector()
        sched = BackgroundScheduler()
        sched.add_job(background_data_collector, 'cron', minute=0)
        sched.start()

    retraining_scheduler_thread = threading.Thread(target=background_retraining_scheduler, daemon=True)
    retraining_scheduler_thread.start()

    # Expose
    # ngrok http --url=sincere-moccasin-likely.ngrok-free.app 80
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)