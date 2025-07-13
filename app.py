import os
import time
import threading
from datetime import datetime, timedelta, timezone as dt_timezone

import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from antares_http import antares
from lib import data 
from lib import algo 

ANTARES_ACCESS_KEY = '5cd4cda046471a89:75f9e1c6b34bf41a'
ANTARES_PROJECT_NAME = 'UjiCoba_TA'
ANTARES_DEVICE_NAME = 'TA_DKT1'

FETCH_ENABLED = False

DATABASE_FILE = 'app.db'

BACKGROUND_FETCH_INTERVAL_SECONDS = 360 * 60

LATITUDE_CONFIG = 14.5833
LONGITUDE_CONFIG = 121.0
APP_DISPLAY_TIMEZONE = "Asia/Kuala_Lumpur"
MAX_FORECAST_HORIZON_APP = 24 * 7

DMS_FEATURES_LIST = ['hour', 'day_of_week', 'day_of_month', 'temperature']
DMS_TARGET_COL_DATAFRAME = 'kWh'    
DB_TARGET_COL_NAME = 'EnergyWh'
DB_TEMP_COL_NAME = 'TemperatureCelsius' 

MAX_LAG_HOURS = 0

RETRAIN_CHECK_INTERVAL_SECONDS = 3600 * 6
RETRAIN_TRIGGER_DAY = 6 
RETRAIN_TRIGGER_HOUR_UTC = 2 

retraining_status_message = None
retraining_status_category = None 
retraining_status_lock = threading.Lock()
last_retrain_completed_utc = datetime.now(dt_timezone.utc) - timedelta(days=8)
retraining_active = False

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DATABASE_FILE}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.urandom(24)
db = SQLAlchemy(app)
app.jinja_env.globals['utc_now'] = lambda: datetime.now(dt_timezone.utc)


class HourlyReading(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp_utc = db.Column(db.String(20), nullable=False, unique=True, index=True)
    EnergyWh = db.Column(db.Float)
    TemperatureCelsius = db.Column(db.Float, nullable=True)

    def __repr__(self):
        return f"<HourlyReading {self.timestamp_utc} - Energy: {self.EnergyWh}>"

def background_data_collector():
    global last_fetched_hour_utc
    print("Simplified Background Data Collector started (no catch-up)...")

    with app.app_context():
        latest_db_entry = HourlyReading.query.order_by(HourlyReading.timestamp_utc.desc()).first()
        if latest_db_entry:
            try:
                parsed_ts = pd.to_datetime(latest_db_entry.timestamp_utc).tz_convert('UTC')
                last_fetched_hour_utc = parsed_ts.replace(minute=0, second=0, microsecond=0)
                print(f"Data Collector: Last stored hour in DB: {last_fetched_hour_utc.isoformat()}")
            except Exception as e:
                print(f"Data Collector: Error parsing last DB timestamp: {e}. Will proceed as if no prior data for this session.")
                last_fetched_hour_utc = None
        else:
            print("Data Collector: No existing data in DB.")
            last_fetched_hour_utc = None

    while True:
        current_time_utc = datetime.now(dt_timezone.utc)
        target_hour_to_process_utc = current_time_utc.replace(minute=0, second=0, microsecond=0)
        target_hour_iso = target_hour_to_process_utc.isoformat()

        print(f"Data Collector: Current target hour is {target_hour_iso}")

        if last_fetched_hour_utc is not None and target_hour_to_process_utc <= last_fetched_hour_utc:
            print(f"Data Collector: Hour {target_hour_iso} already processed or not yet new. Last was {last_fetched_hour_utc.isoformat()}. Sleeping.")
            time.sleep(BACKGROUND_FETCH_INTERVAL_SECONDS)
            continue

        with app.app_context():
            existing_reading_for_target_hour = HourlyReading.query.filter_by(timestamp_utc=target_hour_iso).first()

            if existing_reading_for_target_hour:
                print(f"  Data for {target_hour_iso} already exists in DB. Checking for updates...")

                new_energy_wh_value = None
                try:
                    antares.setAccessKey(ANTARES_ACCESS_KEY)
                    latest_antares_data = antares.get(ANTARES_PROJECT_NAME, ANTARES_DEVICE_NAME)
                    if latest_antares_data and 'content' in latest_antares_data:
                        new_energy_wh_value = latest_antares_data['content'].get('Energy')
                except:
                    print(f"  Error fetching Antares for update check on {target_hour_iso}")

                commit_update_needed = False
                if new_energy_wh_value is not None: 
                    new_energy_wh_value_float = float(new_energy_wh_value) 
                    if existing_reading_for_target_hour.EnergyWh is None or \
                        new_energy_wh_value_float > existing_reading_for_target_hour.EnergyWh: 
                        print(f"  Updating Energy for {target_hour_iso}. Stored: {existing_reading_for_target_hour.EnergyWh}, New: {new_energy_wh_value_float}")
                        existing_reading_for_target_hour.EnergyWh = new_energy_wh_value_float
                        commit_update_needed = True

                if existing_reading_for_target_hour.TemperatureCelsius is None:
                    print(f"  Temperature for {target_hour_iso} was NULL, attempting to fetch and update.")
                    new_temperature_value = None
                    try:
                        date_str_utc = target_hour_to_process_utc.strftime('%Y-%m-%d')
                        temp_df = data.temp_fetch(date_str_utc, date_str_utc, LATITUDE_CONFIG, LONGITUDE_CONFIG, historical=False) 
                        if temp_df is not None and 'temperature' in temp_df:
                            temp_series = temp_df['temperature']
                            new_temperature_value = temp_series.get(target_hour_to_process_utc) 
                            if pd.notna(new_temperature_value):
                                existing_reading_for_target_hour.TemperatureCelsius = float(new_temperature_value)
                                commit_update_needed = True
                            else:
                                print(f"  Fetched temperature for {target_hour_iso} update was NaN/None.")
                    except Exception as e_temp_update:
                        print(f"  Error fetching temperature for update on {target_hour_iso}: {e_temp_update}")

                if commit_update_needed:
                    try:
                        db.session.commit()
                        print(f"  Updated DB for {target_hour_iso}. Energy: {existing_reading_for_target_hour.EnergyWh}, Temp: {existing_reading_for_target_hour.TemperatureCelsius}")
                    except Exception as e_db_update:
                        db.session.rollback()
                        print(f"  Error committing update for {target_hour_iso}: {e_db_update}")
                else:
                    print(f"  No updates needed for existing record {target_hour_iso}.")

                last_fetched_hour_utc = target_hour_to_process_utc
                time.sleep(BACKGROUND_FETCH_INTERVAL_SECONDS)
                continue 

            print(f"  No existing record for {target_hour_iso}. Fetching new data...")
            energy_wh_value = None
            try:
                antares.setAccessKey(ANTARES_ACCESS_KEY)
                latest_antares_data = antares.get(ANTARES_PROJECT_NAME, ANTARES_DEVICE_NAME)
                if latest_antares_data and 'content' in latest_antares_data:
                    raw_energy = latest_antares_data['content'].get('Energy')
                    if raw_energy is not None:
                        try:
                            energy_wh_value = float(raw_energy) 
                        except (ValueError, TypeError) as e_conv:
                            print(f"  Antares 'Energy' value '{raw_energy}' for {target_hour_iso} is not a valid number: {e_conv}")
                            energy_wh_value = None 
                    else:
                        print(f"  Antares data for {target_hour_iso} missing 'Energy' key or value is None.")
                else:
                    print(f"  No valid content from Antares for {target_hour_iso}.")
            except Exception as e_ant:
                print(f"  Error fetching from Antares for {target_hour_iso}: {e_ant}")

            if energy_wh_value is None: 
                print(f"  Skipping DB store for {target_hour_iso} due to missing or invalid Antares energy data.")
                time.sleep(BACKGROUND_FETCH_INTERVAL_SECONDS)
                continue

            temperature_value_float = None 
            try:
                date_str_utc = target_hour_to_process_utc.strftime('%Y-%m-%d')
                temp_df = data.temp_fetch(date_str_utc, date_str_utc, LATITUDE_CONFIG, LONGITUDE_CONFIG, historical=False)
                if temp_df is not None and 'temperature' in temp_df:
                    temp_series = temp_df['temperature']
                    raw_temp = temp_series.get(target_hour_to_process_utc) 
                    if pd.notna(raw_temp): # Check if it's a valid number (not NaN)
                        temperature_value_float = float(raw_temp)
                    else:
                        print(f"  Temperature for {target_hour_iso} from API was NaN/None.")
                else:
                    print(f"  Temperature data not available or in unexpected format from temp_fetch for {target_hour_iso}.")
            except Exception as e_temp:
                print(f"  Error fetching/processing temperature for {target_hour_iso}: {e_temp}")

            new_entry = HourlyReading(
                timestamp_utc=target_hour_iso,
                EnergyWh=energy_wh_value,
                TemperatureCelsius=temperature_value_float
            )
            db.session.add(new_entry)

            try:
                db.session.commit()
                energy_print_val = f"{new_entry.EnergyWh:.2f}"
                temp_print_val = f"{new_entry.TemperatureCelsius:.2f}" 
                print(f"  Stored new record: {new_entry.timestamp_utc} - Energy: {energy_print_val}, Temp: {temp_print_val}")
                last_fetched_hour_utc = target_hour_to_process_utc
            except Exception as e_db_commit:
                db.session.rollback()
                energy_val_on_fail = new_entry.EnergyWh 
                temp_val_on_fail = new_entry.TemperatureCelsius 

                print(f"  DB Commit Error for new record {target_hour_iso}: {e_db_commit}.")
                print(f"    Attempted to store: Energy={repr(energy_val_on_fail)}, Temp={repr(temp_val_on_fail)}")
        
        time.sleep(BACKGROUND_FETCH_INTERVAL_SECONDS)

def schedule_dms_retraining():
    global retraining_active, last_retrain_completed_utc
    with retraining_status_lock:
        if retraining_active:
            print("Retraining is already active. New trigger ignored.")
            return
        retraining_active = True
    print("\n--- INITIATING PROFILE MODEL RETRAINING ---")
    
    try:
        with app.app_context(): 
            training_df_utc = data.get_all_data_from_db_for_training(
                db.session, HourlyReading,
                output_df_target_col=DMS_TARGET_COL_DATAFRAME,
                output_df_temp_col='temperature',
                model_actual_target_attr=DB_TARGET_COL_NAME,
                model_actual_temp_attr=DB_TEMP_COL_NAME
            )

            if training_df_utc.empty or len(training_df_utc) < (MAX_LAG_HOURS + 24):
                msg = (f"  Insufficient data for retraining ({len(training_df_utc)} records). "
                       f"Need > {MAX_LAG_HOURS + 24}. Retraining aborted.")
                print(msg)
                return 

            print(f"  Retraining profile model with {len(training_df_utc)} data points.")
            
            algo.train_profile_model(
                base_data_for_training=training_df_utc,
                features_list=DMS_FEATURES_LIST,
                target_col=DMS_TARGET_COL_DATAFRAME
            )
            
            last_retrain_completed_utc = datetime.now(dt_timezone.utc)

    except Exception as e:
        print(f"--- FATAL ERROR DURING PROFILE MODEL RETRAINING: {e} ---")
        import traceback; traceback.print_exc()
    finally:
        with retraining_status_lock: 
            retraining_active = False
        print("--- Retraining finished ---")

def background_retraining_scheduler():
    print("Background Retraining Scheduler started...")
    global last_retrain_completed_utc 
    while True:
        now_utc = datetime.now(dt_timezone.utc)
        if (now_utc.weekday() == RETRAIN_TRIGGER_DAY and
            now_utc.hour == RETRAIN_TRIGGER_HOUR_UTC and
            (now_utc - last_retrain_completed_utc).days >= 7):
            print(f"Retraining condition met (Day: {now_utc.weekday()}, Hour: {now_utc.hour} UTC). Last: {last_retrain_completed_utc.date()}")
            
            retrain_job_thread = threading.Thread(target=schedule_dms_retraining)
            retrain_job_thread.start() 
        time.sleep(RETRAIN_CHECK_INTERVAL_SECONDS)


@app.route('/')
def home():
    return redirect(url_for('forecast_view'))

@app.route('/database_log')
def database_log_view():
    page = request.args.get('page', 1, type=int)
    per_page = 50 
    pagination = HourlyReading.query.order_by(HourlyReading.timestamp_utc.desc()).paginate(page=page, per_page=per_page, error_out=False)
    readings_for_page = pagination.items
    readings_display = []
    for r in readings_for_page:
        dt_utc = pd.to_datetime(r.timestamp_utc)
        dt_display = dt_utc.tz_convert(APP_DISPLAY_TIMEZONE)
        readings_display.append({
            'id': r.id,
            'timestamp_display': dt_display.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'EnergyWh': r.EnergyWh,
            'TemperatureCelsius': r.TemperatureCelsius
        })
    return render_template('database_log.html', readings=readings_display, pagination=pagination, APP_DISPLAY_TIMEZONE=APP_DISPLAY_TIMEZONE)


@app.route('/forecast')
def forecast_view():
    latest_reading_db = HourlyReading.query.order_by(HourlyReading.timestamp_utc.desc()).first()
    latest_reading_display = None
    if latest_reading_db:
        dt_utc = pd.to_datetime(latest_reading_db.timestamp_utc)
        dt_display = dt_utc.tz_convert(APP_DISPLAY_TIMEZONE)
        latest_reading_display = {
            "timestamp_display": dt_display.strftime('%Y-%m-%d %H:%M:%S %Z'),
            "EnergyWh": f"{latest_reading_db.EnergyWh:.2f}" if latest_reading_db.EnergyWh is not None else "N/A",
            "TemperatureCelsius": f"{latest_reading_db.TemperatureCelsius:.2f}" if latest_reading_db.TemperatureCelsius is not None else "N/A"
        }

    current_retraining_msg = None
    current_retraining_cat = None
    with retraining_status_lock:
        if retraining_status_message:
            current_retraining_msg = retraining_status_message
            current_retraining_cat = retraining_status_category

    return render_template('forecast.html',
                           latest_reading=latest_reading_display,
                           retraining_message=current_retraining_msg,
                           retraining_category=current_retraining_cat)

@app.route('/run_forecast_dms', methods=['POST'])
def run_forecast_dms_api():
    req_data = request.get_json()
    timeframe_selected_str = req_data.get('timeframe')
    try:
        hours_to_forecast = int(timeframe_selected_str.replace('h', ''))
        if not (0 < hours_to_forecast <= MAX_FORECAST_HORIZON_APP):
            raise ValueError("Requested forecast hours exceed maximum trained horizon or is invalid.")
    except (ValueError, TypeError, AttributeError):
        return jsonify({"error": f"Invalid timeframe. Max is {MAX_FORECAST_HORIZON_APP}h."}), 400
    
    num_records_to_fetch = MAX_LAG_HOURS + hours_to_forecast + 5
    history_query_results = HourlyReading.query.order_by(HourlyReading.timestamp_utc.desc()).limit(num_records_to_fetch).all()

    if len(history_query_results) < MAX_LAG_HOURS : 
        return jsonify({"error": f"Insufficient historical data in DB ({len(history_query_results)} records). "
                                 f"Need at least {MAX_LAG_HOURS} for lags."}), 400
    
    if len(history_query_results) < hours_to_forecast + 1 and hours_to_forecast > 0 : 
        print(f"Warning: Not enough historical data ({len(history_query_results)}) to show full {hours_to_forecast}h preceding context.")

    history_list_for_df = []
    for r_hist in history_query_results:
        history_list_for_df.append({
            'DateTime': pd.to_datetime(r_hist.timestamp_utc, utc=True),
            DMS_TARGET_COL_DATAFRAME: r_hist.EnergyWh,
            'temperature': r_hist.TemperatureCelsius
        })
    full_history_df_utc = pd.DataFrame(history_list_for_df).set_index('DateTime').iloc[::-1].sort_index()

    if full_history_df_utc.empty:
         return jsonify({"error": "Failed to construct historical DataFrame for prediction."}), 500

    history_df_for_prediction_features = full_history_df_utc.copy()
    history_for_display_utc = full_history_df_utc.tail(hours_to_forecast).copy() if hours_to_forecast > 0 else pd.DataFrame()

    last_known_history_time_utc = history_df_for_prediction_features.index[-1]
    forecast_period_start_utc = last_known_history_time_utc + timedelta(hours=1)
    forecast_period_end_utc = last_known_history_time_utc + timedelta(hours=hours_to_forecast)
    future_temperatures_df = None 
    try:
        start_date_utc = forecast_period_start_utc.strftime('%Y-%m-%d')
        end_date_utc = forecast_period_end_utc.strftime('%Y-%m-%d')
        temp_fcst_utc = data.temp_fetch(start_date_utc, end_date_utc, LATITUDE_CONFIG, LONGITUDE_CONFIG, historical=False)
        if temp_fcst_utc is not None and not temp_fcst_utc.empty:
            required_future_idx_utc = pd.date_range(start=forecast_period_start_utc, end=forecast_period_end_utc, freq='h', tz='UTC')
            aligned_temps = temp_fcst_utc['temperature'].reindex(required_future_idx_utc, method='ffill').ffill()
            future_temperatures_df = pd.DataFrame({'temperature': aligned_temps})
            if future_temperatures_df['temperature'].isna().any():
                last_hist_temp = history_df_for_prediction_features['temperature'].dropna().iloc[-1] if not history_df_for_prediction_features['temperature'].dropna().empty else 25.0
                future_temperatures_df['temperature'] = future_temperatures_df['temperature'].fillna(last_hist_temp)
    except Exception as e:
        print(f"Warning: Exception during future temperature fetch: {e}. Using fallback.")
    if future_temperatures_df is None or future_temperatures_df.empty:
        print("  Executing temperature fallback for forecast period.")
        last_hist_temp = history_df_for_prediction_features['temperature'].dropna().iloc[-1] if not history_df_for_prediction_features['temperature'].dropna().empty else 25.0
        required_future_idx_utc = pd.date_range(start=forecast_period_start_utc, end=forecast_period_end_utc, freq='h', tz='UTC')
        future_temperatures_df = pd.DataFrame({'temperature': last_hist_temp}, index=required_future_idx_utc)

    try:
        # --- MODIFIED: Call the new profile prediction function ---
        predictions_series_utc = algo.predict_profile(
            history_df=history_df_for_prediction_features, 
            max_horizon_hours=hours_to_forecast,
            features_list=DMS_FEATURES_LIST,
            target_col=DMS_TARGET_COL_DATAFRAME,
            future_exog_series=future_temperatures_df
        )
    except Exception as e_pred_profile:
        print(f"Error during Profile prediction call: {e_pred_profile}")
        import traceback; traceback.print_exc()
        return jsonify({"error": f"Forecast generation error: {str(e_pred_profile)}"}), 500

    if predictions_series_utc.empty:
        return jsonify({"error": "Forecast generation resulted in no prediction data."}), 500

    history_labels_display = []
    history_data_display = []
    if not history_for_display_utc.empty:
        history_display_local = history_for_display_utc.tz_convert(APP_DISPLAY_TIMEZONE)
        history_labels_display = [dt.strftime('%Y-%m-%d %H:%M') for dt in history_display_local.index]
        history_data_display = history_display_local[DMS_TARGET_COL_DATAFRAME].round(2).fillna(np.nan).replace([np.nan], [None]).tolist()

    forecast_labels_display = []
    forecast_data_display = []
    if not predictions_series_utc.empty:
        predictions_display_local = predictions_series_utc.tz_convert(APP_DISPLAY_TIMEZONE)
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
    global retraining_active, retraining_status_message, retraining_status_category 
    
    with retraining_status_lock: 
        if retraining_active:
            flash('Retraining is already in progress. Please wait.', 'warning')
        else:
            retraining_status_message = "Manual retraining triggered. Check server logs for progress. Status will update here."
            retraining_status_category = "info"
            flash('Manual retraining triggered. Status will update on this page shortly.', 'info')
            
            manual_retrain_thread = threading.Thread(target=schedule_dms_retraining)
            manual_retrain_thread.start()
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
    all_db_readings_query = HourlyReading.query.order_by(HourlyReading.timestamp_utc.desc()).limit(required_total_data_points).all()
    
    if len(all_db_readings_query) < required_total_data_points:
        return jsonify({"error": f"Not enough data in DB for {eval_period_days}-day evaluation."}), 400

    db_data_list_eval = []
    for r_eval in all_db_readings_query:
        db_data_list_eval.append({
            'DateTime': pd.to_datetime(r_eval.timestamp_utc, utc=True),
            DMS_TARGET_COL_DATAFRAME: r_eval.EnergyWh,
            'temperature': r_eval.TemperatureCelsius
        })
    full_eval_period_df_utc = pd.DataFrame(db_data_list_eval).set_index('DateTime').iloc[::-1].sort_index()

    actuals_for_evaluation_df = full_eval_period_df_utc.tail(eval_period_hours).copy()
    history_for_prediction_input = full_eval_period_df_utc.loc[full_eval_period_df_utc.index < actuals_for_evaluation_df.index[0]].copy()
    
    if history_for_prediction_input.empty or len(actuals_for_evaluation_df) != eval_period_hours :
         return jsonify({"error": f"Data splitting error for performance evaluation."}), 500

    actuals_series_for_metrics = actuals_for_evaluation_df[DMS_TARGET_COL_DATAFRAME]
    future_exog_for_eval_period = actuals_for_evaluation_df[['temperature']].copy() if 'temperature' in DMS_FEATURES_LIST else None

    try:
        predictions_for_eval_series = algo.predict_profile(
            history_df=history_for_prediction_input,
            max_horizon_hours=eval_period_hours,
            features_list=DMS_FEATURES_LIST,
            target_col=DMS_TARGET_COL_DATAFRAME,
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
        data_collector_thread = threading.Thread(target=background_data_collector, daemon=True)
        data_collector_thread.start()

    retraining_scheduler_thread = threading.Thread(target=background_retraining_scheduler, daemon=True)
    retraining_scheduler_thread.start()
    
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)