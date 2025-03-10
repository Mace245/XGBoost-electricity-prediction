import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

# 1. Load and clean data
df = pd.read_csv('PZEM calib data.csv', skiprows=1, decimal=',')
df.columns = ['UT_V', 'UT_I', 'PZEM_V', 'PZEM_I']
df = df.apply(pd.to_numeric, errors='coerce').dropna()
df.loc[df['UT_V'] > 250, 'UT_V'] = 207.0

# 2. Non-linear calibration function
def polynomial_calibration(X, y, degree=2):
    model = make_pipeline(
        PolynomialFeatures(degree),
        LinearRegression()
    )
    model.fit(X.values.reshape(-1,1), y)
    return model

# 3. Modified calibration and plotting
def calibrate_and_plot_nonlinear(load_df, plot_title='All Loads', degree=2):
    # Voltage calibration
    v_model = polynomial_calibration(load_df['PZEM_V'], load_df['UT_V'], degree)
    v_pred = v_model.predict(load_df['PZEM_V'].values.reshape(-1,1))
    
    # Current calibration
    i_model = polynomial_calibration(load_df['PZEM_I'], load_df['UT_I'], degree)
    i_pred = i_model.predict(load_df['PZEM_I'].values.reshape(-1,1))
    
    # Generate calibration plots
    fig, ax = plt.subplots(2, 2, figsize=(14, 12))
    
    # Voltage comparison
    x_v = np.linspace(load_df['PZEM_V'].min(), load_df['PZEM_V'].max(), 100)
    ax[0,0].scatter(load_df['PZEM_V'], load_df['UT_V'], alpha=0.5)
    ax[0,0].plot(x_v, v_model.predict(x_v.reshape(-1,1)), color='red')
    ax[0,0].set_title(f'{plot_title} Voltage Calibration (Degree {degree})')
    ax[0,0].set_xlabel('PZEM Voltage')
    ax[0,0].set_ylabel('Reference Voltage')
    
    # Current comparison
    x_i = np.linspace(load_df['PZEM_I'].min(), load_df['PZEM_I'].max(), 100)
    ax[0,1].scatter(load_df['PZEM_I'], load_df['UT_I'], alpha=0.5)
    ax[0,1].plot(x_i, i_model.predict(x_i.reshape(-1,1)), color='red')
    ax[0,1].set_title(f'{plot_title} Current Calibration (Degree {degree})')
    ax[0,1].set_xlabel('PZEM Current')
    ax[0,1].set_ylabel('Reference Current')
    
    # Residual plots
    ax[1,0].scatter(load_df['UT_V'], load_df['UT_V'] - v_pred)
    ax[1,0].axhline(0, color='red', linestyle='--')
    ax[1,0].set_title('Voltage Residuals')
    
    ax[1,1].scatter(load_df['UT_I'], load_df['UT_I'] - i_pred)
    ax[1,1].axhline(0, color='red', linestyle='--')
    ax[1,1].set_title('Current Residuals')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate metrics
    v_r2 = r2_score(load_df['UT_V'], v_pred)
    i_r2 = r2_score(load_df['UT_I'], i_pred)
    v_rmse = np.sqrt(mean_squared_error(load_df['UT_V'], v_pred))
    i_rmse = np.sqrt(mean_squared_error(load_df['UT_I'], i_pred))
    
    return {
        'voltage_model': v_model,
        'current_model': i_model,
        'voltage_r2': v_r2,
        'current_r2': i_r2,
        'voltage_rmse': v_rmse,
        'current_rmse': i_rmse
    }

# 4. Perform non-linear calibration
degree = 2  # Can be adjusted based on data complexity
results = calibrate_and_plot_nonlinear(df, degree=degree)

# 5. Display calibration metrics
print(f"\nNon-linear Calibration Metrics (Degree {degree}):")
print(f"Voltage R²: {results['voltage_r2']:.4f}")
print(f"Voltage RMSE: {results['voltage_rmse']:.4f} V")
print(f"Current R²: {results['current_r2']:.4f}")
print(f"Current RMSE: {results['current_rmse']:.4f} A")

# 6. Save model coefficients (example for voltage)
voltage_coeffs = results['voltage_model'].named_steps['linearregression'].coef_
voltage_intercept = results['voltage_model'].named_steps['linearregression'].intercept_
print(f"\nVoltage Polynomial Coefficients:")
print(f"Intercept: {voltage_intercept:.4f}")
print(f"Coefficients: {voltage_coeffs}")