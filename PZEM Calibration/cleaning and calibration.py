import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Load and clean data
df = pd.read_csv('PZEM calib data.csv', skiprows=1, decimal=',')
df.columns = ['UT_V', 'UT_I', 'PZEM_V', 'PZEM_I']

# Convert to numeric values and clean data
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()

print(df)

# 3. Segment data by load type using current ranges
load_conditions = [
    (df['UT_I'] < 0.1, 'Low (8W LED)'),
    ((df['UT_I'] >= 0.1) & (df['UT_I'] < 0.5), 'Medium (36W LED)'),
    (df['UT_I'] >= 0.5, 'High (150W Heater)')
]

# 4. Create calibration functions
def calibrate_and_plot(load_df, load_name):
    # Voltage calibration
    v_model = LinearRegression()
    v_model.fit(load_df[['PZEM_V']], load_df['UT_V'])
    v_slope, v_intercept = v_model.coef_[0], v_model.intercept_
    
    # Current calibration
    i_model = LinearRegression()
    i_model.fit(load_df[['PZEM_I']], load_df['UT_I'])
    i_slope, i_intercept = i_model.coef_[0], i_model.intercept_
    
    # Generate calibration plots
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    
    # Voltage comparison
    ax[0,0].scatter(load_df['PZEM_V'], load_df['UT_V'], alpha=0.5)
    ax[0,0].plot(load_df['PZEM_V'], v_model.predict(load_df[['PZEM_V']]), color='red')
    ax[0,0].set_title(f'{load_name} Voltage Calibration')
    ax[0,0].set_xlabel('PZEM Voltage')
    ax[0,0].set_ylabel('Reference Voltage')
    
    # Current comparison
    ax[0,1].scatter(load_df['PZEM_I'], load_df['UT_I'], alpha=0.5)
    ax[0,1].plot(load_df['PZEM_I'], i_model.predict(load_df[['PZEM_I']]), color='red')
    ax[0,1].set_title(f'{load_name} Current Calibration')
    ax[0,1].set_xlabel('PZEM Current')
    ax[0,1].set_ylabel('Reference Current')
    
    # Residual plots
    ax[1,0].scatter(load_df['UT_V'], load_df['UT_V'] - v_model.predict(load_df[['PZEM_V']]))
    ax[1,0].axhline(0, color='red', linestyle='--')
    ax[1,0].set_title('Voltage Residuals')
    
    ax[1,1].scatter(load_df['UT_I'], load_df['UT_I'] - i_model.predict(load_df[['PZEM_I']]))
    ax[1,1].axhline(0, color='red', linestyle='--')
    ax[1,1].set_title('Current Residuals')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'voltage_slope': v_slope,
        'voltage_intercept': v_intercept,
        'current_slope': i_slope,
        'current_intercept': i_intercept
    }

# 5. Perform calibration for each load type
calibration_results = {}
for condition, name in load_conditions:
    load_df = df[condition].copy()
    print(load_df)
    if not load_df.empty:
        calibration_results[name] = calibrate_and_plot(load_df, name)

# 6. Show calibration coefficients
for load_type, coeffs in calibration_results.items():
    print(f'\n{load_type} Calibration:')
    print(f'Voltage: Corrected = {coeffs["voltage_slope"]:.4f} * PZEM + {coeffs["voltage_intercept"]:.4f}')
    print(f'Current: Corrected = {coeffs["current_slope"]:.4f} * PZEM + {coeffs["current_intercept"]:.4f}')

# 7. Save calibration parameters
pd.DataFrame(calibration_results).T.to_csv('pzem_calibration_coefficients.csv')