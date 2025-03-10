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

# 2. Handle outliers
df.loc[df['UT_V'] > 250, 'UT_V'] = 207.0  # Fix typo in high-power measurements

# 3. Create consolidated calibration model
def calibrate_and_plot(load_df, plot_title='All Loads'):
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
    ax[0,0].set_title(f'{plot_title} Voltage Calibration')
    ax[0,0].set_xlabel('PZEM Voltage')
    ax[0,0].set_ylabel('Reference Voltage')
    
    # Current comparison
    ax[0,1].scatter(load_df['PZEM_I'], load_df['UT_I'], alpha=0.5)
    ax[0,1].plot(load_df['PZEM_I'], i_model.predict(load_df[['PZEM_I']]), color='red')
    ax[0,1].set_title(f'{plot_title} Current Calibration')
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

# 4. Create consolidated deviation analysis
def plot_current_deviations(load_df):
    plt.figure(figsize=(12, 6))
    
    # Calculate current deviations
    current_dev = load_df['PZEM_I'] - load_df['UT_I']
    
    # Create line plot with markers
    plt.plot(load_df.index, current_dev, 
             marker='o', linestyle='--', 
             color='#2ca02c', linewidth=1, 
             markersize=6, markerfacecolor='white')
    
    # Add zero reference line
    plt.axhline(0, color='red', linestyle='-', linewidth=1)
    
    # Formatting
    plt.title('Current Measurement Deviations (All Loads)\n(PZEM Current - Reference Current)')
    plt.xlabel('Measurement Index')
    plt.ylabel('Deviation (A)')
    plt.grid(True, alpha=0.3)
    plt.xticks(load_df.index)
    
    # Add deviation values as text annotations
    for i, dev in enumerate(current_dev):
        plt.text(load_df.index[i], dev, f'{dev:.3f}A', 
                 ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.show()

# 5. Perform consolidated calibration
calibration_results = calibrate_and_plot(df)

# 6. Show consolidated calibration coefficients
print('\nConsolidated Calibration Parameters:')
print(f"Voltage: Corrected = {calibration_results['voltage_slope']:.4f} * PZEM + {calibration_results['voltage_intercept']:.4f}")
print(f"Current: Corrected = {calibration_results['current_slope']:.4f} * PZEM + {calibration_results['current_intercept']:.4f}")

# 7. Display consolidated current deviations
print("\nConsolidated Current Deviation Analysis:")
plot_current_deviations(df)
current_dev = df['PZEM_I'] - df['UT_I']
print(f"\nCurrent Deviation Statistics:")
print(f"Mean: {current_dev.mean():.4f} A")
print(f"Std Dev: {current_dev.std():.4f} A")
print(f"Max: {current_dev.max():.4f} A")
print(f"Min: {current_dev.min():.4f} A")

# 8. Save calibration parameters
pd.DataFrame(calibration_results, index=[0]).to_csv('pzem_consolidated_calibration.csv', index=False)