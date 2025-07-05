import matplotlib.pyplot as plt
import numpy as np
import os

# --- Data extracted from your comments ---

# Data for n_estimators
n_estimators_vals = [1000, 2000, 3000]
n_estimators_scores = [28.121, 28.644, 28.96]

# Data for learning_rate
learning_rate_vals = [0.01, 0.05, 0.1]
learning_rate_scores = [28.121, 32.21, 30.949]

# Data for max_depth
max_depth_vals = [3, 5, 7]
max_depth_scores = [25.906, 27.139, 28.121]

# Data for colsample_bytree
colsample_bytree_vals = [0.1, 0.5, 1.0]
colsample_bytree_scores = [31.779, 27.134, 25.906]

# Data for subsample
subsample_vals = [0.1, 0.3, 0.5, 1.0]
subsample_scores = [28.121, 26.511, 27.004, 28.165]

# Data for reg_alpha
reg_alpha_vals = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]
reg_alpha_scores = [26.529, 26.529, 26.574, 26.568, 26.511, 28.054]

# Data for reg_lambda
reg_lambda_vals = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]
reg_lambda_scores = [26.689, 26.591, 26.531, 26.727, 26.639, 26.511]


# --- Generate each plot in a separate window ---

# Plot 1: n_estimators
plt.figure(figsize=(6, 6))
plt.plot(n_estimators_vals, n_estimators_scores, marker='o', color='b')
plt.title('n_estimators vs MAPE')
plt.xlabel('Estimators')
plt.ylabel('MAPE')
plt.xticks(ticks=n_estimators_vals)
plt.yticks(ticks=n_estimators_scores)
plt.grid(True, linestyle='--')
filename = os.path.join("1_n_estimators_effect.png")
plt.savefig(filename)
plt.close() # Close the figure to free up memory
print(f"Saved graph to: {filename}")

# Plot 2: learning_rate
plt.figure(figsize=(6, 6))
plt.plot(learning_rate_vals, learning_rate_scores, marker='o', color='b')
plt.title('learning_rate vs MAPE')
plt.xlabel('learning_rate')
plt.ylabel('MAPE')
plt.xticks(ticks=learning_rate_vals)
plt.yticks(ticks=learning_rate_scores)
plt.grid(True, linestyle='-')
filename = os.path.join("2_learning_rate_effect.png")
plt.savefig(filename)
plt.close()
print(f"Saved graph to: {filename}")

# Plot 3: max_depth
plt.figure(figsize=(6, 6))
plt.plot(max_depth_vals, max_depth_scores, marker='o', color='b')
plt.title('max_depth vs MAPE')
plt.xlabel('max_depth')
plt.ylabel('MAPE')
plt.xticks(ticks=max_depth_vals)
plt.yticks(ticks=max_depth_scores)
plt.grid(True, linestyle='-')
filename = os.path.join("3_max_depth_effect.png")
plt.savefig(filename)
plt.close()
print(f"Saved graph to: {filename}")

# Plot 4: colsample_bytree
plt.figure(figsize=(6, 6))
plt.plot(colsample_bytree_vals, colsample_bytree_scores, marker='o', color='b')
plt.title('colsample_bytree vs MAPE')
plt.xlabel('colsample_bytree')
plt.xticks(ticks=colsample_bytree_vals)
plt.yticks(ticks=colsample_bytree_scores)
plt.ylabel('MAPE')
plt.grid(True, linestyle='-')
filename = os.path.join("4_colsample_bytree_effect.png")
plt.savefig(filename)
plt.close()
print(f"Saved graph to: {filename}")

# Plot 5: subsample
plt.figure(figsize=(6, 6))
plt.plot(subsample_vals, subsample_scores, marker='o', color='b')
plt.title('subsample vs MAPE')
plt.xlabel('Subsample Ratio')
plt.ylabel('MAPE')
plt.xticks(ticks=subsample_vals)
plt.yticks(ticks=subsample_scores)
plt.grid(True, linestyle='-')
filename = os.path.join("5_subsample_effect.png")
plt.savefig(filename)
plt.close()
print(f"Saved graph to: {filename}")

# Plot 6: reg_alpha (L1 Regularization)
plt.figure(figsize=(6, 6))
plt.plot(reg_alpha_vals, reg_alpha_scores, marker='o', color='b')
plt.title('reg_alpha vs MAPE')
plt.xlabel('reg_alpha')
plt.ylabel('MAPE')
plt.xticks(ticks=reg_alpha_vals)
plt.yticks(ticks=reg_alpha_scores)
plt.grid(True, linestyle='-')
filename = os.path.join("6_reg_alpha_effect.png")
plt.savefig(filename)
plt.close()
print(f"Saved graph to: {filename}")

# Plot 7: reg_lambda (L2 Regularization)
plt.figure(figsize=(6, 6))
plt.plot(reg_lambda_vals, reg_lambda_scores, marker='o', color='b')
plt.title('reg_lambda vs MAPE')
plt.xlabel('reg_lambda')
plt.ylabel('MAPE')
plt.xticks(ticks=reg_lambda_vals)
plt.yticks(ticks=reg_lambda_scores)
plt.grid(True, linestyle='-')
filename = os.path.join("7_reg_lambda_effect.png")
plt.savefig(filename)
plt.close()
print(f"Saved graph to: {filename}")