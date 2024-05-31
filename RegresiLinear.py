import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit

# Path to your CSV file
file_path = 'Student_Performance.csv'

# Load the dataset
data = pd.read_csv(file_path)

# Data
X = data['Hours Studied'].values.reshape(-1, 1)
y = data['Performance Index'].values

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)
y_pred_linear = linear_model.predict(X)

# Plot Linear Regression
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred_linear, color='red', label='Linear regression')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.title('Linear Regression')
plt.legend()
plt.show()

# Calculate RMS error for Linear Regression
rms_error_linear = np.sqrt(mean_squared_error(y, y_pred_linear))
print(f'RMS Error (Linear Regression): {rms_error_linear}')

# Power Function: y = a * x^b
def power_function(x, a, b):
    return a * np.power(x, b)

# Fit Power Function to Data
params, _ = curve_fit(power_function, data['Hours Studied'], data['Performance Index'])
a, b = params
y_pred_power = power_function(data['Hours Studied'], a, b)

# Plot Power Regression
plt.scatter(data['Hours Studied'], data['Performance Index'], color='blue', label='Data points')
plt.plot(data['Hours Studied'], y_pred_power, color='green', label='Power regression')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.title('Power Regression')
plt.legend()
plt.show()

# Calculate RMS error for Power Regression
rms_error_power = np.sqrt(mean_squared_error(data['Performance Index'], y_pred_power))
print(f'RMS Error (Power Regression): {rms_error_power}')
