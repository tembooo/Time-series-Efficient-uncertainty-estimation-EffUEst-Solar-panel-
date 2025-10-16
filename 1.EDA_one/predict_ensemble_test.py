# predict_ensemble_test.py
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_utils import create_sliding_windows

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Configuration
input_window = 168
output_window = 168
batch_size = 32
selected_features = [
    'air_temperature', 'diffuse_r', 'elspot', 'full_solar', 'global_r',
    'gust_speed', 'relative_humidity', 'sunshine', 'wind_speed',
    'hour', 'weekday', 'is_weekend', 'is_holiday', 'is_long_holiday',
    'avg_temperature', 'temp_lag1', 'is_peak_hour'
]

# Load Test Data
df_Test = pd.read_excel("X_test_scaled.xlsx")
X_test_full = df_Test[selected_features].to_numpy()
y_test_full = df_Test['energy'].to_numpy().reshape(-1, 1)

scaler_y = StandardScaler()
y_test_full_scaled = scaler_y.fit_transform(y_test_full)

X_test, y_test = create_sliding_windows(X_test_full, y_test_full_scaled, input_window, output_window)
X_test = X_test.reshape((-1, input_window, len(selected_features)))
y_test = y_test.reshape((-1, output_window))

# Load Ensemble Outputs
df_mu = pd.read_excel("ensemble_uncertainty.xlsx", sheet_name="Mean Prediction")
df_std = pd.read_excel("ensemble_uncertainty.xlsx", sheet_name="Uncertainty (StdDev)")
mu_pred = df_mu.drop(columns=["Week"]).to_numpy()
std_pred = df_std.drop(columns=["Week"]).to_numpy()

# Inverse Transform True Values
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

# Metrics
mae = mean_absolute_error(y_true.flatten(), mu_pred.flatten())
rmse = np.sqrt(mean_squared_error(y_true.flatten(), mu_pred.flatten()))
var_pred = std_pred ** 2
nll = np.mean(
    0.5 * np.log(var_pred + 1e-6) + 0.5 * (y_true - mu_pred)**2 / (var_pred + 1e-6)
)

print("🔍 Evaluation Metrics:")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"NLL:  {nll:.4f}")

# Save Results
df_pred = pd.DataFrame(mu_pred, columns=[f"t+{i+1}" for i in range(output_window)])
df_std = pd.DataFrame(std_pred, columns=[f"t+{i+1}" for i in range(output_window)])
df_true = pd.DataFrame(y_true, columns=[f"t+{i+1}" for i in range(output_window)])
df_pred.insert(0, "Week", range(1, len(df_pred)+1))
df_std.insert(0, "Week", range(1, len(df_std)+1))
df_true.insert(0, "Week", range(1, len(df_true)+1))

with pd.ExcelWriter("ensemble_predictions.xlsx") as writer:
    df_true.to_excel(writer, sheet_name="True Energy", index=False)
    df_pred.to_excel(writer, sheet_name="Predicted Energy", index=False)
    df_std.to_excel(writer, sheet_name="Uncertainty (StdDev)", index=False)

print("✅ Ensemble predictions saved to ensemble_predictions.xlsx")
