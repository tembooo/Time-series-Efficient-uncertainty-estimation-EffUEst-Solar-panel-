# ensemble_rlstm_uncertainty.py
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from data_utils import create_sliding_windows

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Configuration
input_window = 168
output_window = 168
batch_size = 32
num_models = 5
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

# Load Ensemble Models
ensemble_models = [
    tf.keras.models.load_model(f'top_model_{i+1}.h5', compile=False)
    for i in range(num_models)
]

# Uncertainty Estimation
print("🔹 Running: Ensemble Uncertainty Quantification")
y_pred_ensemble = np.zeros((len(X_test), output_window, num_models, 2))
for i, model in enumerate(ensemble_models):
    preds = model.predict(X_test, batch_size=batch_size, verbose=2)
    y_pred_ensemble[:, :, i, 0] = preds[:, :, 0]  # Mean
    y_pred_ensemble[:, :, i, 1] = tf.nn.softplus(preds[:, :, 1]).numpy() + 1e-6  # Variance

# Aggregate Predictions
mu_pred_scaled = np.mean(y_pred_ensemble[:, :, :, 0], axis=2)
var_pred_scaled = np.mean(y_pred_ensemble[:, :, :, 1], axis=2)
std_pred_scaled = np.sqrt(var_pred_scaled)

# Inverse Transform
mu_pred = scaler_y.inverse_transform(mu_pred_scaled.reshape(-1, 1)).reshape(mu_pred_scaled.shape)
std_pred = std_pred_scaled * scaler_y.scale_[0]
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

# Save Predictions
df_mu = pd.DataFrame(mu_pred, columns=[f"t+{i+1}" for i in range(output_window)])
df_std = pd.DataFrame(std_pred, columns=[f"t+{i+1}" for i in range(output_window)])
df_mu.insert(0, "Week", range(1, len(df_mu)+1))
df_std.insert(0, "Week", range(1, len(df_std)+1))

with pd.ExcelWriter("ensemble_uncertainty.xlsx") as writer:
    df_mu.to_excel(writer, sheet_name="Mean Prediction", index=False)
    df_std.to_excel(writer, sheet_name="Uncertainty (StdDev)", index=False)

print("✅ Uncertainty data saved to ensemble_uncertainty.xlsx")
