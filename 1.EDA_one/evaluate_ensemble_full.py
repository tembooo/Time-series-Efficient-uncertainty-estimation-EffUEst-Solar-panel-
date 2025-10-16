# evaluate_ensemble_full.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load Predictions
df_pred = pd.read_excel("ensemble_predictions.xlsx", sheet_name="Predicted Energy")
df_std = pd.read_excel("ensemble_predictions.xlsx", sheet_name="Uncertainty (StdDev)")
df_true = pd.read_excel("ensemble_predictions.xlsx", sheet_name="True Energy")

mu_pred = df_pred.drop(columns=["Week"]).to_numpy()
std_pred = df_std.drop(columns=["Week"]).to_numpy()
y_true = df_true.drop(columns=["Week"]).to_numpy()

mu_flat = mu_pred.reshape(-1)
std_flat = std_pred.reshape(-1)
y_true_flat = y_true.reshape(-1)

# Evaluation Metrics
nll = np.mean(0.5 * (np.log(std_flat**2 + 1e-6) + ((y_true_flat - mu_flat)**2) / (std_flat**2 + 1e-6)) + 0.5 * np.log(2 * np.pi))
mae = mean_absolute_error(y_true_flat, mu_flat)
rmse = np.sqrt(mean_squared_error(y_true_flat, mu_flat))
print(f"\n📊 Ensemble Model Evaluation:")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"NLL:  {nll:.4f}")

# Weekly Metrics
weekly_true = y_true.reshape(-1, 168).mean(axis=1)
weekly_pred = mu_pred.reshape(-1, 168).mean(axis=1)
weekly_mae = mean_absolute_error(weekly_true, weekly_pred)
weekly_rmse = np.sqrt(mean_squared_error(weekly_true, weekly_pred))
print(f"Weekly MAE:  {weekly_mae:.2f}")
print(f"Weekly RMSE: {weekly_rmse:.2f}")

# Coverage
def compute_coverage(y_true, mu_pred, std_pred, z=1.96):
    lower = mu_pred - z * std_pred
    upper = mu_pred + z * std_pred
    return np.mean((y_true >= lower) & (y_true <= upper)) * 100

coverage_95 = compute_coverage(y_true_flat, mu_flat, std_flat, z=1.96)
print(f"📈 95% Prediction Interval Coverage: {coverage_95:.2f}%")

# Plotting Function
def plot_with_uncertainty(y_true, mu, std, start_idx, end_idx, title):
    plt.figure(figsize=(20, 5))
    end_idx = min(end_idx, len(mu))
    x = np.arange(start_idx, end_idx)
    plt.plot(x, y_true[start_idx:end_idx], label='True Energy', color='blue')
    plt.plot(x, mu[start_idx:end_idx], label='Predicted Mean', color='black')
    plt.fill_between(x,
                     mu[start_idx:end_idx] - 1.96 * std[start_idx:end_idx],
                     mu[start_idx:end_idx] + 1.96 * std[start_idx:end_idx],
                     color='orange', alpha=0.3, label='95% CI')
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Energy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()

# Visualizations
plot_with_uncertainty(y_true_flat, mu_flat, std_flat, 0, 168, "Ensemble Forecast - 1 Week")
plot_with_uncertainty(y_true_flat, mu_flat, std_flat, 0, 24*30*3, "Ensemble Forecast - 3 Months")
plot_with_uncertainty(y_true_flat, mu_flat, std_flat, 0, 24*30*6, "Ensemble Forecast - 6 Months")
plot_with_uncertainty(y_true_flat, mu_flat, std_flat, 0, 24*365, "Ensemble Forecast - 1 Year")
plot_with_uncertainty(y_true_flat, mu_flat, std_flat, 0, len(mu_flat), "Ensemble Forecast - Full Test")

# Save Metrics
metrics = {
    "MAE": [mae],
    "RMSE": [rmse],
    "NLL": [nll],
    "Weekly MAE": [weekly_mae],
    "Weekly RMSE": [weekly_rmse],
    "95% Coverage": [coverage_95]
}
df_metrics = pd.DataFrame(metrics)
df_metrics.to_csv("ensemble_metrics_summary.csv", index=False)
print("✅ Evaluation results saved to ensemble_metrics_summary.csv")
