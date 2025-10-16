import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, Input, callbacks
import tensorflow_probability as tfp
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_utils import create_sliding_windows, create_tf_dataset
import ast
import gc
import joblib

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Configuration
input_window = 168
output_window = 168
batch_size = 4  # Reduced to prevent OOM
random_seed = 42
lambda_reg = 0.0002
selected_features = [
    'air_temperature', 'diffuse_r', 'elspot', 'full_solar', 'global_r',
    'gust_speed', 'relative_humidity', 'sunshine', 'wind_speed',
    'hour', 'weekday', 'is_weekend', 'is_holiday', 'is_long_holiday',
    'avg_temperature', 'temp_lag1', 'is_peak_hour'
]

# Load best parameters
print("🔹 Loading best parameters...")
with open("best_params.txt", "r") as f:
    best_params = ast.literal_eval(f.read())
print(f"📊 Best parameters: {best_params}")

# Load Data
print("🔹 Loading preprocessed data...")
df_Train = pd.read_excel("X_train_scaled.xlsx")
df_Val = pd.read_excel("X_val_scaled.xlsx")
df_Test = pd.read_excel("X_test_scaled.xlsx")
X_train_full = df_Train[selected_features].to_numpy()
X_val_full = df_Val[selected_features].to_numpy()
X_test_full = df_Test[selected_features].to_numpy()
y_train_full = df_Train['energy'].to_numpy().reshape(-1, 1).astype(np.float32)
y_val_full = df_Val['energy'].to_numpy().reshape(-1, 1).astype(np.float32)
y_test_full = df_Test['energy'].to_numpy().reshape(-1, 1).astype(np.float32)

# Normalize energy
scaler_y = StandardScaler()
y_train_full_scaled = scaler_y.fit_transform(y_train_full)
y_val_full_scaled = scaler_y.transform(y_val_full)
y_test_full_scaled = scaler_y.transform(y_test_full)
print(f"📊 Energy stats: train_mean={y_train_full.mean():.2f}, train_std={y_train_full.std():.2f}")
print(f"   Train scaled: mean={y_train_full_scaled.mean():.2f}, std={y_train_full_scaled.std():.2f}")
print(f"   Val scaled: mean={y_val_full_scaled.mean():.2f}, std={y_val_full_scaled.std():.2f}")

# Create sliding windows
X_train, y_train = create_sliding_windows(X_train_full, y_train_full_scaled, input_window, output_window)
X_val, y_val = create_sliding_windows(X_val_full, y_val_full_scaled, input_window, output_window)
X_test, y_test = create_sliding_windows(X_test_full, y_test_full_scaled, input_window, output_window)

X_train = X_train.reshape((-1, input_window, len(selected_features)))
X_val = X_val.reshape((-1, input_window, len(selected_features)))
X_test = X_test.reshape((-1, input_window, len(selected_features)))
y_train = y_train.reshape((-1, output_window))
y_val = y_val.reshape((-1, output_window))
y_test = y_test.reshape((-1, output_window))
print(f"📊 Data shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_val={X_val.shape}, y_val={y_val.shape}, X_test={X_test.shape}, y_test={y_test.shape}")

train_dataset = create_tf_dataset(X_train, y_train, batch_size)
val_dataset = create_tf_dataset(X_val, y_val, batch_size)
test_dataset = create_tf_dataset(X_test, y_test, batch_size)

# Hybrid Loss Function
@tf.function
def hybrid_loss(y_true, y_pred, alpha=0.2):
    mu = y_pred[:, :, 0]
    log_var = y_pred[:, :, 1]
    var = tf.nn.softplus(log_var) + 1e-6
    y_true = tf.cast(y_true, tf.float32)
    nll = tf.reduce_mean(0.5 * (tf.math.log(var) + tf.square(y_true - mu) / var))
    mse = tf.reduce_mean(tf.square(y_true - mu))
    peak_mask = y_true > tfp.stats.percentile(y_true, 80.0, axis=-1, keepdims=True)
    mse_peak = tf.where(peak_mask, tf.square(y_true - mu) * 1.5, tf.square(y_true - mu))
    reg = lambda_reg * tf.reduce_mean(log_var ** 2)
    tf.print("Loss components - NLL:", nll, "MSE:", mse, "MSE_peak:", tf.reduce_mean(mse_peak), "Reg:", reg)
    return alpha * nll + (1 - alpha) * tf.reduce_mean(mse_peak) + reg

# Custom Callback for Intermediate NLL Reporting
class NLLCallback(callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        val_pred = []
        for i in range(0, len(self.validation_data[0]), batch_size * 2):
            batch_pred = self.model.predict(self.validation_data[0][i:i + batch_size * 2], batch_size=batch_size, verbose=2)
            val_pred.append(batch_pred)
        val_pred = np.concatenate(val_pred, axis=0)
        mu = val_pred[:, :, 0]
        log_var = val_pred[:, :, 1]
        var = tf.nn.softplus(log_var) + 1e-6
        y_true = tf.convert_to_tensor(self.validation_data[1], dtype=tf.float32)
        nll = tf.reduce_mean(0.5 * (tf.math.log(var) + tf.square(y_true - mu) / var)).numpy()
        print(f"Epoch {epoch+1} - Val NLL: {nll:.4f}, Mu mean: {mu.mean():.4f}, Log_var mean: {log_var.mean():.4f}")
        logs['val_nll'] = nll
        tf.keras.backend.clear_session()
        gc.collect()

# Build Weekly Coverage Table
def build_weekly_coverage_table(y_true, mu_pred, std_pred, time_index, z=1.96, output_file="coverage_report.html"):
    y_true = y_true.reshape((-1, output_window))
    mu_pred = mu_pred.reshape((-1, output_window))
    std_pred = std_pred.reshape((-1, output_window))
    rows = []
    for week in range(y_true.shape[0]):
        y_w = y_true[week]
        mu_w = mu_pred[week]
        std_w = std_pred[week]
        lower = mu_w - z * std_w
        upper = mu_w + z * std_w
        within = (y_w >= lower) & (y_w <= upper)
        coverage = np.mean(within) * 100
        outside_count = np.sum(~within)
        total = len(y_w)
        date_str = str(time_index.iloc[week].date()) if week < len(time_index) else f"Week {week+1}"
        rows.append({
            "Week": week + 1,
            "Start Date": date_str,
            "Coverage (%)": f"{coverage:.2f}",
            "Outside Count": outside_count,
            "Total Points": total
        })
    df = pd.DataFrame(rows)
    html_path = output_file
    df.to_html(html_path, index=False, justify='center', border=0)
    print(f"✅ HTML coverage report saved ➜ {html_path}")

# Build RLSTM Model
def build_rlstm_model(hp, input_shape, output_window=168):
    seed_offset = hp['seed_offset']
    tf.random.set_seed(random_seed + seed_offset)
    lstm_units = hp['lstm_units']
    dropout_rate = hp['dropout_rate']
    learning_rate = hp['learning_rate']
    inputs = Input(shape=input_shape)
    x = layers.LSTM(lstm_units, return_sequences=True, dropout=dropout_rate,
                    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed_offset))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LSTM(lstm_units // 2, return_sequences=True, dropout=dropout_rate,
                    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed_offset+1))(x)
    x = layers.LSTM(lstm_units // 4, return_sequences=False, dropout=dropout_rate,
                    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed_offset+2))(x)
    x_residual = layers.Dense(lstm_units // 4, activation='relu')(x)
    x = layers.RepeatVector(output_window)(x + x_residual)
    x = layers.LSTM(lstm_units // 2, return_sequences=True, dropout=dropout_rate,
                    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed_offset+3))(x)
    x = layers.LSTM(lstm_units // 4, return_sequences=True, dropout=dropout_rate,
                    kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed_offset+4))(x)
    x = layers.TimeDistributed(layers.Dense(lstm_units // 4, activation='relu',
                     kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed_offset+5)))(x)
    outputs = layers.TimeDistributed(layers.Dense(2, kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed_offset+6),
                          bias_initializer=tf.keras.initializers.Constant(value=[0.0, -3.0])))(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate),
                  loss=lambda y_true, y_pred: hybrid_loss(y_true, y_pred))
    model.summary()
    return model

# Train Model
print("🔹 Training single RLSTM model with best hyperparameters...")
model = build_rlstm_model(best_params, (input_window, len(selected_features)), output_window)
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=200,
    callbacks=[
        NLLCallback((X_val, y_val)),
        callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, min_delta=0.001),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6),
        callbacks.ModelCheckpoint('single_rlstm_best.h5', monitor='val_loss', save_best_only=True, mode='min')
    ],
    verbose=2
)

# Define time_test for test set
time_test = pd.Series(pd.to_datetime(df_Test['datetime'])[input_window:input_window + len(y_test)])

# Evaluate Model
print("🔹 Evaluating model on test set...")
y_pred = []
for i in range(0, len(X_test), batch_size * 2):
    batch_pred = model.predict(X_test[i:i + batch_size * 2], batch_size=batch_size, verbose=0)
    y_pred.append(batch_pred)
y_pred = np.concatenate(y_pred, axis=0)
mu_pred_scaled = y_pred[:, :, 0]
log_var_pred = y_pred[:, :, 1]
var_pred = tf.nn.softplus(log_var_pred).numpy() + 1e-6
std_pred_scaled = np.sqrt(var_pred)
mu_pred = scaler_y.inverse_transform(mu_pred_scaled.reshape(-1, 1)).reshape(mu_pred_scaled.shape)
std_pred = std_pred_scaled * scaler_y.scale_[0]
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

# Compute metrics
total_var = std_pred ** 2
nll = np.mean(0.5 * (np.log(total_var.flatten()) + (y_true.flatten() - mu_pred.flatten())**2 / total_var.flatten()) + 0.5 * np.log(2 * np.pi))
mae = mean_absolute_error(y_true.flatten(), mu_pred.flatten())
rmse = np.sqrt(mean_squared_error(y_true.flatten(), mu_pred.flatten()))
print(f"\n📊 Single RLSTM Model Evaluation:")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"NLL:  {nll:.4f}")

# Generate Coverage Report
print("🔹 Generating coverage report...")
build_weekly_coverage_table(
    y_true=y_true,
    mu_pred=mu_pred,
    std_pred=std_pred,
    time_index=time_test,
    output_file="coverage_report.html"
)

# Save predictions
df_true = pd.DataFrame(y_true, columns=[f"t+{i+1}" for i in range(output_window)])
df_pred = pd.DataFrame(mu_pred, columns=[f"t+{i+1}" for i in range(output_window)])
df_std = pd.DataFrame(std_pred, columns=[f"t+{i+1}" for i in range(output_window)])
df_true.insert(0, "Week", range(1, len(df_true)+1))
df_pred.insert(0, "Week", range(1, len(df_pred)+1))
df_std.insert(0, "Week", range(1, len(df_std)+1))
with pd.ExcelWriter("single_rlstm_predictions.xlsx") as writer:
    df_true.to_excel(writer, sheet_name="True Energy", index=False)
    df_pred.to_excel(writer, sheet_name="Predicted Energy", index=False)
    df_std.to_excel(writer, sheet_name="Uncertainty (StdDev)", index=False)
print("✅ Saved single_rlstm_predictions.xlsx")

# Save scaler_y
joblib.dump(scaler_y, "scaler_y.joblib")
print("✅ Saved scaler_y.joblib")

# Clean up
tf.keras.backend.clear_session()
gc.collect()
print("✅ Single RLSTM training complete")
