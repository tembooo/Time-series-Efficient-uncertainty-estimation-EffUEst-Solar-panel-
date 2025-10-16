import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, Input, callbacks
import tensorflow_probability as tfp
import optuna
import datetime
import gc
import joblib
from sklearn.preprocessing import StandardScaler
from data_utils import create_sliding_windows, create_tf_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Configuration
input_window = 168
output_window = 168
batch_size = 8  # Further reduced to prevent OOM
random_seed = 42
lambda_reg = 0.0002
selected_features = [
    'air_temperature', 'diffuse_r', 'elspot', 'full_solar', 'global_r',
    'gust_speed', 'relative_humidity', 'sunshine', 'wind_speed',
    'hour', 'weekday', 'is_weekend', 'is_holiday', 'is_long_holiday',
    'avg_temperature', 'temp_lag1', 'is_peak_hour'
]

# Load Data
df_Train = pd.read_excel("X_train_scaled.xlsx")
df_Val = pd.read_excel("X_val_scaled.xlsx")
X_train_full = df_Train[selected_features].to_numpy()
X_val_full = df_Val[selected_features].to_numpy()
y_train_full = df_Train['energy'].to_numpy().reshape(-1, 1).astype(np.float32)
y_val_full = df_Val['energy'].to_numpy().reshape(-1, 1).astype(np.float32)

# Normalize energy
scaler_y = StandardScaler()
y_train_full_scaled = scaler_y.fit_transform(y_train_full)
y_val_full_scaled = scaler_y.transform(y_val_full)
print(f"📊 Energy stats: train_mean={y_train_full.mean():.2f}, train_std={y_train_full.std():.2f}")
print(f"   Train scaled: mean={y_train_full_scaled.mean():.2f}, std={y_train_full_scaled.std():.2f}")
print(f"   Val scaled: mean={y_val_full_scaled.mean():.2f}, std={y_val_full_scaled.std():.2f}")

X_train, y_train = create_sliding_windows(X_train_full, y_train_full_scaled, input_window, output_window)
X_val, y_val = create_sliding_windows(X_val_full, y_val_full_scaled, input_window, output_window)

X_train = X_train.reshape((-1, input_window, len(selected_features)))
X_val = X_val.reshape((-1, input_window, len(selected_features)))
y_train = y_train.reshape((-1, output_window))
y_val = y_val.reshape((-1, output_window))
print(f"📊 Data shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_val={X_val.shape}, y_val={y_val.shape}")

train_dataset = create_tf_dataset(X_train, y_train, batch_size)
val_dataset = create_tf_dataset(X_val, y_val, batch_size)

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

# Model Builder
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
                          bias_initializer=tf.keras.initializers.Constant(value=[0.0, -1.0])))(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=optimizers.Adam(learning_rate),
                  loss=lambda y_true, y_pred: hybrid_loss(y_true, y_pred))
    return model

# NLL Callback for Intermediate Reporting
class NLLCallback(callbacks.Callback):
    def __init__(self, validation_data, trial):
        super().__init__()
        self.validation_data = validation_data
        self.trial = trial

    def on_epoch_end(self, epoch, logs=None):
        val_pred = self.model.predict(self.validation_data[0], batch_size=batch_size, verbose=2)
        mu = val_pred[:, :, 0]
        log_var = val_pred[:, :, 1]
        var = tf.nn.softplus(log_var) + 1e-6
        y_true = tf.convert_to_tensor(self.validation_data[1], dtype=tf.float32)
        nll = tf.reduce_mean(0.5 * (tf.math.log(var) + tf.square(y_true - mu) / var)).numpy()
        print(f"Epoch {epoch+1} - Val NLL: {nll:.4f}, Mu mean: {mu.mean():.4f}, Log_var mean: {log_var.mean():.4f}")
        self.trial.report(nll, step=epoch)
        tf.keras.backend.clear_session()
        gc.collect()
        if self.trial.should_prune():
            raise optuna.TrialPruned()

# Objective Function
def objective(trial):
    hp = {
        'lstm_units': trial.suggest_int('lstm_units', 512, 1536, step=128),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.3, step=0.05),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True),
        'seed_offset': trial.suggest_int('seed_offset', 0, 100, step=5)
    }
    print(f"🔍 Trial params: {hp}")

    model = build_rlstm_model(hp, (input_window, len(selected_features)), output_window)
    nll_callback = NLLCallback((X_val, y_val), trial)

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=50,
        callbacks=[
            nll_callback,
            callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
            callbacks.ModelCheckpoint(f'trial_{trial.number}_best.h5', monitor='val_loss', save_best_only=True, mode='min')
        ],
        verbose=2  # More detailed logging
    )

    val_pred = model.predict(X_val, batch_size=batch_size, verbose=2)
    mu = val_pred[:, :, 0]
    log_var = val_pred[:, :, 1]
    var = tf.nn.softplus(log_var) + 1e-6
    y_true = tf.convert_to_tensor(y_val, dtype=tf.float32)
    nll = tf.reduce_mean(0.5 * (tf.math.log(var) + tf.square(y_true - mu) / var)).numpy()
    print(f"🔍 Trial NLL: {nll:.4f}, Mu mean: {mu.mean():.4f}, Log_var mean: {log_var.mean():.4f}")
    tf.keras.backend.clear_session()
    gc.collect()
    return nll

# Run Optimization
print("🔹 Starting Optuna Optimization")
study_name = f"rlstm_nll_opt_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
storage = optuna.storages.RDBStorage(url="sqlite:///optuna.db")
pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
study = optuna.create_study(direction='minimize', storage=storage, study_name=study_name, load_if_exists=True, pruner=pruner)
study.optimize(objective, n_trials=50)
print("✅ Study Complete")

# Save Results
with open("best_params.txt", "w") as f:
    f.write(str(study.best_params))
with open("top_trials_params.txt", "w") as f:
    top5 = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE], key=lambda t: t.value)[:5]
    f.write(str([t.params for t in top5]))

# Save scaler_y
joblib.dump(scaler_y, "scaler_y.joblib")
