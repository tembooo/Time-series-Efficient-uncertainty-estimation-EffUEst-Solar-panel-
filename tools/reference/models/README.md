#  how many models that would be the best choice --- week14


| Model                    | MAE   | RMSE  | NLL    | Training Time |
| ------------------------ | ----- | ----- | ------ | ------------- |
| **Single RLSTM**         | 17.32 | 26.48 | 4.9123 | 9 hours       |
| **Ensemble (3 models)**  | 15.79 | 24.87 | 4.6071 | 1 day         |
| **Ensemble (5 models)**  | 15.54 | 24.41 | 4.4977 | 2 days        |
| **Ensemble (7 models)**  | 15.38 | 23.96 | 4.5219 | 3 days        |
| **Ensemble (10 models)** | 15.47 | 23.95 | 4.5148 | 4 days        |


- Moving from Single → Ensemble improves MAE and RMSE.
- Best RMSE and MAE: 7 models (MAE = 15.38, RMSE = 23.96).
- Best NLL: 5 models (4.4977).

After 7 models, improvements are marginal while computational cost increases.

#  RLSTM Ensemble Pipeline with the optuna and comparision with the baseline model- --- week12
📂 Available Downloads
- 📄 [Python files ](https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models/2.EDA2.zip)
- 📄 [Optuna DB](https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models/optuna10.db) ---> study_name = "rlstm_nll_opt_20250722_022330"  ---> for investigating the optuna u can use the Dashboardfile :
- 📄 [Dashboard](https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models/Dashboard.ipynb) 
###### Table: Inputs, Outputs, and Purpose of Each Script
for running the all of the parts we need to run the " pipeline.py". 

| **Script Name**                 | **Inputs**                                                                                | **Outputs**                                                                              | **Purpose (Short Description)**                                                                |
| ------------------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `preprocessing.py`              | Raw data (likely Parquet or Excel)                                                        | `X_train_scaled.xlsx`, `X_val_scaled.xlsx`, `X_test_scaled.xlsx`                         | Cleans, engineers features, standardizes data, and saves train/val/test splits.                |
| `optuna_rlstm_train.py`         | `X_train_scaled.xlsx`, `X_val_scaled.xlsx`                                                | `optuna.db`, `best_params.txt`, `top_trials_params.txt`, `trial_*.h5`, `scaler_y.joblib` | Performs hyperparameter tuning with Optuna to minimize NLL loss for RLSTM.                     |
| `single_rlstm_train.py`         | `X_train_scaled.xlsx`, `X_val_scaled.xlsx`, `X_test_scaled.xlsx`, `best_params.txt`       | `single_rlstm_predictions.xlsx`, `coverage_report.html`, `scaler_y.joblib`               | Trains and evaluates a single RLSTM model using best Optuna hyperparameters.                   |
| `final_rlstm_model.py`          | `X_train_scaled.xlsx`, `X_val_scaled.xlsx`, `X_test_scaled.xlsx`, `top_trials_params.txt` | `final_rlstm_predictions.xlsx`, `scaler_y.joblib`, `final_rlstm_model_*.h5`              | Trains an ensemble of RLSTM models using top 5 parameter sets from Optuna.                     |
| `ensemble_rlstm_uncertainty.py` | `X_test_scaled.xlsx`, `top_model_*.h5`                                                    | `ensemble_uncertainty.xlsx`                                                              | Aggregates ensemble predictions to estimate mean and uncertainty (std dev).                    |
| `predict_ensemble_test.py`      | `ensemble_uncertainty.xlsx`, `X_test_scaled.xlsx`                                         | `ensemble_predictions.xlsx`                                                              | Evaluates ensemble predictions (MAE, RMSE, NLL) and saves results for further analysis.        |
| `evaluate_ensemble_full.py`     | `ensemble_predictions.xlsx`                                                               | `ensemble_metrics_summary.csv`, `*.png` plots                                            | Computes full evaluation metrics and visualizes prediction intervals over various time ranges. |

📊 Top 5 Trials from Optuna
after the optimization i found the six best answer and based on that i use the first one for the based line model 
| **Trial** | **NLL (Value)** | **LSTM Units** | **Dropout Rate** | **Learning Rate** | **Seed Offset** | **Used For**     |
| --------- | --------------- | -------------- | ---------------- | ----------------- | --------------- | ---------------- |
| **2**     | **-0.21876**    | 1280           | 0.10             | 4.8007e-05        | 95              | ✅ Baseline Model and Ensemble Model |
| 4         | -0.16710        | 1280           | 0.15             | 3.3056e-05        | 40              | ✅ Ensemble Model |
| 14        | -0.14946        | 1536           | 0.15             | 4.2758e-05        | 60              | ✅ Ensemble Model |
| 20        | -0.14160        | 896            | 0.15             | 5.7414e-05        | 90              | ✅ Ensemble Model |
| 3         | -0.12856        | 768            | 0.25             | 4.6894e-05        | 95              | ✅ Ensemble Model |

then based on these answers i made the ensembel model and we want to compare the "single model" to "Ensemble model" . 

✅ RLSTM Model Evaluation Summary
| **Metric**                | **Ensemble Model**                    | **Single (Baseline) Model**               |
| ------------------------- | ------------------------------------- | ----------------------------------------- |
| **MAE**                   | **16.42**                             | 19.30                                     |
| **RMSE**                  | **25.63**                             | 29.63                                     |
| **NLL**                   | **4.5765**                            | 4.8958                                    |
| **Coverage (±2 Std Dev)** | **96.05%**                            | 87.94%                                    |
| **Predictions File**      | `final_rlstm_predictions.xlsx`        | `single_rlstm_predictions.xlsx`           |
| **Uncertainty Report**    | `ensemble_rlstm_uncertainty_all.xlsx` | `single_rlstm_uncertainty_all_weeks.html` |
| **Training Type**         | Top 5 Optuna Trials (Ensemble)        | Best Optuna Trial Only                    |

and plots: 
- 📄 [single model plot](https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models/single_rlstm_uncertainty_all_weeks.html)
- 📄 [Ensamble plot](https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models/ensemble_rlstm_uncertainty_all_weeks.html)
- 📄 [plot code](https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models/plot.ipynb)
![RLSTM Vs LSTM](Images/11.jpg)



#  RLSTM Ensemble Pipeline Overview - --- week11

📂 Available Downloads
- 📄 [EDA Files](https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models/2.EDA2.zip)
- 📄 [Optuna DB](https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models/optuna2.db)
###### 🧭 Summary Table

| Step | Script Name                     | Description                             |
| ---- | ------------------------------- | --------------------------------------- |
| 1️⃣  | `run_pipeline.py`               | Master script to run the full pipeline  |
| 2️⃣  | `data_utils.py`                 | Utility functions (windows, datasets)   |
| 3️⃣  | `preprocessing.py`              | Load raw data + feature engineering     |
| 4️⃣  | `optuna_rlstm_train.py`         | Optuna tuning with hybrid loss (NLL)    |
| 5️⃣  | `final_rlstm_model.py`          | Train best model and top-5 ensemble     |
| 6️⃣  | `ensemble_rlstm_uncertainty.py` | Predict ensemble + uncertainty estimate |
| 7️⃣  | `predict_ensemble_test.py`      | Evaluate ensemble on test set           |
| 8️⃣  | `evaluate_ensemble_full.py`     | Final metrics, plots, and coverage      |


In this moment the code is still runing but i put the optuna output untill now in the below and it is in trial 3 out of 50 

![RLSTM Vs LSTM](Images/10.jpg)

1. is my Eda format for code okay ? I uploaded in the git lab? 
2. is it okay that based on loss function and the optuna process okay ? because i used hybrid for loss and nll for optimize is it true path 

##### loss function
```python 
@tf.function
def hybrid_loss(y_true, y_pred, alpha=0.2):
    mu = y_pred[:, :, 0]
    log_var = y_pred[:, :, 1]
    var = tf.nn.softplus(log_var) + 1e-6
    y_true = tf.cast(y_true, tf.float32)
    # Negative Log-Likelihood (NLL)
    nll = tf.reduce_mean(0.5 * (tf.math.log(var) + tf.square(y_true - mu) / var))
    # Mean Squared Error (MSE)
    mse = tf.reduce_mean(tf.square(y_true - mu))
    # Peak-weighted MSE (focus on higher true values)
    peak_mask = y_true > tfp.stats.percentile(y_true, 80.0, axis=-1, keepdims=True)
    mse_peak = tf.where(peak_mask, tf.square(y_true - mu) * 1.5, tf.square(y_true - mu))
    # Regularization on log variance
    reg = lambda_reg * tf.reduce_mean(log_var ** 2)
    # Print debug info
    tf.print("Loss components - NLL:", nll, "MSE:", mse, "MSE_peak:", tf.reduce_mean(mse_peak), "Reg:", reg)
    # Final hybrid loss
    return alpha * nll + (1 - alpha) * tf.reduce_mean(mse_peak) + reg

```
###### optuna part
```python 
def objective(trial):
    # Define hyperparameters
    hp = {
        'lstm_units': trial.suggest_int('lstm_units', 512, 1536, step=128),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.3, step=0.05),
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-4, log=True),
        'seed_offset': trial.suggest_int('seed_offset', 0, 100, step=5)
    }
    print(f"🔍 Trial params: {hp}")

    # Build and compile model
    model = build_rlstm_model(hp, (input_window, len(selected_features)), output_window)

    # NLL callback to report intermediate results
    nll_callback = NLLCallback((X_val, y_val), trial)

    # Train model
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
        verbose=2
    )

    # Final validation NLL computation
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


```
3. based on your previous guidness is everything suitable?
4. if there is any advice i am willing to know them? 
5. as you can see in the picture the nll on val-loss is about negative value is it okay ? 



###### 📝 Script Descriptions

🔁 `run_pipeline.py`
**Purpose**: Master controller to run all steps in order.  
**Calls**:
1. `preprocessing.py`  
2. `optuna_rlstm_train.py`  
3. `final_rlstm_model.py`  
4. `ensemble_rlstm_uncertainty.py`  
5. `predict_ensemble_test.py`  
6. `evaluate_ensemble_full.py`  

###### 🧹 `preprocessing.py`
**Purpose**: Load and engineer raw data + standardize.  
**Key Outputs**:
- `X_train_scaled.xlsx`
- `X_val_scaled.xlsx`
- `X_test_scaled.xlsx`

######  🧪 `optuna_rlstm_train.py`
**Purpose**: Hyperparameter tuning using Optuna.  
**Loss**: Hybrid (NLL + MSE_peak + Reg)  
**Saves**:
- `optuna.db`
- `best_params.txt`
- `top_trials_params.txt`
- `trial_*.h5` (checkpoints)

###### 🧠 `final_rlstm_model.py`
**Purpose**: Train best and top-5 models.  
**Saves**:
- `best_model.h5`
- `top_model_1.h5` to `top_model_5.h5`

###### 🔍 `ensemble_rlstm_uncertainty.py`
**Purpose**: Ensemble predictions + uncertainty.  
**Saves**:
- `ensemble_uncertainty.xlsx`  
  - Sheets: Mean Prediction, Uncertainty (StdDev)

###### 📊 `predict_ensemble_test.py`
**Purpose**: Compare ensemble prediction vs ground truth.  
**Metrics**:
- MAE, RMSE, NLL  
**Saves**:
- `ensemble_predictions.xlsx`

###### 📈 `evaluate_ensemble_full.py`
**Purpose**: Final evaluation + multi-horizon metrics.  
**Saves**:
- `ensemble_metrics_summary.csv`
- Multi-time-horizon plots

###### 🧰 `data_utils.py`
**Purpose**: Shared functions:
- `create_sliding_windows()`
- `create_tf_dataset()`
- `create_enhanced_features()`
- `standardize_features()`

✅ This pipeline supports full training, uncertainty modeling, and evaluation of probabilistic RLSTM ensembles.



# 16 📌 Execution Order for RLSTM Ensemble Pipeline
📄 [1.EDA files](https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models/1.EDA.zip)
📄 [Optuna db](https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models/optuna.db)



🧭 Summary Table

| Step | Script Name                     | Description                             |
| ---- | ------------------------------- | --------------------------------------- |
| 1️⃣  | `run_pipeline.py`               | Master script to run the full pipeline  |
| 2️⃣  | `data_utils.py`                 | Utility functions (windows, datasets)   |
| 3️⃣  | `preprocessing.py`              | Load raw data + feature engineering     |
| 4️⃣  | `optuna_rlstm_train.py`         | Optuna tuning with hybrid loss (NLL)    |
| 5️⃣  | `final_rlstm_model.py`          | Train best model and top-5 ensemble     |
| 6️⃣  | `ensemble_rlstm_uncertainty.py` | Predict ensemble + uncertainty estimate |
| 7️⃣  | `predict_ensemble_test.py`      | Evaluate ensemble on test set           |
| 8️⃣  | `evaluate_ensemble_full.py`     | Final metrics, plots, and coverage      |



 ###### 🔁 `run_pipeline.py`
**Purpose**: Master controller to execute the entire workflow step by step.

**Responsibilities**:
- Runs each of the following scripts in sequence:
  1. `preprocessing.py`
  2. `optuna_rlstm_train.py`
  3. `final_rlstm_model.py`
  4. `ensemble_rlstm_uncertainty.py`
  5. `predict_ensemble_test.py`
  6. `evaluate_ensemble_full.py`
- Streams console output and handles errors gracefully.



###### 🧹 `preprocessing.py`
**Purpose**: Prepare raw data and perform feature engineering + standardization.

**Key Tasks**:
- Reads input datasets (`.xlsx`, `.parquet`, etc.)
- Adds engineered features:
  - `hour`, `weekday`, `is_weekend`, `is_holiday`, `avg_temperature`, `temp_lag1`, etc.
- Fills missing values and standardizes selected features.
- Saves:
  - `X_train_scaled.xlsx`
  - `X_val_scaled.xlsx`
  - `X_test_scaled.xlsx`



###### 🧪 `optuna_rlstm_train.py`
**Purpose**: Run Optuna hyperparameter tuning using hybrid loss (NLL + MSE).

**Key Tasks**:
- Defines custom hybrid loss for probabilistic forecasting.
- Builds and trains RLSTM model with parameters:
  - `lstm_units`, `dropout_rate`, `learning_rate`, `seed_offset`
- Uses Optuna with pruning based on validation NLL.
- Saves:
  - Study to `optuna.db`
  - Best params to `best_params.txt`
  - Top 5 params to `top_trials_params.txt`



###### 🧠 `final_rlstm_model.py`
**Purpose**: Train the final model and top-5 ensemble models based on Optuna results.

**Key Tasks**:
- Loads `best_params.txt` and trains a single best model → saves as `best_model.h5`
- Loads `top_trials_params.txt` and trains 5 ensemble models with varying seeds → saves as `top_model_1.h5` to `top_model_5.h5`



###### 🔍 `ensemble_rlstm_uncertainty.py`
**Purpose**: Run ensemble inference and estimate predictive uncertainty.

**Key Tasks**:
- Loads 5 saved ensemble models.
- Predicts mean and variance for each.
- Averages across ensemble:
  - Final mean = average of predicted means
  - Final variance = average of predicted variances
- Saves results to:
  - `ensemble_uncertainty.xlsx` with:
    - `Mean Prediction` sheet
    - `Uncertainty (StdDev)` sheet



###### 📊 `predict_ensemble_test.py`
**Purpose**: Evaluate the ensemble predictions against test set ground truth.

**Key Tasks**:
- Loads:
  - `ensemble_uncertainty.xlsx`
  - `X_test_scaled.xlsx`
- Inverse transforms predictions and targets.
- Computes metrics:
  - **MAE**, **RMSE**, **NLL**
- Saves results to:
  - `ensemble_predictions.xlsx`



######  📈 `evaluate_ensemble_full.py`
**Purpose**: Perform final evaluation and uncertainty visualization.

**Key Tasks**:
- Loads predictions from `ensemble_predictions.xlsx`
- Computes:
  - MAE, RMSE, NLL
  - Weekly MAE, Weekly RMSE
  - 95% coverage of prediction intervals
- Plots:
  - 1-week forecast with 95% confidence interval
  - 3-month forecast with 95% CI
- Saves:
  - `ensemble_metrics_summary.csv`



###### 🧰 `data_utils.py`
**Purpose**: Contains shared utility functions used across scripts.

**Key Functions**:
- `create_sliding_windows()`: Generates input-output pairs for time series.
- `create_tf_dataset()`: Converts NumPy arrays into TensorFlow `tf.data.Dataset`.
- `create_enhanced_features()`: Adds time/holiday/lag features to DataFrames.
- `standardize_features()`: Standardizes train/val/test splits using `StandardScaler`.


###### 📁 Key Output Files

| File                          | Description                                      |
|-------------------------------|--------------------------------------------------|
| `optuna.db`                   | Optuna study history (for dashboard)             |
| `best_params.txt`            | Best hyperparameters (from Optuna)               |
| `top_trials_params.txt`      | Top 5 hyperparameter sets for ensemble           |
| `best_model.h5`              | Single best RLSTM model                          |
| `top_model_1.h5` to `_5.h5`  | Ensemble RLSTM models                            |
| `ensemble_uncertainty.xlsx` | Mean predictions and uncertainty (std dev)       |
| `ensemble_predictions.xlsx` | True vs predicted values + uncertainty           |
| `ensemble_metrics_summary.csv` | Final evaluation metrics                       |



✅ **Ready to use!** This full pipeline supports training, uncertainty estimation, and model evaluation for probabilistic forecasting using ensemble RLSTM models.


# 15. Optimization Based on MAE – Code 15 (Ensemble) - --- week10

This week, I used **Optuna** to optimize the MAE and build an **ensemble model** based on the best hyperparameters.

**Code Reference:**
📄 [Code 15 – RLSTM Ensemble Model (Optimized)](https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models/15_RLSTM_one_Week_with_best_model_optimization_tf_keras_update_optuna_5_essemble.ipynb)

### 🔗 Output Visualizations

* 📊 [Weekly Forecast Plot](https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models/plot2weekytestcode15.html)
* 📊 [Hourly Forecast Plot](https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models/plot2hourstestcode15.html)
### 🔗 Pure train data
* 📊 [Weekly Train](https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models/plot2weeky_train_true_only.html)
* 📊 [Hourly Train](https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models/plot_train_true__hour_energy.html)

### ✅ Best Hyperparameters (Found by Optuna)

| Hyperparameter  | Value            |
| --------------- | ---------------- |
| `lstm_units`    | **1792**         |
| `dropout_rate`  | **0.15**         |
| `learning_rate` | **0.0003058621** |
| `seed_offset`   | **20**           |



| **Hyperparameter / Setting**          | **Code 1 – RLSTM – Version A**             | **Code 2 – RLSTM – Version B**         | **Code 3 – RLSTM – Optimized Model 1**      | **Code 4 – RLSTM – Optuna Final Ensemble**      |
| ------------------------------------- | ------------------------------------------ | -------------------------------------- | ------------------------------------------- | ----------------------------------------------- |
| `window_size`                         | 336                                        | 336                                    | 168                                         | 168                                             |
| `output_window`                       | 168                                        | 168                                    | 168                                         | 168                                             |
| `n_ensemble`                          | 5                                          | 5                                      | 1 (single model)                            | 5                                               |
| `epochs`                              | 150                                        | 150                                    | 200                                         | 200                                             |
| `batch_size`                          | 16                                         | 16                                     | 32                                          | 32                                              |
| `learning_rate`                       | 0.0001                                     | 0.0001                                 | **9.13e-5**                                 | ✅ **3.06e-4** (from Optuna)                     |
| `dropout_rate`                        | 0.1                                        | 0.1                                    | **0.15**                                    | ✅ **0.15**                                      |
| **LSTM architecture**                 | Encoder: 512→256→128<br>Decoder: 256→128 ⭐ | Encoder: 168→84→42<br>Decoder: 84→42 ⭐ | Encoder: 1536→768→384<br>Decoder: 768→384 ⭐ | ✅ Encoder: 1792→896→448<br>Decoder: 896→448 ⭐   |
| `dense_units`                         | 100                                        | 100                                    | 384 (derived from LSTM)                     | 448 (from LSTM units)                           |
| `lstm_units_1`                        | 512 ⭐                                      | 168 ⭐                                  | **1536 ⭐**                                  | ✅ **1792 ⭐**                                    |
| `lstm_units_2`                        | 256 → 128 ⭐                                | 84 → 42 ⭐                              | 768 → 384 ⭐                                 | ✅ 896 → 448 ⭐                                   |
| `output_layer bias init`              | \[0.0, -3.0]                               | \[0.0, -3.0]                           | \[0.0, -3.0]                                | \[0.0, -3.0]                                    |
| **MC Dropout at test time**           | ❌ No ⭐                                     | ✅ Yes ⭐                                | ❌ No                                        | ❌ No                                            |
| **Total variance in ensemble**        | ✅ Yes                                      | ✅ Yes                                  | ✅ Yes (from `log_var`)                      | ✅ Yes (from `log_var`)                          |
| **NLL regularization (`lambda_reg`)** | ✅ λ = 0.0003                               | ✅ λ = 0.0003                           | ✅ **λ = 0.0002**                            | ✅ **λ = 0.0002**                                |
| **Evaluation: NLL calc method**       | Total Variance (no MC) ⭐                   | MC Dropout + Total Variance ⭐          | Total Variance (no MC) ⭐                    | Total Variance (no MC) ⭐                        |
| **Loss function**                     | Hybrid: α=0.3 NLL + MSE + Reg              | Hybrid: α=0.3 NLL + MSE + Reg          | Hybrid: **α=0.2 NLL + MSE + Reg**           | ✅ Hybrid: α=0.2 NLL + MSE + Reg                 |
| `variance scaling in plot`            | 0.8                                        | 0.8                                    | 1.0                                         | ✅ 1.0                                           |
| `random seed base`                    | 42                                         | 42                                     | 42                                          | 42                                              |
| `MC Samples`                          | 10                                         | 10                                     | Not used (❌ MC dropout)                     | ❌ Not used (MC dropout disabled)                |
| `GPU used`                            | GPU 1 ⭐                                    | GPU 0 ⭐                                | GPU 0                                       | ✅ GPU 0 (Tesla V100 32GB)                       |
| **MAE (example output)**              | \~27.28 ⭐                                  | \~28.08 ⭐                              | ✅ **18.05**                                 | ✅ **17.43**                                     |
| **RMSE (example output)**             | \~37.60 ⭐                                  | \~26.13 ⭐                              | ✅ **27.59**                                 | ✅ **26.36**                                     |
| **NLL (example output)**              | \~4.81 ⭐                                   | \~4.98 ⭐                               | ✅ **4.6438**                                | ✅ **92344.89** 🔺 (due to std \~ 1e-3 constant) |
| **Coverage (95%)**                    | \~96.62% ⭐                                 | \~97.01% ⭐                             | ✅ **96.53%**                                | ✅ **96.53%**                                    |
| **Training time per model**           | \~570 min                                  | \~570 min                              | \~100 min (1 model)                         | ✅ \~16 min/model × 5 = **80 min**               |









---

# 14. RLSTM for One-Week Prediction Horizon – Code 3 (Single Model) --- week9

During week 9, I worked on **Code 3**, which implements an **optimized single RLSTM model**.

**Code Reference:**
📄 [Code 3 – RLSTM Optimized Model (Single)](https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models14_RLSTM_one_Week_with_best_model_optimization_tf_keras_update_optuna.ipynb)

This model is **not an ensemble**. When I attempted to build ensemble models based on this architecture, the performance degraded significantly compared to previous ensemble implementations.

---

## 📊 Comparison of Models (Code 1, 2, and 3)

| **Hyperparameter / Setting**          | **Code 1 – RLSTM – Version A**             | **Code 2 – RLSTM – Version B**         | **Code 3 – RLSTM – Optimized Model 1**      |
| ------------------------------------- | ------------------------------------------ | -------------------------------------- | ------------------------------------------- |
| `window_size`                         | 336                                        | 336                                    | 168                                         |
| `output_window`                       | 168                                        | 168                                    | 168                                         |
| `n_ensemble`                          | 5                                          | 5                                      | 1 (single model)                            |
| `epochs`                              | 150                                        | 150                                    | 200                                         |
| `batch_size`                          | 16                                         | 16                                     | 32                                          |
| `learning_rate`                       | 0.0001                                     | 0.0001                                 | **9.13e-5**                                 |
| `dropout_rate`                        | 0.1                                        | 0.1                                    | **0.15**                                    |
| **LSTM architecture**                 | Encoder: 512→256→128<br>Decoder: 256→128 ⭐ | Encoder: 168→84→42<br>Decoder: 84→42 ⭐ | Encoder: 1536→768→384<br>Decoder: 768→384 ⭐ |
| `dense_units`                         | 100                                        | 100                                    | 384 (derived from LSTM)                     |
| `lstm_units_1`                        | 512 ⭐                                      | 168 ⭐                                  | **1536 ⭐**                                  |
| `lstm_units_2`                        | 256 → 128 ⭐                                | 84 → 42 ⭐                              | 768 → 384 ⭐                                 |
| `output_layer bias init`              | \[0.0, -3.0]                               | \[0.0, -3.0]                           | \[0.0, -3.0]                                |
| **MC Dropout at test time**           | ❌ No ⭐                                     | ✅ Yes ⭐                                | ❌ No                                        |
| **Total variance in ensemble**        | ✅ Yes                                      | ✅ Yes                                  | ✅ Yes (from `log_var`)                      |
| **NLL regularization (`lambda_reg`)** | ✅ λ = 0.0003                               | ✅ λ = 0.0003                           | ✅ **λ = 0.0002**                            |
| **Evaluation: NLL calc method**       | Total Variance (no MC) ⭐                   | MC Dropout + Total Variance ⭐          | Total Variance (no MC) ⭐                    |
| **Loss function**                     | Hybrid: α=0.3 NLL + MSE + Reg              | Hybrid: α=0.3 NLL + MSE + Reg          | Hybrid: **α=0.2 NLL + MSE + Reg**           |
| `variance scaling in plot`            | 0.8                                        | 0.8                                    | 1.0                                         |
| `random seed base`                    | 42                                         | 42                                     | 42                                          |
| `MC Samples`                          | 10                                         | 10                                     | Not used (❌ MC dropout)                     |
| `GPU used`                            | GPU 1 ⭐                                    | GPU 0 ⭐                                | GPU 0                                       |
| **MAE (example output)**              | \~27.28 ⭐                                  | \~28.08 ⭐                              | ✅ **18.05**                                 |
| **RMSE (example output)**             | \~37.60 ⭐                                  | \~26.13 ⭐                              | ✅ **27.59**                                 |
| **NLL (example output)**              | \~4.81 ⭐                                   | \~4.98 ⭐                               | ✅ **4.6438**                                |
| **Coverage (95%)**                    | \~96.62% ⭐                                 | \~97.01% ⭐                             | ✅ **96.53%**                                |
| **Training time per model**           | \~570 min                                  | \~570 min                              | \~100 min (1 model)                         |























## 12,13. RLSTM for one week prediction horizon  --- week8


[Code 1 – RLSTM - one week – Version A](https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models/12.RLSTM-one-Week.ipynb)

[Code 2 – RLSTM- one week  – Version B](https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models/13.RLSTM-one-Week.ipynb)

| **Hyperparameter / Setting**         | **Code 1 – RLSTM – Version A**             | **Code 2 – RLSTM – Version B**         |
| ------------------------------------ | ------------------------------------------ | -------------------------------------- |
| **window\_size**                     | 336                                        | 336                                    |
| **output\_window**                   | 168                                        | 168                                    |
| **n\_ensemble**                      | 5                                          | 5                                      |
| **epochs**                           | 150                                        | 150                                    |
| **batch\_size**                      | 16                                         | 16                                     |
| **learning\_rate**                   | 0.0001                                     | 0.0001                                 |
| **dropout\_rate**                    | 0.1                                        | 0.1                                    |
| **LSTM architecture**                | Encoder: 512→256→128<br>Decoder: 256→128 ⭐ | Encoder: 168→84→42<br>Decoder: 84→42 ⭐ |
| **dense\_units**                     | 100                                        | 100                                    |
| **lstm\_units\_1**                   | 512 ⭐                                      | 168 ⭐                                  |
| **lstm\_units\_2**                   | 256 → 128 ⭐                                | 84 → 42 ⭐                              |
| **output\_layer bias init**          | \[0.0, -3.0]                               | \[0.0, -3.0]                           |
| **MC Dropout at test time**          | ❌ No ⭐                                     | ✅ Yes ⭐                                |
| **Total variance in ensemble**       | ✅ Yes                                      | ✅ Yes                                  |
| **NLL regularization (lambda\_reg)** | ✅ `lambda_reg = 0.0003`                    | ✅ `lambda_reg = 0.0003`                |
| **Evaluation: NLL calc method**      | Total Variance (no MC) ⭐                   | MC Dropout + Total Variance ⭐          |
| **Loss function**                    | Hybrid: `alpha=0.3` NLL + MSE + Reg        | Hybrid: `alpha=0.3` NLL + MSE + Reg    |
| **Variance scaling in plot**         | 0.8                                        | 0.8                                    |
| **Random seed base**                 | 42                                         | 42                                     |
| **MC Samples**                       | 10                                         | 10                                     |
| **GPU used**                         | GPU 1 ⭐                                    | GPU 0 ⭐                                |
| **MAE** (example output)             | \~27.28 ⭐                                  | \~28.08 ⭐                              |
| **RMSE** (example output)            | \~37.60 ⭐                                  | \~26.13 ⭐                              |
| **NLL** (example output)             | \~4.81 ⭐                                   | \~4.98 ⭐                               |
| **Coverage (95%)** (example output)  | \~96.62% ⭐                                 | \~97.01% ⭐                             |
| **Training time per model**          | \~570 min                                   | \~570 min                                |









My Questions: 
Q1: The RLSTM model shows a flat predicted mean for a 7-day forecast. Could the current architecture (256 LSTM units, 2 layers) be insufficient for capturing the temporal patterns in energy data over 168 hours? Would increasing the number of layers or units help?

Q2: We’re using a 336-hour input window to predict 168 hours of energy data. Is this input length adequate, or should we consider a longer historical context (for example 21 days) to improve forecasting accuracy?

Q3: The NLL loss with lambda_reg=0.001 results in a high NLL (5.41) and a flat mean. Could the regularization be over-penalizing variance, and would adjusting it or using a hybrid loss (MSE + NLL) improve the mean prediction?

Q4: Given the poor results, would a Transformer or attention-based model be more suitable for multi-step energy forecasting compared to RLSTM or we should move back to encode decoder structure?


My Guess about it is The 336-hour (14-day) input window might not provide enough historical context to predict 168 hours accurately, or the features might lack predictive power for such a long horizon.  And The NLL loss with regularization (lambda_reg=0.001) might prioritize variance over mean accuracy, leading to a flat mean with wide uncertainty.
And on optuna We face memory issues also so how we should deal with them?

## 11. RLSTM for one week prediction horizon  --- week7 
Now I tried to implement the RLSTM for one week but the result is not reliable, I think the sliding windows is not suitable However I used the input_window = 336  # Extended to 14 days for better context but the result is not acceptable. 

[11.RLSTM_for_one-week.ipynb ]( https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models/11.RLSTM-one-Week.ipynb )




#### Initial Phase (Epochs 1–5)
- `val_loss` decreases steadily.
- The model is learning and generalizing well to the validation data.

#### Mid Phase (Epochs ~6–15)
- `val_loss` begins to increase or fluctuate, while training loss continues to decrease.
- This indicates overfitting: the model is starting to memorize training data rather than generalizing.

#### Later Phase (Epochs >15)
- `val_loss` spikes again (e.g., 0.6609, 0.6852), even though training loss remains low.
- This is a sign of strong overfitting.
- The learning rate is reduced (e.g., from `1e-4` to `5e-5`), likely due to the `ReduceLROnPlateau` callback.

---

### Solutions to Mitigate Overfitting

1. **Use EarlyStopping**
   - Apply `EarlyStopping(monitor='val_loss', patience=5 or 10)` to stop training before overfitting worsens.

2. **Tune ReduceLROnPlateau**
   - Lower the `patience` parameter so the learning rate reduces earlier.
   - This helps the model converge more smoothly and avoid sharp increases in `val_loss`.

3. **Increase Dropout Rate**
   - Add or increase dropout in LSTM or dense layers (e.g., `dropout=0.3`) to regularize the model.

4. **Apply K-Fold Cross-Validation**
   - This ensures the model generalizes well across multiple subsets of the validation data.

5. **Adjust NLL Regularization (`lambda_reg`)**
   - If the model becomes overconfident while `val_loss` increases, slightly increase `lambda_reg` to penalize this.






## 10. comparison LSTM and RLSTM for one hour prediction horizon  --- week7 

The **RLSTM** architecture, as shown in the image, extends the standard **LSTM** by introducing additional **randomized gates** and pathways to enhance learning diversity. While the traditional **LSTM** uses three gates—**input (iₜ)**, **forget (fₜ)**, and **output (oₜ)**—to control information flow, the **RLSTM** incorporates **reset (rₜ)** and **write (wₜ)** gates, introducing stochastic behavior into memory updates. Unlike the deterministic cell state update in LSTM, **RLSTM** computes a modified candidate memory **C′ₜ** using gated randomness, which is then combined with past memory for the final **Cₜ**. This stochasticity enables **RLSTM** to act like an implicit ensemble, promoting **diversity**, **regularization**, and **robust uncertainty estimation**, which are crucial in your ensemble forecasting tasks.


[10.RLSTM.ipynb ]( https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models/10.RLSTM.ipynb )

Based on this article:

[A novel Encoder-Decoder model based on read-first LSTM for air pollutant prediction ]( https://www.sciencedirect.com/science/article/abs/pii/S0048969720380384?via%3Dihub)


I think the next step for 1 hours that would be forcasting is Optimization by Optuna. 


![RLSTM Vs LSTM](Images/9.jpg)
Hyperparameter / Setting           | Version 1 –– week4 | Version 2 –– week5 | Version 3 –– week5 | Version 4 –– week6 (Optuna tuned) | RLSTM – Version 5 –– week6
-----------------------------------|--------------------|--------------------|--------------------|-----------------------------------|------------------------------
window_size                        | 336                | 336                | 168                | 168                               | 168
n_ensemble                         | 7                  | 7                  | 5                  | 5                                 | 5
epochs                             | 50                 | 50                 | 50                 | 50                                | 50
batch_size                         | 32                 | 32                 | 32                 | 32                                | 32
learning_rate                      | 0.0005             | 0.0005             | 0.0005             | 8.575e-5                          | 8.575e-5
dropout_rate                       | 0.3                | None               | None               | None                              | 0.2
LSTM architecture                  | 2 stacked LSTMs + Dropout | 2 stacked LSTMs | Encoder-Decoder LSTM | Encoder RLSTM -Decoder LSTM         | Encoder-Decoder LSTM + Dropout
dense_units                        | 20                 | 20                 | 20                 | 25                                | 25
lstm_units_1                       | 100                | 100                | 100                | 125                               | 125
lstm_units_2                       | 50                 | 50                 | 50 → decoder(50→25) | 62 → decoder(62→31) (approx.)    | 62 → decoder(62→31)
output_layer bias init             | [0.0, -2.0]        | [0.0, -2.0]        | [0.0, -3.0]        | [0.0, -3.0]                       | [0.0, -3.0]
MC Dropout                         | ✅ Yes (n_iter=20) | ❌ No              | ❌ No              | ❌ No                             | ❌ No ✅ train= 0.2
Total variance in ensemble         | ❌ No              | ✅ Yes             | ✅ Yes             | ✅ Yes                            | ✅ Yes
NLL regularization                 | ❌ No              | ❌ No              | ✅ lambda_reg=0.01 | ✅ lambda_reg=0.001415            | ✅ lambda_reg=0.001415
Evaluation: NLL calc method        | MC Dropout average | Manual from variance | Manual from variance | Manual from variance          | Manual from variance
MAE                                | 20.00              | 37.13              | 31.91              | 24.02                             | **18.74**
RMSE                               | 28.06              | 51.65              | 45.33              | 33.64                             | **26.13**
NLL                                | 433.97             | 5.15               | 5.06               | 4.72                              | **4.58**
Time                               | 1H:30M             | 35:00M             | 28M                | ~28M                              | ~28M


## 9. Normal LSTM encoder-decoder for one week prediction horizon  --- week7 
it took 32 hours. but the result was not suitable.

I think it because of sliding windows, since the sliding window and prediction horizon is as same as each other. 

Additionally I think the importance of the previous

[9.Optimization_for_one_week.ipynb ]( https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models/9.Optimization_for_one_week.ipynb )

| Hyperparameter | Description                               | Search Space             | Sampling Strategy         |
| -------------- | ----------------------------------------- | ------------------------ | ------------------------- |
| `lstm_units_1` | First encoder LSTM layer units            | 10 to 200 (log scale)    | `suggest_int(log=True)`   |
| `lstm_units_2` | Second encoder LSTM layer units           | 10 to 200 (log scale)    | `suggest_int(log=True)`   |
| `lstm_units_3` | First decoder LSTM layer units            | 10 to 200 (log scale)    | `suggest_int(log=True)`   |
| `lstm_units_4` | Second decoder LSTM layer units           | 10 to 200 (log scale)    | `suggest_int(log=True)`   |
| `dense_units`  | Fully connected dense layer after decoder | 10 to 50 (log scale)     | `suggest_int(log=True)`   |
| `lambda_reg`   | NLL regularization strength               | 1e-4 to 1e-1 (log scale) | `suggest_float(log=True)` |

📊 **Final Evaluation (Optimized Model)**  
**MAE**: 46.73  
**RMSE**: 58.72  
**NLL**: 5.48  

**Baseline MAE (7-day avg)**: 52.29  
**Baseline RMSE (7-day avg)**: 59.62  

but the plot is not suitable.

## 8. comparison between Baseline model and ensamble LSTM  --- week6

Related file: [Week8_link](https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models/8.Simple_lstm_with_optimization.ipynb
)


0. SSH for external source
- VS code is connected to the university resource.
1. Baseline Comparison
- Build a simple baseline model (e.g., weekly average).
- Compare with LSTM model to validate improvement.
2. Improve Output Quality
- Analyze uncertainty outputs and NLL behavior.
- Refine output strategy (e.g., test square instead of softplus). ----> softplus
3. Optimization Preparation
- Refactor code into reusable functions.
- Convert Jupyter code to script for long runs.
- Implement Optuna.
**- Set up storage backend (e.g., RDB).**
4. Main Goals for the Week
- Identify and tune key hyperparameters.
- Automate parameter optimization.


#### Baseline model 
| **Aspect**                | **Description**                                                          |
| ------------------------- | ----------------------------------------------------------------------- |
|  **Model Type**         | Simple 7-day average model                                               |
| **Input Window**        | 168 hours (past 7 days)                                                  |
| **Forecast Horizon**   | 168 hours (next 7 days)                                                  |
|  **Forecasting Method** | For each hour in the forecast: use the average of the previous 168 hours |
|  **Formula**            | `forecast_t = mean(y[t-168 : t])`                                        |
|  **Purpose**            | Serve as a baseline for comparison with advanced models like LSTM        |
|  **Evaluation Metrics** | - MAE (Mean Absolute Error): **52.29**                                   |




#### Comparison between models
| Feature                      | **previous**                                                                                                            | **New model**                                                                                                     |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **Output Type**              | Single-step (predicts 1 time step ahead)                                                                               | Multi-step (predicts full 168-hour sequence ahead)                                                              |
| **LSTM Structure**           | - LSTM(100, return\_seq=True) <br> - LSTM(50, return\_seq=False) <br> - Decoder: RepeatVector(1) → LSTM(50) → LSTM(25) | - LSTM(125, return\_seq=True) <br> - LSTM(62, return\_seq=False) <br> - RepeatVector(168) → LSTM(62) → LSTM(31) |
| **Decoder Type**             | Decoder for **single output step**                                                                                     | Decoder for **sequence output** using `TimeDistributed(Dense(2))`                                               |
| **Loss Function**            | NLL loss using `[mu, log_var]` (with `softplus`) + L2 reg on log\_var                                                  | NLL loss using `[mu, std²]` directly + L2 reg on std²                                                           |
| **Training Target**          | Predict a **single value with uncertainty**                                                                            | Predict the **entire sequence with uncertainty**                                                                |
| **Prediction Runtime**       | Faster (only 1 step prediction)                                                                                        | Heavier but more informative (full sequence forecast)                                                           |



### Optuna Hyperparameter Optimization Results --> 640 minutes

| Trial | LSTM Units | Learning Rate         | Dense Units | Lambda Reg         | MAE Value         | Status   |
|-------|------------|------------------------|-------------|--------------------|-------------------|----------|
| 0     | 150        | 4.30e-05               | 20          | 0.01381            | 51.93             | Completed |
| 1     | 75         | 4.06e-04               | 15          | 0.00177            | 52.31             | Completed |
| 2     | 125        | 8.58e-05               | 25          | 0.00141            | **48.69** ✅      | **Best** |
| 3     | 150        | 1.56e-05               | 15          | 0.00338            | 50.69             | Completed |
| 4     | 150        | 3.38e-05               | 10          | 0.00583            | 51.18             | Completed |
| 5     | 50         | 4.05e-04               | 10          | 0.00246            | 51.41             | Completed |
| 6     | 75         | 3.87e-05               | 25          | 0.00483            | 51.11             | Completed |
| 7     | 125        | 5.11e-05               | 10          | 0.03837            | 50.28             | Completed |
| 8     | 75         | 3.92e-05               | 10          | 0.02025            | 49.97             | Completed |
| 9     | 125        | 1.78e-04               | 25          | 0.00174            | 54.80             | Completed |

### Findings
| Hyperparameter / Setting        | Version 1 –– week4                | Version 2 –– week5              | Version 3 –– week5                    | Version 4 –– week6 (Optuna tuned)     |
|--------------------------------|----------------------------------|--------------------------------|--------------------------------------|----------------------------------------|
| window_size                    | 336                              | 336                            | 168                                  | 168                                    |
| n_ensemble                     | 7                                | 7                              | 5                                    | 5                                      |
| epochs                         | 50                               | 50                             | 50                                   | 50                                     |
| batch_size                     | 32                               | 32                             | 32                                   | 32                                     |
| learning_rate                  | 0.0005                           | 0.0005                         | 0.0005                               | 8.575e-5                               |
| dropout_rate                   | 0.3                              | None                           | None                                 | None                                   |
| LSTM architecture              | 2 stacked LSTMs + Dropout        | 2 stacked LSTMs                | Encoder-Decoder LSTM                 | Encoder-Decoder LSTM                   |
| dense_units                    | 20                               | 20                             | 20                                   | 25                                     |
| lstm_units_1                   | 100                              | 100                            | 100                                  | 125                                    |
| lstm_units_2                   | 50                               | 50                             | 50 → decoder(50→25)                  | 62 → decoder(62→31) (approx.)          |
| output_layer bias init         | [0.0, -2.0]                      | [0.0, -2.0]                    | [0.0, -3.0]                          | [0.0, -3.0]                            |
| MC Dropout                     | ✅ Yes (n_iter=20)               | ❌ No                          | ❌ No                                | ❌ No                                  |
| Total variance in ensemble     | ❌ No                            | ✅ Yes                         | ✅ Yes                               | ✅ Yes                                 |
| NLL regularization             | ❌ No                            | ❌ No                          | ✅ lambda_reg=0.01                   | ✅ lambda_reg=0.001415                 |
| Evaluation: NLL calc method    | MC Dropout average               | Manual from variance           | Manual from variance                 | Manual from variance                   |
| **MAE**                        | **20.00**                        | **37.13**                      | **31.91**                            | **24.02**                              |
| **RMSE**                       | **28.06**                        | **51.65**                      | **45.33**                            | **33.64**                              |
| **NLL**                        | **433.97**                       | **5.15**                       | **5.06**                             | **4.72**                               |
| **Time**                       | **1H:30M**                       | **35:00M**                     | **28M**                              | **~28M** (similar to V3)               |


#### Future Hyper parameters
| Name                      | Value / Setting | Type              | Description                                     |
| ------------------------- | --------------- | ----------------- | ----------------------------------------------- |
| `lstm_units`       ✅       | 125             | Hyperparameter    | Units in first LSTM encoder layer               |
| `dense_units`       ✅      | 25              | Hyperparameter    | Units in Dense layer after decoder              |
| `window_size`             | 168             | Hyperparameter    | Input time window size                          |
| `n_ensemble`              | 5               | Hyperparameter    | Number of ensemble models                       |
| `epochs`                  | 50              | Hyperparameter    | Maximum training epochs                         |
| `batch_size`              | 32              | Hyperparameter    | Batch size used during training                 |
| `learning_rate`   ✅        | 8.575e-05       | Hyperparameter    | Learning rate for Adam optimizer                |
| `lambda_reg`       ✅       | 0.001415        | Hyperparameter    | Regularization coefficient in NLL loss          |
| `encoder_1_units`         | 125             | Derived Parameter | LSTM encoder layer 1 units                      |
| `encoder_2_units`         | 62              | Derived Parameter | LSTM encoder layer 2 units (half of encoder\_1) |
| `decoder_1_units`         | 62              | Derived Parameter | LSTM decoder layer 1 units (same as encoder\_2) |
| `decoder_2_units`         | 31              | Derived Parameter | LSTM decoder layer 2 units (half of decoder\_1) |
| `repeat_vector_length`    | 1               | Parameter         | Length of repeated vector for decoder input     |
| `output_bias_initializer` | \[0.0, -3.0]    | Parameter         | Initial bias values for output layer            |
| `selected_features`       | 18 features     | Parameter         | Engineered input features for time series       |

#### My questions: 

-Based on your recommendation, I have set the forecasting horizon to 7 days. However, implementing a multi-step decoder with TimeDistributed for hourly predictions has proven challenging. Could you suggest an optimal decoder configuration?

-I have modularized the training loop for Optuna optimization. Which hyperparameters (e.g., LSTM units, learning rate) would you recommend including in the search space?

-I’ve been working on reducing the wide confidence intervals in the results. While tuning regularization helped to some extent, could you suggest more advanced techniques to narrow these intervals while keeping the NLL low?




## 7. Compare the --- LSTM Model Development Guidelines --- week5 

Related files: 

[6.Simple lstm - Version1.ipynb](https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models/6.Simple_lstm_-_Version1.ipynb)

[7.Encoder-Decoder - Version2 and 3.ipynb](https://git.it.lut.fi/datanalysis/effuest/-/blob/main/tools/reference/models/7.Simple_lstm_-_Version2_and_3.ipynb)

### 1. Architecture First, Hyperparameters Later 

* Start with moving from basic LSTM to an **Encoder-Decoder** or **Sequence-to-Sequence** architecture.
* **Redesign the network** using Encoder-Decoder LSTM structure.
* **Dropout rate** should be fixed (e.g., `0.3`) — do not tune it.
* Keep **batch size**, **epochs**, and **learning rate** fixed unless a bottleneck is observed.
* Avoid tuning **learning rate** and **epoch count** during the initial experiments.

### 2. Avoid Bayesian Optimization

* Use **Grid Search** for small search spaces.
* Use **TPE/Hyperband** for larger search spaces.
* You may use **Optuna** or **KerasTuner** for tuning, but avoid overly complex frameworks unless necessary.

### 3. Fix Performance Bottlenecks

* Avoid mixing **NumPy (CPU)** and **TensorFlow (GPU)** during **MC Dropout inference** to reduce RAM↔VRAM data transfer overhead.
* Perform all operations inside **Keras/TensorFlow** and convert to NumPy **only at the end**.

### 4. Add Adversarial Training *(Optional)*

* Consider adding **adversarial training** to improve robustness. This is supported by literature, though optional in early experiments.

### 5. Hyperparameter Tuning Strategy

* For now, manually define a **small grid** (e.g., 3–4 values per parameter).
* Use **Optuna with TPE sampler** if you want to expand tuning later.
* **Do not tune** dropout rate or batch size — treat these as fixed constants.

### Structure of the DNN 

Version1: 

![Version 1](Images/1.notron_version1.h5.png)

Version2: 

![Version 2](Images/2.notron_version2.h5.png)

Version3: 

![Version 3](Images/3.notron_version3.h5.png)

| Hyperparameter / Setting        | Version 1 --week4                        | Version 2-- week5                    | Version 3-- week5                         |
|--------------------------------|----------------------------------|-----------------------------------|-----------------------------------|
| window_size                    | 336                              | 336                               | 168                               |
| n_ensemble                     | 7                                | 7                                 | 5                                 |
| epochs                         | 50                               | 50                                | 50                                |
| batch_size                     | 32                               | 32                                | 32                                |
| learning_rate                  | 0.0005                           | 0.0005                            | 0.0005                            |
| dropout_rate                   | 0.3                              | None                              | None                              |
| LSTM architecture              | 2 stacked LSTMs + Dropout        | 2 stacked LSTMs (no dropout)      | Encoder-Decoder LSTM              |
| dense_units                    | 20                               | 20                                | 20                                |
| lstm_units_1                   | 100                              | 100                               | 100                               |
| lstm_units_2                   | 50                               | 50                                | 50 → decoder(50→25)               |
| output_layer bias init         | [0.0, -2.0]                      | [0.0, -2.0]                       | [0.0, -3.0]                       |
| MC Dropout                     | ✅ Yes (n_iter=20)               | ❌ No                             | ❌ No                             |
| Total variance in ensemble     | ❌ No                            | ✅ Yes                            | ✅ Yes                            |
| NLL regularization             | ❌ No                            | ❌ No                             | ✅ lambda_reg=0.01                |
| Evaluation: NLL calc method    | MC Dropout average               | Manual from variance              | Manual from variance              |
| Time horizon plots             | ✅ 1 week + multiple spans       | ❌ 1 week only                    | ❌ 1 week only                    |
| **MAE**                        | **20.00**                        | **37.13**                         | **31.91**                         |
| **RMSE**                       | **28.06**                        | **51.65**                         | **45.33**                         |
| **NLL**                        | **433.97**                       | **5.15**                          | **5.06**                          |
| **Time**                        | **1H:30M**                       | **35:00M**                          | **28M**                          |
#### for uncertainty estimation 


| Version   | Uncertainty Method                        | Main Technique Used                  |
| --------- | ----------------------------------------- | ------------------------------------ |
| Version 1 | ✅ **MC Dropout** (stochastic inference)   | Dropout with `training=True`         |
| Version 2 | ✅ **Total Predictive Variance**           | Deep Ensemble without Dropout        |
| Version 3 | ✅ **Total Variance + NLL Regularization** | Deep Ensemble + NLL Reg + No Dropout |



## 6.Improve the parameters and hyper parameters 
For this week we need to find: 
1. best method for uncerteintly instead of Drop out 
2. detect the parameters and hyperparameters of the model and improve them. 
3. finding the automatical way to optimize the parameters automaticly 

#### Compare the article with the code 
| **Element**                           | **In Our Code?** | **In the Article?** | **Recommendation**                                                            |
| ------------------------------------- | :---------------: | :-----------------: | ----------------------------------------------------------------------------- |
| Ensemble of independent NNs           |         ✔️        |          ✔️         | OK                                                                            |
| Probabilistic output (mean, variance) |         ✔️        |          ✔️         | OK                                                                            |
| NLL as loss (proper scoring rule)     |         ✔️        |          ✔️         | OK                                                                            |
| Diversity via random initialization   |         ✔️        |          ✔️         | OK, but also ensure you shuffle the data for each model                       |
| Bagging (bootstrap)                   |         ❌         |          ✖️         | Article says **don’t use bagging** for NNs; training on all data is preferred |
| MC Dropout                            |         ✔️        |          ➖          | Optional. Article uses it for comparison, not as main method                  |
| **Adversarial Training** (FGSM)       |         ❌         |          ✔️         | **Implement this for improved smoothness & robustness**                       |
| OOD/Domain Shift Evaluation           |         ❌         |          ✔️         | Optionally, add OOD tests to assess calibration/uncertainty                   |
| Combining predictions as mixture      |         ✔️        |          ✔️         | OK                                                                            |
| Evaluation: MAE, RMSE, NLL            |         ✔️        |          ✔️         | OK                                                                            |
| Hyperparam: batch size, learning rate |         ✔️        |          ✔️         | You use reasonable defaults; the article used Adam, lr=0.1, batch=100         |
| Visualization (uncertainty intervals) |         ✔️        |          ✔️         | OK                                                                            |

| Category             | Your Implementation       | Paper’s Approach                  | Room for Improvement |
| -------------------- | ------------------------- | --------------------------------- | -------------------- |
| Ensemble             | ✅ Yes                     | ✅ Yes                             | None                 |
| NLL Loss             | ✅ Yes                     | ✅ Yes                             | None                 |
| MC Dropout           | ✅ Yes                     | ✅ Optional                        | Try ensemble-only    |
| Adversarial Training | ❌ No                      | ✅ Yes                             | **Add this**         |
| Mixture Variance     | ❌ Averaged std            | ✅ Mean of variance + var of means | **Fix this**         |
| Evaluation           | ✅ Strong (MAE, RMSE, NLL) | ✅ Same                            | Add calibration plot |






##### Parameters and hyper parameters 

| **Category**         | **Parameter/Hyperparameter** | **Value/Type**                     | **Description** |
|----------------------|------------------------------|------------------------------------|-----------------|
| **Data Preparation** | Selected Features            | 15 features (listed in code)       | Weather, temporal, and holiday features |
|                      | Window Size                  | 168 (7 days in hours)              | Lookback period for LSTM |
|                      | Scaling Method               | StandardScaler                     | Normalizes features and target |
| **Model Architecture** | LSTM Units                  | 50                                 | Number of LSTM neurons |
|                      | Dense Layer Units            | 20                                 | Neurons in hidden dense layer |
|                      | Output Layer                 | 2 (μ and log-σ²)                   | Probabilistic output |
|                      | Dropout Rates                | 0.3 (all layers)                   | MC Dropout probability |
|                      | Activation                   | ReLU (Dense layer)                 | Nonlinear activation |
| **Training**         | Ensemble Size                | 5 models                           | Number of models in ensemble |
|                      | Epochs                       | 30                                 | Training iterations |
|                      | Batch Size                   | 32                                 | Samples per gradient update |
|                      | Optimizer                    | Adam                               | Optimization algorithm |
|                      | Learning Rate                | 0.001                              | Step size for weight updates |
|                      | Clipnorm                     | 1.0                                | Gradient clipping value |
| **Inference**        | MC Dropout Iterations        | 20                                 | Forward passes per prediction |
| **Loss Function**    | NLL Loss                     | Custom implementation              | Negative Log Likelihood |



#### These are Parameters

| **Parameter**                    | **Description**                                                           |
| -------------------------------- | ------------------------------------------------------------------------- |
| LSTM input weights               | Weight matrix connecting inputs to the LSTM’s hidden state                |
| LSTM recurrent weights           | Weight matrix connecting the LSTM’s previous hidden state to current      |
| LSTM biases                      | Bias vectors for the LSTM gates                                           |
| Dense (20 units) weights         | Weight matrix of the intermediate Dense layer                             |
| Dense (20 units) biases          | Bias vector of the intermediate Dense layer                               |
| Output Dense (μ, log σ²) weights | Weight matrix of the final Dense layer (predicting mean and log-variance) |
| Output Dense (μ, log σ²) biases  | Bias vector of the final Dense layer                                      |


#### Hyperparameters

| **Hyperparameter**      | **Default / Example** | **Description**                                                                                     |
| ----------------------- | --------------------- | --------------------------------------------------------------------------------------------------- |
| window\_size            | 168                   | Length of the input time window (hours) for the LSTM                                                |
| n\_ensemble             | 5                     | Number of separate models in the deep ensemble                                                      |
| n\_iter                 | 20                    | Number of MC-Dropout forward passes per model                                                       |
| lstm\_units             | 50                    | Number of neurons in the LSTM layer                                                                 |
| dense\_units            | 20                    | Number of neurons in the intermediate Dense layer                                                   |
| dropout\_rate           | 0.3                   | Dropout probability applied in LSTM and Dense layers                                                |
| learning\_rate          | 0.001                 | Learning rate for the optimizer (Adam)                                                              |
| clipnorm                | 1.0                   | Gradient clipping norm to prevent exploding gradients                                               |
| batch\_size             | 32                    | Number of samples per gradient update                                                               |
| epochs                  | 30                    | Maximum number of full passes through the training data                                             |
| selected\_features      | list of 15 features   | Input features used by the model (e.g., temperature, solar radiation, wind speed, time flags, etc.) |
| rolling\_window\_length | 7                     | Window length (days) for computing rolling features like average temperature or long-holiday flag   |

#### maybe other change that would be have good efficient on the result

| **Item**              | **Type**           | **Reason**                                             |
| --------------------- | ------------------ | ------------------------------------------------------ |
| Selected Features     | **Hyperparameter** | You choose them before training.                       | 
| Window Size           | **Hyperparameter** | Determines the input shape — **not learned**.          |✅
| uncertainty method    | **Hyperparameter** | Chosen preprocessing strategy.                         |
| Activation (ReLU)     | **Hyperparameter** | Chosen nonlinearity.                                   |
| Optimizer (Adam)      | **Hyperparameter** | Chosen algorithm for gradient descent.                 |
| Loss Function (NLL)   | **Hyperparameter** | Chosen objective function.                             |


#### For this week I decided to optimize the parameters
I will optimize based on this 
- **window_size**: Number of past time steps (e.g. 168 hours = 1 week) used as input to predict the next value.
- **n_ensemble**: Number of independently trained LSTM models in the ensemble to improve robustness and estimate uncertainty.
- **n_iter**: Number of stochastic forward passes with dropout per model during MC Dropout inference to sample predictions.
- **lstm_units**: Number of hidden units in the LSTM layer, controlling the capacity to capture temporal patterns.
- **dense_units**: Number of neurons in the dense (fully connected) layer after LSTM for further feature transformation.
- **dropout_rate**: Fraction of neurons randomly dropped during training (and MC inference) to regularize and estimate uncertainty.
- **learning_rate**: Step size used by the optimizer (Adam) to update model weights during training.
- **clipnorm**: Maximum allowed norm for gradient clipping to prevent exploding gradients during backpropagation.
- **batch_size**: Number of training samples used per model weight update.
- ***epochs***: Total number of times the model is trained over the entire training dataset.


#### Ways to Estimate Uncertainty in Machine Learning
| Type                            | Method                          | Description                                                               |
| ------------------------------- | ------------------------------- | ------------------------------------------------------------------------- |
| **Bayesian**                    | Variational Inference (VI)      | Learns a probability distribution over model weights.                     |
|                                 | MCMC (Markov Chain Monte Carlo) | Samples weights from a posterior distribution.                            |
|                                 | Laplace Approximation           | Approximates the posterior with a Gaussian around the MAP estimate.       |
| **Frequentist / Deterministic** | Deep Ensembles                  | Trains multiple models and aggregates their predictions.                  |
|                                 | MC Dropout                      | Applies dropout at inference time to simulate model sampling.             |
|                                 | Test-Time Augmentation (TTA)    | Adds perturbations to inputs at test time to simulate uncertainty.        |
|                                 | Quantile Regression             | Predicts percentiles of the output distribution (e.g., 5th, 95th).        |
|                                 | Conformal Prediction            | Builds prediction intervals with statistical guarantees.                  |
|                                 | SWA / SWAG                      | Averages weights over training (or samples Gaussian from them).           |
|                                 | Gaussian Processes              | Directly models uncertainty through a kernel-based approach.              |
| **Output-based**                | Predictive Variance (NLL Loss)  | Models mean and variance directly and trains via Negative Log Likelihood. |
|                                 | Mixture Density Networks (MDN)  | Predicts a mixture of probability distributions (e.g., Gaussians).        |
#### Final Comparison Table — Uncertainty Methods for Time Series
| Method                       | Type        | Implementation Complexity | LSTM Compatibility  | Uncertainty Accuracy | Time Series Friendly |
| ---------------------------- | ----------- | ------------------------- | ------------------- | -------------------- | -------------------- |
| MC Dropout                   | Frequentist | Low                       | ✅ Yes               | Medium–Good          | ✅ Yes                |
| Deep Ensembles               | Frequentist | Medium                    | ✅ Yes               | ⭐ Excellent          | ✅ Yes                |
| Variational Inference (VI)   | Bayesian    | High                      | ⚠️ Possible         | ⭐ Excellent          | ✅ Yes                |
| Quantile Regression          | Frequentist | Medium                    | ✅ Yes               | Moderate             | ✅ Yes                |
| SWA / SWAG                   | Frequentist | Medium                    | ✅ Yes               | Good                 | ✅ Yes                |
| Conformal Prediction         | Statistical | Medium                    | ⚠️ Needs adaptation | Good                 | ✅ With modification  |
| Predictive Variance (NLL)    | Frequentist | Low                       | ✅ Yes               | ⭐ Excellent          | ✅ Yes                |
| Test-Time Augmentation (TTA) | Frequentist | Low                       | ⚠️ Limited use      | Moderate             | ✅ With care          |
#### which teqniques dis we use? 
| Method                    | Implemented? | Notes                                 |
| ------------------------- | ------------ | ------------------------------------- |
| Predictive Variance (NLL) | ✅ Yes        | Main method used for uncertainty      |
| Deep Ensembles            | ✅ Yes        | Multiple models trained independently |
| MC Dropout                | ✅ Yes        | Dropout used at inference time        |
| Variational Inference     | ❌ No         | Requires Bayesian RNN setup           |
| Quantile Regression       | ❌ No         | Needs different loss and output       |
| Conformal Prediction      | ❌ No         | External post-hoc method              |
| SWA / SWAG                | ❌ No         | Weight averaging across epochs        |
| Test-Time Augmentation    | ❌ No         | Not used in your data pipeline        |


## Mixture or simple model ? based on the Article  

| Category             | Your Implementation       | Paper’s Approach                  | Room for Improvement |
| -------------------- | ------------------------- | --------------------------------- | -------------------- |
| Ensemble             | ✅ Yes                     | ✅ Yes                             | None                 |
| NLL Loss             | ✅ Yes                     | ✅ Yes                             | None                 |
| MC Dropout           | ✅ Yes                     | ✅ Optional                        | Try ensemble-only    |
| Adversarial Training | ❌ No                      | ✅ Yes                             | **Add this**         |
| Mixture Variance     | ❌ Averaged std            | ✅ Mean of variance + var of means | **Fix this**         |
| Evaluation           | ✅ Strong (MAE, RMSE, NLL) | ✅ Same                            | Add calibration plot |

based on this I tried to implement the mixture variance but it does not have any impact 
•  Code 1 (Simple): MAE 20.62, RMSE 29.05, NLL 446.44 — Runtime: 35 minutes
•  Code 2 (Improved): MAE 20.73, RMSE 29.66, NLL 451.08 — Runtime: 1h 30 minutes

and I put it in this file :  [**100.code1 and 2.ipynb **](tools/reference/models/100.code1_and_2.ipynb)

I implemented a second version of the LSTM ensemble based on the method proposed in the Deep Ensembles paper, where uncertainty is estimated by combining both aleatoric and epistemic variance. This approach is more aligned with the research literature and theoretically provides better-calibrated uncertainty.


## 5.Secondmodel with Drup out in Test.ipynb --> Droup out in on input and output
#### Goals of this week( week 4)
1. We need to improve the accuracy ( with lstm, or with add others features ).
2. Desinging the evaluation system based on the article.
3. He asked me about " How we updated the parameters and hyperparameters and propagation". 
4. we need to find out how many CNN models we need in ensemble models ? ( in this moment we have the 5 CNN models in one pakage ) --> so how can we define the number of the CNN ( or any other model) in one ensemble models. 

#### evaluation phase
- Is the mean prediction accurate? (Use RMSE, MAE — you already do this ✅)
- Is the uncertainty estimate meaningful and reliable? (This is what's missing ❗)
- based on the article --> Relevant Formulas --> Negative Log-Likelihood (NLL)  --> Paper Section: 2.2.1 and 3.1
![Synthetic Data](Images/7.jpg)
- Brier Score – another proper scoring rule --> based on the article --> Paper Section: 2.2

![Synthetic Data](Images/8.jpg)
- Calibration and Entropy – Section 3.5 & 3.6 

```python 
@tf.function
def nll_loss(y_true, y_pred):
    mu = y_pred[:, 0]
    log_var = y_pred[:, 1]
    var = tf.nn.softplus(log_var) + 1e-2
    return tf.reduce_mean(0.5 * (tf.math.log(var) + tf.square(y_true - mu) / var))
```

I have made solid progress with our energy forecast model! Initially, I built an LSTM (MAE 20.62, RMSE 29.05, NLL 446.44) that displayed positive signs. With your valuable feedback on handling weekends/holidays, uncertainty, and long-term trends, we upgraded to an Improved Deep Ensemble LSTM (MAE 20.00, RMSE 28.06, NLL 433.97), outpacing the old CNN.

- Weekends/Holidays: Added features like holiday lags and a 14-day window, improving peak alignment (see 1-week chart).
- Uncertainty: Enhanced with MC-Dropout and NLL, tightening 95% CI, though NLL still needs calibration.
- Long-term Trends: Deepened the network and expanded the window, boosting 1-month accuracy.
- In one week chart The LSTM’s predicted mean aligns better with true energy peaks than a CNN likely would, which tended to smooth them.
- In one month chart The LSTM tracks weekly cycles more accurately than a CNN, which struggled with long-term trends, though some deviations persist.


These changes address your concerns effectively, though we will refining uncertainty further.  for more better uncertainty.
and some other question I need to ask
- What ensemble size (e.g., 5-10) optimizes NLL for uncertainty?
- Any suggestions for variance regularization to lower NLL?



## 4.Secondmodel with Drup out in Test.ipynb --> Droup out in on input and output

![Synthetic Data](Images/6.jpg)

Deep Ensemble CNN with MC Dropout for Energy Forecasting

This project uses a **Convolutional Neural Network (CNN)** ensemble with **Monte Carlo Dropout** for short-term and mid-term energy consumption forecasting. It includes calendar-based features and multiple time-span evaluations.
1. Load Scaled Datasets
We begin by loading the preprocessed datasets.
2.Add Weekend and Holiday Features
We load a holiday file and generate new binary columns:
- is_weekend: 1 if Saturday or Sunday
- is_holiday: 1 if the date matches holidays
```python
holidays_df = pd.read_excel("3.Holydays .xlsx")
holidays_df['DateKEY'] = pd.to_datetime(holidays_df['DateKEY'], format='%Y%m%d')

def create_weekend_holiday_feature(df, holidays_df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['is_weekend'] = df['datetime'].dt.weekday.isin([5, 6]).astype(int)
    df['is_holiday'] = df['datetime'].dt.date.isin(holidays_df['DateKEY'].dt.date).astype(int)
    return df

df_Train = create_weekend_holiday_feature(df_Train, holidays_df)
df_Val = create_weekend_holiday_feature(df_Val, holidays_df)
df_Test = create_weekend_holiday_feature(df_Test, holidays_df)
```
3. Extract Time-Based Features
4. Select Input Features
The selected feature set:
```python
'air_temperature', 'diffuse_r', 'elspot', 'full_solar', 'global_r',
'gust_speed', 'relative_humidity', 'sunshine', 'wind_speed',
'hour', 'weekday', 'is_weekend', 'is_holiday'
```
5. Create Sliding Windows
Each sample consists of the previous 48 hours to predict the next hour’s energy.
6. Define the CNN Model with Dropout
The CNN is built using Conv1D layers and Dropout for uncertainty modeling.
7. Train Ensemble of CNN Models
We train 5 models independently as an ensemble
8. Monte Carlo Dropout Predictions
We use dropout at inference time to obtain uncertainty.
9. Ensemble Aggregation and Evaluation
We average across ensemble members and compute MAE and RMSE
10. Plot Forecast for One Day
11.Forecasts Over Multiple Time Spans

## 📊 Comparison of Model 1 and Model 2 (Deep Ensemble CNNs with MC Dropout)

| Feature/Aspect                     | **Model 1**                                                                 | **Model 2**                                                                                              |
|-----------------------------------|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| 🔁 Dropout Activation             | Only enabled manually in each layer using `training=True` in model layers | Uses standard `Dropout()` layers + `training=True` at inference (more stable and conventional)          |
| 📆 Calendar Features              | ❌ Not included                                                             | ✅ Includes `is_weekend` and `is_holiday` as binary features                                              |
| 🕒 Time Features                  | ❌ Not included                                                             | ✅ Includes `hour` and `weekday` as input features                                                        |
| 📦 Input Features Used            | 9 weather-related features                                                 | 13 features (weather + time + calendar context)                                                          |
| 🧠 Model Structure                | Functional API with manual dropout calls per layer                        | Sequential API with standard dropout; cleaner and easier to scale                                        |
| 📉 Sliding Window Function        | Basic, no holidays or time awareness                                      | Extended version includes timestamps and handles enriched feature sets                                    |
| 📅 Holiday Awareness              | ❌ No holiday detection                                                     | ✅ Uses an external file (`3.Holydays.xlsx`) to detect public holidays                                    |
| 📊 Plotting Scope                 | One-day forecast only                                                     | Forecasts visualized across multiple time spans (1 day to full test range)                               |
| 📈 Evaluation Depth               | Only single evaluation (one-day plot)                                     | Full evaluation across various spans + uncertainty bands                                                 |
| 🔍 Prediction Averaging          | Manual mean and std across ensemble outputs                               | Same, but more clearly structured and scalable                                                           |
| 🧪 Experimental Clarity           | Less modular, fewer explanations and test variations                      | More modular, better structured, includes different durations and insights                               |

---

 ✅ Why Model 2 is Better

Model 2 improves on Model 1 in several important ways:

1. **More Context-Aware Features**  
   It incorporates time-based (`hour`, `weekday`) and calendar-based (`is_weekend`, `is_holiday`) features. These are essential in energy forecasting, where demand typically varies across weekdays, weekends, and holidays.

2. **Cleaner Architecture**  
   It uses a more conventional and modular `Sequential` Keras model structure, which is easier to interpret and extend.

3. **Automatic Feature Enrichment**  
   Model 2 processes and includes weekend/holiday information programmatically, making it better aligned with real-world applications.

4. **Forecast Evaluation Over Time**  
   Rather than visualizing only a single day like Model 1, Model 2 evaluates performance across a range of time spans (1 day, 1 week, 1 month, etc.), allowing for more comprehensive performance insights.

5. **Better Scalability and Reusability**  
   Model 2 separates out preprocessing, model construction, and evaluation more cleanly, making it easier to scale, tune, or adapt to other datasets.

6. **Uncertainty Awareness Across Horizons**  
   With multi-span visualization and ensemble-based MC Dropout, Model 2 gives more actionable forecasts by including uncertainty bands over different durations.


when we want to compare the program with the code we can find out : 

| **Feature/Technique**                                                    | **Implemented in Your Code** | **Notes**                                                                                                                                      |
| ------------------------------------------------------------------------ | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Deep Ensembles (multiple independently trained NNs)**                  | ✅ Yes                        | You train 5 independent CNNs (`n_ensemble = 5`).                                                                                               |
| **Random initialization of each model**                                  | ✅ Yes                        | Each model uses new weights by default.                                                                                                        |
| **MC Dropout for uncertainty estimation**                                | ✅ Yes                        | Dropout is applied at test time using `training=True`.                                                                                         |
| **Sliding window for time series input**                                 | ✅ Yes                        | You use a 48-timestep sliding window over features.                                                                                            |
| **Mean and Std Deviation for prediction aggregation**                    | ✅ Yes                        | You aggregate across ensemble and MC dropout samples.                                                                                          |
| **Proper scoring rule: Mean Squared Error (MSE)**                        | ✅ Yes                        | You use `loss='mse'`, which is a proper scoring rule.                                                                                          |
| **Negative Log Likelihood (NLL) with heteroscedastic output**            | ❌ No                         | The paper recommends modeling both mean & variance of predictions (heteroscedastic Gaussian) and optimizing NLL. Your code only predicts mean. |
| **Adversarial training for robust uncertainty**                          | ❌ No                         | The paper proposes adversarial examples to smooth predictive distribution. Your code does not use adversarial examples.                        |
| **Bagging or bootstrapping for diversity**                               | ❌ No                         | Not recommended in paper; also not used by you. Instead, random initialization and shuffling suffice.                                          |
| **Calibration evaluation (e.g. NLL, Brier Score, Reliability diagrams)** | ❌ No                         | You evaluate using MAE and RMSE, but not NLL or Brier Score which are common for uncertainty.                                                  |
| **Out-of-distribution (OOD) evaluation**                                 | ❌ No                         | Paper tests on OOD datasets (e.g., NotMNIST); you only test on one university dataset.                                                         |
| **Histogram or entropy of predictive probabilities**                     | ❌ No                         | Paper evaluates predictive entropy on unknown vs known distributions.                                                                          |
| **Accuracy vs Confidence Curves**                                        | ❌ No                         | Not included in your evaluation, but shown in the paper.                                                                                       |


---

🟢 **Conclusion**: Model 2 is a robust, scalable, and context-aware evolution of Model 1. It integrates more relevant features and provides better interpretability, making it a stronger candidate for real-world deployment.



## 3.Secondmodel with Drup out in Test.ipynb --> Droup out in on input and output

![Synthetic Data](Images/5.jpg)

## 2.First model with Drup out.ipynb --> Just drop-out for the Train data | not drop out for the test data
- Predict 1 hour ahead  ➝ (short term) baseline model (high frequency) Sliding windows is 48 H 
- Predict 24 hours ahead ➝ mid-term horizon (aggregation optional) 

| Window Length | Description                        | When to Use                                         |
| ---------------- | ---------------------------------- | --------------------------------------------------- |
| 24 hours         | 1 day of history                   | If daily cycles (day-night pattern) are strong      |
|** 48–72 hours  **    | 2–3 days                           | Captures recent shifts like weather changes         |
| 168 hours        | 1 week                             | Captures **weekly cycles** (weekend/weekday effect) |
| >168 hours       | Long-term memory (costly to train) | Use if you have large seasonal/weekly dependencies  |

### Test data result

![Synthetic Data](Images/2.jpg)

![Synthetic Data](Images/3.jpg)

![Synthetic Data](Images/4.jpg)


## 1.Core_En_Regression_ToyData.ipynb

* **Efficient Uncertainty Estimation (EffUEst)** is a deep learning platform for predictive uncertainty quantification with ensemble models.  
* The project employs a toy regression task with the target function defined as \( y = 10 \sin(x) \) with heteroscedastic Gaussian noise.  
* Five fully connected neural networks (ensemble models) are each trained separately on the same data.  
* Each network is specified with two hidden layers and provides both the mean and variance of its prediction.  
* The loss function is negative log-likelihood (NLL), allowing the model to learn correct predictions as well as associated uncertainty.  
* Gradient clipping and Glorot initialization are used to stabilize and accelerate training.  
* Ensemble uncertainty is approximated by combining both prediction disagreement and model variance.  
* The project employs a formal training loop, where each model is iteratively trained and tested on mini-batches.  
* Predictions are done over an input grid of values to plot the confidence intervals and the mean prediction.  
* It is compared between an individual model and the ensemble in order to show the gains in ensemble learning reliability.  
* All outputs are plotted neatly, demonstrating ensemble-based uncertainty's capacity to better represent variability in the underlying function than single models.  
* The work is designed modularly so that any dataset and noise profile can be fed into the design.  
* Code is organized into reusable utilities, according to research project documentation and reproducibility guidelines.  
* EffUEst is a foundation for uncertainty-aware modeling in areas where the confidence of predictions is just as valuable as precision.


### 1. `1.png`: True Function vs. Noisy Observations
![Synthetic Data](tools/reference/figures/1.png)
* Visualizes the synthetic data set for training.  
* Noisy observations are black dots generated by the function \( y = 10 \sin(x) + \epsilon \).  
* Heteroscedastic noise: drawn from \( \mathcal{N}(0, 3^2) \) for \( x < 0 \), and \( \mathcal{N}(0, 1^2) \) for \( x \geq 0 \).  
* The green line is the underlying true function \( 10 \sin(x) \), without any noise.  
* The purpose of this plot is to see the variability in data and the type of noise added during generation.
---
![Synthetic Data](tools/reference/figures/2.png)
### 2. `2.png`: Ensemble Prediction vs. Ground Truth

* The plot contrasts the ensemble model prediction with the true function.  
* The green line is the ground truth \( 10 \sin(x) \).  
* The blue dashed lines represent the predicted uncertainty bounds of the ensemble: \( \mu \pm \sigma \).  
* The middle solid blue line is the ensemble mean prediction across 5 models.  
* The ensemble is picking up both the general shape of the function and the varying levels of uncertainty — higher for \( x < 0 \), lower for \( x \geq 0 \).  
* It demonstrates the ability of the ensemble to model both data distribution and confidence throughout the input space.
---
### 3. `3.png`: Ensemble vs. Single Network Comparison
![Synthetic Data](tools/reference/figures/3.png)
* It contrasts the predictive uncertainty of the ensemble against that of a single neural network.  
* Blue dashed lines show ensemble uncertainty bounds \( \mu \pm \sigma \), with the solid blue line being the ensemble mean.  
* The magenta dash-dot lines represent the single model's \( \mu \pm \sigma \) bounds, and the magenta solid line represents its mean.  
* The ensemble has smoother and more believable uncertainty intervals, especially where the data is noisier.  
* The single model shows more variation and overestimation in regions with less data support.  
* This comparison validates the primary claim: ensemble methods yield more stable and trustworthy uncertainty estimates than a single model alone.



# Preprocecing 


| Feature             | `resample()`                                   | `asfreq()`                                            |
| ------------------- | ---------------------------------------------- | ----------------------------------------------------- |
| **Main purpose**    | **Aggregation of data**                        | **Select a specific value at new frequency**          |
| **Statistical ops** | Yes (e.g., `mean()`, `sum()`, etc.)            | No (just selects a value like last timestamp)         |
| **Handling NaNs**   | Easier (supports `.ffill()`, `.bfill()`, etc.) | Typically needs manual filling if needed              |
| **Typical usage**   | Smooth data (e.g., average over a month)       | Retain exact value at period-end (e.g., end of month) |

Example: 
```python
# asfreq method is used to convert a time series to a specified frequency. Here it is monthly frequency.
# Set figure width and height
plt.rcParams["figure.figsize"] = [12,6]

SF["Humidity"].asfreq('M').plot() 
SF['Humidity'].resample('M').mean().plot(style=':')
SF["Humidity"].asfreq('M').shift(1).plot()  # lagged by a month

plt.title('Humidity in San Francisco over time (Monthly frequency)')
plt.ylabel('Humidity')
plt.xlabel('Date time (in years)')
plt.legend(['asfreq', 'resample mean', 'asfreq + shift'],
           loc='upper left');
plt.show()
```
