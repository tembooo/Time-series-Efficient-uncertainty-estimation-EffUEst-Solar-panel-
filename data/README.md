# Data

The *parquet* files in this folder contain the cleaned-up partitions to be used for the project. Datasets 0-2 are numbered chronologically and are intended to be used for testing, validation, and training respectively.

Each file contains the following columns:
- *datetime*: UTC-encoded timestamp for each sample
- *air_temperature* \[$^\circ C$\]
- *cloud_amount* \[[Okta](https://en.wikipedia.org/wiki/Okta)\]
- *dewpoint_temperature* \[$^\circ C$\]
- *difuse_r* \[$W/m^2$\]
- *elspot*: Elspot spot-price in $€$
- *energy*: Aggregated hourly power consumption in $kW$
- *full_solar*: Aggregated hourly power generation in $kW$
- *global_r* \[$W/m^2$\]
- *gust_speed* \[$m/s$\]
- *horizontal_visibility* \[$m$\]
- *pressure* \[$hPa$\]
- *relative_humidity* \[$\%$\]
- *sunshine* \[?\] (usage not recommended as the source is not well-documented)
- *wind_direction* \[$^\circ$\]
- *wind_speed* \[$m/s$\]

Pending: code example on how to format for ANN usage.


# Step1 : Undrestanding the data 

| cloud_amount | dewpoint_temperature | diffuse_r | elspot | energy | full_solar | global_r | gust_speed | horizontal_visibility | pressure | relative_humidity | sunshine | wind_direction | wind_speed |
|--------------|----------------------|-----------|--------|--------|------------|----------|------------|------------------------|----------|--------------------|----------|------------------|------------|
| 8            | 4.3                  | 0.8       | 44.41  | 161.6  | 16.33925   | 2.6      | 10.3       | 40000                 | 1032     | 85                 | 0        | 240              | 6.7        |
| 8            | 4.0                  | 0.7       | 44.05  | 160.6  | 16.53429   | 2.7      | 9.3        | 40000                 | 1031.5   | 84                 | 0        | 230              | 5.7        |
| 8            | 4.1                  | 1.0       | 44.19  | 160.2  | 17.13169   | 2.8      | 8.7        | 40000                 | 1031.2   | 85                 | 0        | 240              | 6.2        |
| 8            | 4.1                  | 1.1       | 44.66  | 159.0  | 16.91387   | 2.8      | 8.7        | 35000                 | 1030.4   | 85                 | 0        | 230              | 6.2        |
| 8            | 4.2                  | 1.0       | 46.45  | 158.0  | 16.95934   | 2.9      | 8.7        | 35000                 | 1030.3   | 86                 | 0        | 250              | 6.2        |
| 8            | 4.4                  | 1.0       | 47.42  | 155.8  | 16.69028   | 3.3      | 9.3        | 35000                 | 1029.8   | 87                 | 0        | 250              | 6.2        |
| 7            | 4.3                  | 1.0       | 48.76  | 156.6  | 16.74959   | 3.8      | 6.7        | 30000                 | 1029.8   | 87                 | 0        | 260              | 4.6        |
| 7            | 4.2                  | 7.1       | 49.18  | 156.4  | 16.06958   | 8.8      | 7.2        | 30000                 | 1029.9   | 86                 | 0        | 260              | 5.1        |
| 6            | 4.0                  | 50.9      | 48.38  | 135.3  | 135.33496  | 84.6     | 6.2        | 40000                 | 1029.5   | 85                 | 0        | 260              | 5.1        |
| 7            | 3.9                  | 58.1      | 47.82  | 150.2  | 4264.5732  | 84.6     | 6.2        | 40000                 | 1029.3   | 83                 | 35       | 270              | 4.1        |
---
**Testing data** shape is (2208, 16) and number of rows are (2208).

**Validation data** shape is (5856, 16) and number of rows are (5856).

**Training data shape** is (11712, 16) and number of rows are (11712).

| Dataset        | Rows   | Portion (%)                       |
| -------------- | ------ | --------------------------------- |
| **Training**   | 11,712 | **59.2%** (`11712 / 19776 × 100`) |
| **Validation** | 5,856  | **29.6%** (`5856 / 19776 × 100`)  |
| **Testing**    | 2,208  | **11.2%** (`2208 / 19776 × 100`)  |

Future1: In future i will try to evaluate this portion as well  --> 🔄 70% Training / 20% Validation / 10% Testing
```python
import pandas as pd
# Load the dataset
df = pd.read_parquet("dataset_0.parquet")
# Remove timezone info from the datetime column
df["datetime"] = df["datetime"].dt.tz_localize(None)
# Save to Excel
df.to_excel("0_weather_data_testing_0.xlsx", index=False)
print("Excel file saved successfully.")
print(f"Testing data shape is {df.shape} and number of rows are ({len(df)}).")

################################################################
df = pd.read_parquet("dataset_1.parquet")
# Remove timezone info from the datetime column
df["datetime"] = df["datetime"].dt.tz_localize(None)
# Save to Excel
df.to_excel("1_weather_data_validation_1.xlsx", index=False)
print("Excel file saved successfully.")
print(f"Validation data shape is {df.shape} and number of rows are ({len(df)}).")

################################################################

df = pd.read_parquet("dataset_2.parquet")
# Remove timezone info from the datetime column
df["datetime"] = df["datetime"].dt.tz_localize(None)
# Save to Excel
df.to_excel("2_weather_data_training_2.xlsx", index=False)
print("Excel file saved successfully.")
print(f"Training data shape is {df.shape} and number of rows are ({len(df)}).")
```
---
There is no any missing value. 
```python
df.isnull().sum()
```
All of data types are float64 
```python
import pandas as pd
# Load your data
df = pd.read_parquet("dataset_2.parquet")
# Check full data types
print(" Full Data Types:")
print(df.dtypes)
# Separate columns by type
datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64[ns, UTC]']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
# Show results
print("\n Datetime Columns:", datetime_cols)
print(" Numerical Columns:", numerical_cols)
print(" Categorical Columns:", categorical_cols)
```
### Full correlation matrix
|                       |   datetime |   air_temperature |   cloud_amount |   dewpoint_temperature |   diffuse_r |   elspot |   energy |   full_solar |   global_r |   gust_speed |   horizontal_visibility |   pressure |   relative_humidity |   sunshine |   wind_direction |   wind_speed |
|:----------------------|-----------:|------------------:|---------------:|-----------------------:|------------:|---------:|---------:|-------------:|-----------:|-------------:|------------------------:|-----------:|--------------------:|-----------:|-----------------:|-------------:|
| datetime              |      1     |             0.141 |          0.068 |                  0.229 |      -0.029 |   -0.366 |    0.06  |       -0.042 |     -0.056 |        0.173 |                   0.122 |     -0.25  |               0.124 |     -0.047 |           -0.019 |        0.144 |
| air_temperature       |      0.141 |             1     |         -0.261 |                  0.891 |       0.418 |   -0.024 |    0.035 |        0.593 |      0.507 |       -0.008 |                   0.24  |      0.119 |              -0.485 |      0.383 |           -0.039 |       -0.033 |
| cloud_amount          |      0.068 |            -0.261 |          1     |                 -0.022 |      -0.09  |    0.027 |    0.033 |       -0.35  |     -0.295 |        0.119 |                  -0.464 |     -0.288 |               0.556 |     -0.406 |           -0.106 |        0.085 |
| dewpoint_temperature  |      0.229 |             0.891 |         -0.022 |                  1     |       0.296 |   -0.043 |    0.01  |        0.301 |      0.278 |       -0.06  |                   0.022 |     -0.025 |              -0.044 |      0.161 |           -0.085 |       -0.089 |
| diffuse_r             |     -0.029 |             0.418 |         -0.09  |                  0.296 |       1     |    0.183 |    0.214 |        0.523 |      0.693 |        0.08  |                   0.158 |      0.047 |              -0.364 |      0.383 |            0.05  |        0.072 |
| elspot                |     -0.366 |            -0.024 |          0.027 |                 -0.043 |       0.183 |    1     |    0.269 |        0.113 |      0.158 |       -0.159 |                  -0.127 |      0.245 |              -0.025 |      0.107 |           -0.079 |       -0.15  |
| energy                |      0.06  |             0.035 |          0.033 |                  0.01  |       0.214 |    0.269 |    1     |        0.133 |      0.164 |        0.1   |                  -0.022 |      0.024 |              -0.056 |      0.114 |            0.002 |        0.084 |
| full_solar            |     -0.042 |             0.593 |         -0.35  |                  0.301 |       0.523 |    0.113 |    0.133 |        1     |      0.768 |        0.085 |                   0.243 |      0.193 |              -0.685 |      0.568 |            0.02  |        0.075 |
| global_r              |     -0.056 |             0.507 |         -0.295 |                  0.278 |       0.693 |    0.158 |    0.164 |        0.768 |      1     |        0.12  |                   0.208 |      0.134 |              -0.569 |      0.773 |            0.053 |        0.114 |
| gust_speed            |      0.173 |            -0.008 |          0.119 |                 -0.06  |       0.08  |   -0.159 |    0.1   |        0.085 |      0.12  |        1     |                   0.063 |     -0.304 |              -0.107 |      0.084 |            0.117 |        0.96  |
| horizontal_visibility |      0.122 |             0.24  |         -0.464 |                  0.022 |       0.158 |   -0.127 |   -0.022 |        0.243 |      0.208 |        0.063 |                   1     |      0.191 |              -0.563 |      0.237 |            0.241 |        0.1   |
| pressure              |     -0.25  |             0.119 |         -0.288 |                 -0.025 |       0.047 |    0.245 |    0.024 |        0.193 |      0.134 |       -0.304 |                   0.191 |      1     |              -0.322 |      0.152 |           -0.043 |       -0.289 |
| relative_humidity     |      0.124 |            -0.485 |          0.556 |                 -0.044 |      -0.364 |   -0.025 |   -0.056 |       -0.685 |     -0.569 |       -0.107 |                  -0.563 |     -0.322 |               1     |     -0.528 |           -0.099 |       -0.112 |
| sunshine              |     -0.047 |             0.383 |         -0.406 |                  0.161 |       0.383 |    0.107 |    0.114 |        0.568 |      0.773 |        0.084 |                   0.237 |      0.152 |              -0.528 |      1     |            0.078 |        0.087 |
| wind_direction        |     -0.019 |            -0.039 |         -0.106 |                 -0.085 |       0.05  |   -0.079 |    0.002 |        0.02  |      0.053 |        0.117 |                   0.241 |     -0.043 |              -0.099 |      0.078 |            1     |        0.115 |
| wind_speed            |      0.144 |            -0.033 |          0.085 |                 -0.089 |       0.072 |   -0.15  |    0.084 |        0.075 |      0.114 |        0.96  |                   0.1   |     -0.289 |              -0.112 |      0.087 |            0.115 |        1     |
```python
df.corr() 
```

| Insight Type         | Observed Patterns                            |
| -------------------- | -------------------------------------------- |
| Strong relationships | Sunshine ↔ Radiation ↔ Temperature           |
| Redundancy           | Wind & solar variables have internal overlap |
| Noisy variables      | `elspot`, `wind_direction` show weak links   |
| Data quality         | Strong patterns suggest clean, physical data |

![Correlation Heatmap](Images/2.png)

Some columns have strong collerations but we can not delete them because they carry on important information. inthe below we have description of them : 

## Feature Descriptions

### `datetime`
UTC timestamp for the precise date and time when the measurements were recorded. This column is required for both sorting the data chronologically and for the generation of time-based features such as hour of day, day of week, etc.

### `air_temperature`
Represents ambient air temperature at the time of observation, in degrees Celsius. It is a basic meteorological parameter and is used in energy demand, weather, and renewable generation forecast models.

### `cloud_amount`
Fraction of the sky covered by cloud, in Okta units. Varies between 0 (clear sky) and 8 (completely cloudy). It directly affects solar radiation and solar power production.

### `dewpoint_temperature`
The temperature at which the air is fully saturated with water vapor, resulting in condensation. In degrees Celsius. Closely correlated with humidity and may be an indicator of fog or dew.

### `diffuse_r`
The amount of diffuse solar radiation flux received at the surface in watts per square meter (W/m²). Sunlight that has been scattered by the atmosphere and is useful for cloudy-sky modeling.

### `elspot`
Day-ahead electricity price in euros. These prices are provided for the next day and should not be predicted within this model. They might be used as input variables if modeling energy consumption or generation.

### `energy`
Overall hourly electricity consumption, in kilowatts (kW). This is typically one of the primary target variables for regression or prediction models in energy systems.

### `full_solar`
Total solar power generation in kilowatts over an hour. It is often modeled or forecasted from environmental and meteorological inputs such as radiation and cloud cover.

### `global_r`
Global solar irradiance, including both direct and diffuse components, in W/m². A good measure of available solar energy at any given time.

### `gust_speed`
Maximum wind gust speed registered during the interval, typically for a few seconds. Given in units of meters per second (m/s). Indicates wind variability and peak conditions.

### `horizontal_visibility`
The horizontal distance the observer can clearly see, given in meters. Low visibility may be due to fog, mist, or heavy precipitation and can impact solar and temperature dynamics.

### `pressure`
Atmospheric pressure in hectopascals (hPa). Trends in pressure are an excellent indicator of changing weather conditions and are utilized extensively in weather prediction models.

### `relative_humidity`
Percentage of moisture in the air relative to the air's capacity to hold moisture. It is a component in the description of comfort and facilitates the modeling of cloud cover and dew.

### `sunshine`
Estimated sunshine intensity or duration during the hour. This variable is characterized as poorly documented and not generally recommended for use in models unless validated independently.

### `wind_direction`
Direction of wind origin, in degrees (0° = North, 90° = East, 180° = South, 270° = West). May be used to model wind-dependent processes in conjunction with wind speed.

### `wind_speed`
Mean wind speed over the hour, in meters per second (m/s). A key parameter in wind energy forecasting, weather modeling, and air movement dynamics.




# Outliers
1. Summary Statistics (Z-Score or IQR Method)

| Feature                         | IQR Outliers | Z-Score Outliers |
| ------------------------------- | ------------ | ---------------- |
| cloud\_amount                   | 529          | 138              |
| full\_solar                     | 503          | 63               |
| global\_r                       | 380          | 71               |
| diffuse\_r                      | 379          | 61               |
| elspot                          | 164          | 45               |
| sunshine                        | 85           | 73               |
| relative\_humidity              | 154          | 37               |
| wind\_speed                     | 41           | 12               |
| air\_temperature                | 13           | 6                |
| dewpoint\_temp.                 | 14           | 3                |
| gust\_speed                     | 9            | 2                |
| energy                          | 0            | 0                |
| visibility, pressure, wind\_dir | 0            | 0                |

2. 2. Boxplots (Visual Detection)

![Boxplot of Numerical Features](Images/3.png)
