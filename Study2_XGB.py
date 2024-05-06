import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# Load data
data_path = 'study1_final_all.csv'
data = pd.read_csv(data_path)
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)

# Ensure 'Scode' is treated as a string to avoid any numeric interpretation
data['Scode'] = data['Scode'].astype(str)

# Splitting unique stock codes into training and testing sets
unique_codes = data['Scode'].unique()
train_codes, test_codes = train_test_split(unique_codes, test_size=0.4, random_state=42)

# Splitting the dataset based on the train and test codes
train_data = data[data['Scode'].isin(train_codes)].dropna()
test_data = data[data['Scode'].isin(test_codes)].dropna()

# Data preprocessing
imputer = KNNImputer(n_neighbors=5)
scaler = StandardScaler()

# Feature engineering for train and test sets
features_train = train_data[['GB_emotion']].copy()
features_train['GB_emotion'] = imputer.fit_transform(features_train[['GB_emotion']].values.reshape(-1, 1))
features_scaled_train = scaler.fit_transform(features_train)

features_test = test_data[['GB_emotion']].copy()
features_test['GB_emotion'] = imputer.transform(features_test[['GB_emotion']].values.reshape(-1, 1))
features_scaled_test = scaler.transform(features_test)

# Target variable handling for train and test sets
target_train = train_data['Wretnd'].values
target_test = test_data['Wretnd'].values

# Initialize and train XGBoost model
model = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)
model.fit(features_scaled_train, target_train)

# Predict on test data
y_pred = model.predict(features_scaled_test)

# Evaluate the model
mse = mean_squared_error(target_test, y_pred)
mae = mean_absolute_error(target_test, y_pred)
r2 = r2_score(target_test, y_pred)

# Print evaluation metrics
print(f'MSE: {mse}')
print(f'MAE: {mae}')
print(f'R^2 Score: {r2}')
