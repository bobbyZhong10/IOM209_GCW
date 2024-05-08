import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Bidirectional
import optuna

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
features_scaled_train = scaler.fit_transform(features_train['GB_emotion'].values.reshape(-1, 1))
features_scaled_train = np.reshape(features_scaled_train, (features_scaled_train.shape[0], 1, 1))

features_test = test_data[['GB_emotion']].copy()
features_test['GB_emotion'] = imputer.transform(features_test[['GB_emotion']].values.reshape(-1, 1))
features_scaled_test = scaler.transform(features_test['GB_emotion'].values.reshape(-1, 1))
features_scaled_test = np.reshape(features_scaled_test, (features_scaled_test.shape[0], 1, 1))

# Target variable handling for train and test sets
target_train = train_data['Wretnd'].values
target_test = test_data['Wretnd'].values

# Optuna objective function
def objective(trial):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []

    for train_index, test_index in kf.split(features_scaled_train):
        X_train, X_test = features_scaled_train[train_index], features_scaled_train[test_index]
        y_train, y_test = target_train[train_index], target_train[test_index]

        # Model building
        model = tf.keras.Sequential([
            Input(shape=(1, 1)),
            Bidirectional(LSTM(trial.suggest_int('lstm_units', 50, 200), return_sequences=True)),
            Dropout(trial.suggest_float('dropout_rate', 0.1, 0.5)),
            tf.keras.layers.SimpleRNN(trial.suggest_int('lstm_units', 50, 200)),
            Dense(1, activation='relu')
        ])

        # Compile model, add gradient clipping to prevent exploding gradients
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')

        # TensorBoard integration
        from tensorflow.keras.callbacks import TensorBoard
        import time
        unique_mod_name = f"lstm_model_{int(time.time())}"
        tensorboard = TensorBoard(log_dir=f"logs/{unique_mod_name}")

        model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), callbacks=[tensorboard])
