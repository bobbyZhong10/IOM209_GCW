import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

# Load data
data_path = 'study1_final_all.csv'
data = pd.read_csv(data_path)
data['Date'] = pd.to_datetime(data['Date']).dt.to_period('W').dt.to_timestamp()  # Convert to standard date format

# Data preprocessing
features = data[['Date', 'GB_emotion']].copy()
features['GB_emotion'].fillna(features['GB_emotion'].mean(numeric_only=True), inplace=True)
target = data[['Date', 'Wretnd']].copy()
target['Wretnd'].fillna(target['Wretnd'].mean(numeric_only=True), inplace=True)

# Aggregate by date to calculate total return
target = target.groupby('Date').sum()
features = features.groupby('Date').mean()

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled = np.reshape(features_scaled, (features_scaled.shape[0], 1, features.shape[1]))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target.values, test_size=0.2, random_state=42, shuffle=False)
dates_train, dates_test = train_test_split(data['Date'].unique(), test_size=0.2, random_state=42, shuffle=False)

def objective(params):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.LSTM(int(params['first_lstm_units']), return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(int(params['second_lstm_units'])),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    result = model.fit(X_train, y_train, epochs=int(params['epochs']), batch_size=int(params['batch_size']), verbose=0, validation_split=0.2)
    val_loss = np.min(result.history['val_loss'])
    return {'loss': val_loss, 'status': STATUS_OK}

# Define search space
space = {
    'first_lstm_units': hp.quniform('first_lstm_units', 30, 70, 10),
    'second_lstm_units': hp.quniform('second_lstm_units', 10, 50, 10),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01)),
    'epochs': hp.quniform('epochs', 30, 100, 10),
    'batch_size': hp.quniform('batch_size', 16, 64, 8)
}

# Run Hyperopt
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

print("Best parameters:", best)

# Retrain model with best parameters
final_model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.LSTM(int(best['first_lstm_units']), return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(int(best['second_lstm_units'])),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])
final_optimizer = tf.keras.optimizers.Adam(learning_rate=best['learning_rate'])
final_model.compile(optimizer=final_optimizer, loss='mean_squared_error')
final_model.fit(X_train, y_train, epochs=int(best['epochs']), batch_size=int(best['batch_size']), verbose=0)

# Save prediction results
predictions = final_model.predict(X_test)
np.savetxt("predictions.csv", predictions, delimiter=",")

# Calculate evaluation metrics
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print(f"MSE: {mse}, RMSE: {rmse}, R²: {r2}")

# Plot predictions vs actual results
plt.figure(figsize=(10, 6))
plt.plot(dates_test, y_test, label='Actual')
plt.plot(dates_test, predictions, label='Predicted', linestyle='--')
plt.title('Comparison of Predicted and Actual Results')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()
