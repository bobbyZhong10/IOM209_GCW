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
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0), loss='mean_squared_error')

        # Model training
        model.fit(X_train, y_train, epochs=trial.suggest_int("epochs", 10, 100), batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]), verbose=0)

        # Model evaluation
        mse = model.evaluate(X_test, y_test, verbose=0)
        if np.isnan(mse):
            continue
        mse_scores.append(mse)

    return np.mean(mse_scores) if mse_scores else float('inf')

# Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Using best parameters to rebuild the model for testing
best_model = tf.keras.Sequential([
    Input(shape=(1, 1)),
    Bidirectional(LSTM(study.best_trial.params['lstm_units'])),
    Dropout(study.best_trial.params['dropout_rate']),
    Dense(1)
])

best_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=study.best_trial.params['learning_rate']), loss='mean_squared_error')
best_model.fit(features_scaled_train, target_train, epochs=study.best_trial.params['epochs'], batch_size=study.best_trial.params['batch_size'], verbose=1)

# Predicting the test set and calculating metrics
test_predictions = best_model.predict(features_scaled_test).flatten()
mse = mean_squared_error(target_test, test_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(target_test, test_predictions)
mae = mean_absolute_error(target_test, test_predictions)

# Correct Prediction Direction Proportion
direction_predictions = np.sign(test_predictions[1:] - test_predictions[:-1])
true_directions = np.sign(target_test[1:] - target_test[:-1])
correct_directions = direction_predictions == true_directions
direction_accuracy = np.mean(correct_directions)

# Display metrics
print(f"Test MSE: {mse}")
print(f"Test RMSE: {rmse}")
print(f"Test R2: {r2}")
print(f"Test MAE: {mae}")
print(f"Direction Accuracy: {direction_accuracy * 100:.2f}%")

# Visualize test set results
plt.figure(figsize=(10, 5))
plt.plot(test_data['Date'], target_test, label='True Values')
plt.plot(test_data['Date'], test_predictions, label='Predictions')
plt.title('Test Set Predictions vs True Values')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.savefig('test_predictions.png')
plt.show()
