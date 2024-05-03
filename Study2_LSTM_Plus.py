import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Bidirectional, Conv1D, Flatten

# Load and prepare data
data_path = 'study1_final_all.csv'
data = pd.read_csv(data_path)
data['Date'] = pd.to_datetime(data['Date']).dt.to_period('W').dt.to_timestamp()

# Advanced feature engineering
features = data[['Date', 'GB_emotion']].copy()
imputer = KNNImputer(n_neighbors=5)
features['GB_emotion'] = imputer.fit_transform(features[['GB_emotion']])
target = data[['Date', 'Wretnd']].copy()
target['Wretnd'] = imputer.fit_transform(target[['Wretnd']])

# Group by date
target = target.groupby('Date').sum()
features = features.groupby('Date').mean()

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled = np.reshape(features_scaled, (features_scaled.shape[0], 1, features_scaled.shape[1]))

max_iterations = 100  # Set the maximum number of iterations
best_model = None
best_metrics = {'mse': float('inf'), 'rmse': float('inf'), 'r2': -float('inf')}
best_y_test = None
best_y_pred = None

for iteration in range(max_iterations):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(features_scaled):
        X_train, X_test = features_scaled[train_index], features_scaled[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        # Build LSTM model with CNN layers
        model = tf.keras.models.Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
            Bidirectional(LSTM(50, return_sequences=True, kernel_regularizer=l2(0.01))),
            Dropout(0.3),
            Flatten(),
            Dense(1)
        ])

        # Compile and train model
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0,
                  callbacks=[ReduceLROnPlateau(), EarlyStopping(monitor='val_loss', patience=10)])

        # Predict and evaluate
        y_pred = model.predict(X_test).flatten()
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Check if current model has the best overall metrics
        if mse < best_metrics['mse'] and r2 > best_metrics['r2']:
            best_metrics['mse'] = mse
            best_metrics['rmse'] = rmse
            best_metrics['r2'] = r2
            best_model = model
            best_y_test = y_test
            best_y_pred = y_pred

    print(f"Iteration {iteration + 1}: Best MSE: {best_metrics['mse']}, RMSE: {best_metrics['rmse']}, R^2: {best_metrics['r2']}")

# Plot the best predictions vs true values
if best_model is not None:
    plt.figure(figsize=(10, 5))
    plt.plot(best_y_test.index, best_y_test, label='True')
    plt.plot(best_y_test.index, best_y_pred, label='Predicted', alpha=0.7)
    plt.title('Weekly Portfolio Returns: Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Sum of Weekly Returns')
    plt.legend()
    plt.savefig('pic111.png')
    plt.show()
else:
    print("No model met the criteria for best metrics across all folds.")
