import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
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

# K-fold cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

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

    # Compile model
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # Train model
    model.fit(X_train, y_train, validation_split=0.2, epochs=200, batch_size=32, verbose=0, callbacks=[reduce_lr, early_stopping])

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results.append((mse, rmse, r2))

# Calculate average of the results
average_results = np.mean(results, axis=0)
print(f"Average MSE: {average_results[0]}, Average RMSE: {average_results[1]}, Average R^2: {average_results[2]}")
