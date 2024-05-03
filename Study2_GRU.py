import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GRU, Bidirectional, Conv1D, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Load and prepare data
data_path = 'study1_final_all.csv'  # Update with your actual path
data = pd.read_csv(data_path)
data['Date'] = pd.to_datetime(data['Date']).dt.to_period('W').dt.to_timestamp()

# Advanced feature engineering
features = data[['Date', 'GB_emotion']].copy()  # Add more feature engineering here if needed
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

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(features_scaled, target, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build model with GRU layers
model = tf.keras.models.Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
    Bidirectional(GRU(100, return_sequences=True, kernel_regularizer=l2(0.01))),
    Dropout(0.5),
    Bidirectional(GRU(50, return_sequences=False, kernel_regularizer=l2(0.01))),
    Dropout(0.5),
    Dense(50, activation='relu'),
    Dense(1)
])

# Compile and train model
model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1,
              validation_data=(X_val, y_val),
              callbacks=[ReduceLROnPlateau(), EarlyStopping(monitor='val_loss', patience=10)])

# Predict and evaluate
y_pred = model.predict(X_test).flatten()
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Final Model Evaluation Metrics:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Plot the predictions vs true values
plt.figure(figsize=(10, 4))
plt.plot(y_test.index, y_test, 'o-', label='True Sum of Weekly Returns')
plt.plot(y_test.index, y_pred, 'o-', label='Predicted Sum of Weekly Returns')
plt.title('Comparison of Actual vs Predicted Total Weekly Returns')
plt.xlabel('Date')
plt.ylabel('Total Weekly Returns')
plt.legend()
plt.show()
