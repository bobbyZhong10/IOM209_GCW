import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Bidirectional, Flatten
import optuna

# 加载和准备数据
data_path = 'study1_final_all.csv'
data = pd.read_csv(data_path)
data['Date'] = pd.to_datetime(data['Date']).dt.to_period('W').dt.to_timestamp()

# 高级特征工程
features = data[['Date', 'GB_emotion']].copy()
imputer = KNNImputer(n_neighbors=5)
features['GB_emotion'] = imputer.fit_transform(features[['GB_emotion']])
target = data[['Date', 'Wretnd']].copy()
target['Wretnd'] = imputer.fit_transform(target[['Wretnd']])

# 通过日期分组
target = target.groupby('Date').sum()
features = features.groupby('Date').mean()

# 标准化特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled = np.reshape(features_scaled, (features_scaled.shape[0], 1, features_scaled.shape[1]))

# K折交叉验证设置
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

# 记录最后一次迭代的 y_test 和 y_pred
last_y_test = None
last_y_pred = None


def objective(trial):
    # 定义模型参数搜索空间
    lstm_units = trial.suggest_categorical('lstm_units', [20, 50, 100])
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)

    local_results = []
    for train_index, test_index in kf.split(features_scaled):
        X_train, X_test = features_scaled[train_index], features_scaled[test_index]
        y_train, y_test = target.iloc[train_index], target.iloc[test_index]

        # 构建模型
        model = tf.keras.models.Sequential([
            Input(shape=(X_train.shape[1], X_train.shape[2])),
            Bidirectional(LSTM(lstm_units, return_sequences=True)),
            Dropout(dropout_rate),
            Flatten(),
            Dense(1)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(loss='mean_squared_error', optimizer=optimizer)

        # 训练模型
        model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=0)

        # 评估模型
        mse = model.evaluate(X_test, y_test, verbose=0)
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        local_results.append((mse, rmse, r2))
        last_y_test = y_test
        last_y_pred = y_pred

    average_results = np.mean(local_results, axis=0)
    results.append(average_results)
    return average_results[0]  # Minimize MSE


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print('Best trial:', study.best_trial.params)

# 计算整体平均结果
average_mse, average_rmse, average_r2 = np.mean(results, axis=0)
print(f"Average MSE: {average_mse}, Average RMSE: {average_rmse}, Average R^2: {average_r2}")

# 绘制预测与实际结果图
plt.figure(figsize=(10, 5))
plt.plot(last_y_test.index, last_y_test, label='Actual', color='blue', linewidth=2)
plt.plot(last_y_test.index, last_y_pred, label='Predicted', linestyle='--', color='red', linewidth=2)
plt.title('Weekly Portfolio Returns: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Sum of Weekly Returns')
plt.legend()
plt.grid(True)
plt.show()