import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Bidirectional, Conv1D
import optuna

# 加载数据
data_path = 'study1_final_all.csv'
data = pd.read_csv(data_path)
data['Date'] = pd.to_datetime(data['Date'])

# 数据预处理
imputer = KNNImputer(n_neighbors=5)
scaler = StandardScaler()

# 特征工程
features = data[['Date', 'GB_emotion']].copy()
features['GB_emotion'] = imputer.fit_transform(features[['GB_emotion']].values.reshape(-1, 1))
features = features.groupby('Date').mean().reset_index()
features_scaled = scaler.fit_transform(features['GB_emotion'].values.reshape(-1, 1))
features_scaled = np.reshape(features_scaled, (features_scaled.shape[0], 1, 1))

# 目标变量处理
target = data[['Date', 'Wretnd']].copy()
target['Wretnd'] = imputer.fit_transform(target[['Wretnd']].values.reshape(-1, 1))
target = target.groupby('Date').sum().reset_index()

# 为绘图保留日期
plot_dates = target['Date']

# Optuna目标函数
def objective(trial):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []

    for train_index, test_index in kf.split(features_scaled):
        X_train, X_test = features_scaled[train_index], features_scaled[test_index]
        y_train, y_test = target['Wretnd'].iloc[train_index], target['Wretnd'].iloc[test_index]

        # 构建模型
        model = tf.keras.Sequential([
            Input(shape=(1, 1)),
            Bidirectional(LSTM(trial.suggest_int("lstm_units", 50, 300))),
            Dropout(trial.suggest_uniform("dropout_rate", 0.1, 0.6)),
            Dense(1)
        ])

        # 编译模型
        lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mean_squared_error')

        # 训练模型
        model.fit(X_train, y_train, epochs=trial.suggest_int("epochs", 10, 100), batch_size=trial.suggest_categorical("batch_size", [16, 32, 64]), verbose=0)

        # 评估模型
        mse = model.evaluate(X_test, y_test, verbose=0)
        mse_scores.append(mse)

    return np.mean(mse_scores)

# 运行Optuna优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

# 使用最佳参数重新构建模型进行完整数据集的训练和预测
best_model = tf.keras.Sequential([
    Input(shape=(1, 1)),
    Bidirectional(LSTM(study.best_trial.params['lstm_units'])),
    Dropout(study.best_trial.params['dropout_rate']),
    Dense(1)
])

best_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=study.best_trial.params['learning_rate']), loss='mean_squared_error')
best_model.fit(features_scaled, target['Wretnd'].values, epochs=study.best_trial.params['epochs'], batch_size=study.best_trial.params['batch_size'], verbose=1)

# 预测并计算指标
predictions = best_model.predict(features_scaled).flatten()
mse = mean_squared_error(target['Wretnd'].values, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(target['Wretnd'].values, predictions)

# 显示指标
print(f"Final MSE: {mse}")
print(f"Final RMSE: {rmse}")
print(f"Final R2: {r2}")

# 绘制结果
plt.figure(figsize=(10, 5))
plt.plot(plot_dates, target['Wretnd'].values, label='True Values')
plt.plot(plot_dates, predictions, label='Predictions')
plt.title('Comparison of Predictions and True Values')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.savefig('fig222.png')
plt.show()
