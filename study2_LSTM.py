import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf

# 加载数据
data_path = 'study1_final_all.csv'
data = pd.read_csv(data_path)
data['Date'] = pd.to_datetime(data['Date']).dt.to_period('W').dt.to_timestamp()  # 转换为标准日期格式

# 数据预处理
features = data[['Date', 'GB_emotion']].copy()
features['GB_emotion'].fillna(features['GB_emotion'].mean(numeric_only=True), inplace=True)  # 解决 FutureWarning
target = data[['Date', 'Wretnd']].copy()
target['Wretnd'].fillna(target['Wretnd'].mean(numeric_only=True), inplace=True)  # 解决 FutureWarning

# 按日期聚合计算总回报率
target = target.groupby('Date').sum()
features = features.groupby('Date').mean()

# 标准化特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled = np.reshape(features_scaled, (features_scaled.shape[0], 1, features_scaled.shape[1]))

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42, shuffle=False)

# 构建 LSTM 模型
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.LSTM(50, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(20),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

# 编译模型
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# 训练模型
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

# 评估模型
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# 输出结果
print(f"MSE: {mse}, RMSE: {rmse}, R^2: {r2}")

# 保存预测结果
predicted_data = pd.DataFrame(y_pred, index=y_test.index, columns=['Predicted'])

# 绘制预测与实际结果
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label='Actual', color='blue', linewidth=2)
plt.plot(predicted_data.index, predicted_data['Predicted'], label='Predicted', linestyle='--', color='red', linewidth=2)
plt.title('Weekly Portfolio Returns: Actual vs Predicted')
plt.xlabel('Date')
plt.ylabel('Sum of Weekly Returns')
plt.legend()
plt.grid(True)
plt.savefig('fig3.png', format='png')
plt.show()
