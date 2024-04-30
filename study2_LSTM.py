import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

Sequential = tf.keras.models.Sequential
load_model = tf.keras.models.load_model

from tensorflow.keras.layers import LSTM, Dense, Dropout

# 加载数据
data_path = 'study1_final_all.csv'  # 更新为你的CSV文件路径
data = pd.read_csv(data_path)

# 数据预处理
# 示例特征
features = data[['GB_emotion', 'Analyst_emotion']]
target = data['sz_all']

# 检查并清除数据中的NaN值
features.fillna(features.mean(), inplace=True)
target.fillna(target.mean(), inplace=True)

# 标准化特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
features_scaled = np.reshape(features_scaled, (features_scaled.shape[0], 1, features_scaled.shape[1]))

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.2, random_state=42)

# 构建 LSTM 模型
model = Sequential([
    LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.3),  # 增加Dropout比率
    LSTM(20),
    Dropout(0.3),
    Dense(1)
])

# 编译模型，调整学习率
model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# 评估模型
performance = model.evaluate(X_test, y_test)
print("Test Loss:", performance)

# 保存模型
model.save('my_model.keras')

# 预测函数
def predict_with_model(model_path, scaler, new_data):
    model = load_model(model_path)
    new_data_scaled = scaler.transform(new_data)
    new_data_scaled = np.reshape(new_data_scaled, (new_data_scaled.shape[0], 1, new_data_scaled.shape[1]))
    predictions = model.predict(new_data_scaled)
    return predictions
