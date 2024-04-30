import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from sklearn.inspection import plot_partial_dependence
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split

# 数据导入
data = pd.read_csv('study1_final_all.csv')
data = data.dropna(subset=['GB_emotion', 'Analyst_emotion', 'SVI_All'])

# 标准化特征
scaler = StandardScaler()
data[['GB_emotion', 'Analyst_emotion', 'SVI_All']] = scaler.fit_transform(data[['GB_emotion', 'Analyst_emotion', 'SVI_All']])

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 划分数据集
X = data[['GB_emotion', 'Analyst_emotion']]
y = data['SVI_All']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 预测和评估
y_pred = rf_model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R^2:", r2_score(y_test, y_pred))

# 特征重要性
importances = rf_model.feature_importances_
print("Feature importances:", importances)

# # 绘制部分依赖图
# features = ['GB_emotion', 'Analyst_emotion']
# fig, ax = plt.subplots(figsize=(12, 4))
# display = PartialDependenceDisplay.from_estimator(model, X_train, features, ax=ax)
# plt.show()
