import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

# 数据导入和预处理
data = pd.read_csv('study1_final_all.csv')
data = data.dropna(subset=['GB_emotion', 'Analyst_emotion', 'SVI_All'])

# 标准化数据
scaler = StandardScaler()
features = ['GB_emotion', 'Analyst_emotion']
data[features + ['SVI_All']] = scaler.fit_transform(data[features + ['SVI_All']])

# 划分数据集
X = data[features]
y = data['SVI_All']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def objective(params):
    model = GradientBoostingRegressor(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {'loss': mse, 'status': STATUS_OK}

# 定义搜索空间
space = {
    'max_depth': hp.choice('max_depth', range(1, 11)),
    'n_estimators': hp.choice('n_estimators', range(50, 300)),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    'subsample': hp.uniform('subsample', 0.7, 1.0)
}

# 运行 Hyperopt
trials = Trials()
best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=500, trials=trials)

# 最佳超参数
print("Best Hyperparameters:", best_params)

# 从索引转换为实际参数值
real_max_depth = best_params['max_depth'] + 1  # 因为range是从1开始的
real_n_estimators = best_params['n_estimators'] + 50  # 因为range是从50开始的

# 训练最终模型
final_model = GradientBoostingRegressor(
    n_estimators=real_n_estimators,
    max_depth=real_max_depth,
    learning_rate=best_params['learning_rate'],
    subsample=best_params['subsample'],
    random_state=42
)
final_model.fit(X_train, y_train)
y_pred_final = final_model.predict(X_test)

# 性能评估
final_mse = mean_squared_error(y_test, y_pred_final)
final_r2 = r2_score(y_test, y_pred_final)
print("Final MSE:", final_mse)
print("Final R2:", final_r2)

# 特征重要性
feature_importances = final_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

print(importance_df)
