import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 数据导入和预处理
data = pd.read_csv('study1_final_all.csv')
data.dropna(subset=['GB_emotion', 'Analyst_emotion', 'sz_all', 'sz_50'], inplace=True)

# 定义目标和特征
target = 'sz_all'  # 或者 'sz_50'
features = ['GB_emotion', 'Analyst_emotion']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.2, random_state=42)

# 回归任务的模型评估函数
def regression_objective(params):
    model = XGBRegressor(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        learning_rate=params['learning_rate'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        random_state=42
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    # 使用一个简单的平均分来综合考虑三个指标的效果
    loss = (rmse + mae + (1 - r2)) / 3
    return {'loss': loss, 'status': STATUS_OK}

# 定义超参数空间
space = {
    'max_depth': hp.quniform('max_depth', 3, 12, 1),
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 50),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.2),
    'subsample': hp.uniform('subsample', 0.7, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.7, 1)
}

# 运行 Hyperopt
trials = Trials()
best = fmin(fn=regression_objective, space=space, algo=tpe.suggest, max_evals=500, trials=trials)

# 使用找到的最佳超参数训练模型
best_model = XGBRegressor(
    n_estimators=int(best['n_estimators']),
    max_depth=int(best['max_depth']),
    learning_rate=best['learning_rate'],
    subsample=best['subsample'],
    colsample_bytree=best['colsample_bytree'],
    random_state=42
)
best_model.fit(X_train, y_train)
best_predictions = best_model.predict(X_test)

# 计算最终的评估指标
final_rmse = np.sqrt(mean_squared_error(y_test, best_predictions))
final_mae = mean_absolute_error(y_test, best_predictions)
final_r2 = r2_score(y_test, best_predictions)

print("测试集上的评估指标：")
print(f"最终 RMSE: {final_rmse}")
print(f"最终 MAE: {final_mae}")
print(f"最终 R2: {final_r2}")
print("最佳超参数：", best)
