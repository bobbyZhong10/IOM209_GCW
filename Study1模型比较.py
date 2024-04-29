import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

# 设置Seaborn样式
sns.set(style="whitegrid")

# 读取数据
data_path = 'panel data.csv'
data = pd.read_csv(data_path)

# 准备数据列
analyst_columns = ['Numpstisten_A', 'Numneusten_A', 'Numnegasten_A', 'Numsten_A']
gb_columns = ['Tpostnum_G', 'Pospostnum_G', 'Negpostnum_G', 'Readnum_G', 'Commentnum_G']
data = data.dropna(subset=analyst_columns + ['Wretnd'] + gb_columns)

# 特征转换和标准化
X = data[gb_columns]
y = data['Wretnd']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA 降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 数据拆分
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 定义模型
models = {
    'OLS': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
}

# Hyperopt setup for Gradient Boosting
space = {
    'n_estimators': hp.choice('n_estimators', range(20, 201, 10)),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'max_depth': hp.choice('max_depth', range(3, 14, 1)),
    'subsample': hp.uniform('subsample', 0.5, 1.0),
    'min_samples_split': hp.choice('min_samples_split', range(2, 11, 1)),
    'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 15, 1)),
    'random_state': 42
}

def objective(params):
    model = GradientBoostingRegressor(**params)
    score = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5).mean()
    return {'loss': -score, 'status': STATUS_OK}

trials = Trials()
best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
best_params['n_estimators'] += 20
best_params['max_depth'] += 3
best_params['min_samples_split'] += 2
best_params['min_samples_leaf'] += 1
models['Gradient Boosting'] = GradientBoostingRegressor(**best_params)

# K折交叉验证设置
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 模型评估
performance = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    scores = cross_val_score(model, X_pca, y, cv=kf, scoring='r2')
    performance[name] = {
        'MSE': mse,
        'R²': r2,
        'Cross-validation R² Mean': np.mean(scores),
        'Cross-validation R² Std': np.std(scores)
    }

fig, ax = plt.subplots(2, 1, figsize=(10, 10))

# 鲁棒性：MSE和R²
mse_values = [performance[model]['MSE'] for model in models]
r2_values = [performance[model]['R²'] for model in models]
ax[0].bar(models.keys(), mse_values, color='blue', label='MSE')
ax[0].set_title('Robustness Comparison by MSE')
ax[0].set_ylabel('Mean Squared Error')
ax[0].legend(loc='upper left')

ax1 = ax[0].twinx()
ax1.plot(models.keys(), r2_values, color='red', label='R²', marker='o')
ax1.set_ylabel('R² Score')
ax1.legend(loc='upper right')

# K折交叉验证结果
cv_r2_means = [performance[model]['Cross-validation R² Mean'] for model in models]
sns.barplot(ax=ax[1], x=list(models.keys()), y=cv_r2_means, palette='muted')
ax[1].set_title('Cross-validation R² Mean Comparison')
ax[1].set_ylabel('R² Mean')

plt.tight_layout()
plt.savefig('Model_Comparison.png')  # 保存图像
plt.show()

# 输出数值结果
for model in performance:
    print(f"{model} Model Performance:")
    print(f"  MSE: {performance[model]['MSE']}")
    print(f"  R²: {performance[model]['R²']}")
    print(f"  Cross-validation R² Mean: {performance[model]['Cross-validation R² Mean']}")
    print(f"  Cross-validation R² Std: {performance[model]['Cross-validation R² Std']}")
