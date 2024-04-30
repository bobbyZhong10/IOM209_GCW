import pandas as pd
from sklearn.decomposition import FactorAnalysis
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity

# 数据导入
data = pd.read_csv('study1_final_all.csv')

# 清洗数据，去除缺失值
data = data.dropna(subset=['GB_emotion', 'Analyst_emotion'])

# 检查KMO和Bartlett的球形度测试，确认数据适合因子分析
kmo_all, kmo_model = calculate_kmo(data[['GB_emotion', 'Analyst_emotion']])
chi_square_value, p_value = calculate_bartlett_sphericity(data[['GB_emotion', 'Analyst_emotion']])
print("KMO test value:", kmo_model)  # KMO值应大于0.6才适合进行因子分析
print("Bartlett's test p-value:", p_value)  # p值应小于0.05，表明数据间有足够的关联适合进行因子分析

# 因子分析
fa = FactorAnalyzer(n_factors=1, rotation=None)
fa.fit(data[['GB_emotion', 'Analyst_emotion']])

# 获取因子载荷
loadings = fa.loadings_
print("Factor Loadings:\n", loadings)

# 可视化因子载荷
plt.figure(figsize=(6, 4))
plt.bar(['GB_emotion', 'Analyst_emotion'], loadings.flatten(), color='skyblue')
plt.title('Factor Loadings')
plt.ylabel('Loadings')
plt.show()

# 解释的方差
ev, v = fa.get_eigenvalues()
print("Eigenvalues:\n", ev)

# 绘制特征值以决定因子数
plt.scatter(range(1, data[['GB_emotion', 'Analyst_emotion']].shape[1] + 1), ev)
plt.plot(range(1, data[['GB_emotion', 'Analyst_emotion']].shape[1] + 1), ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()
