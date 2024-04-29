import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# 读取数据
data_path = 'PanelData@0429.csv'
data = pd.read_csv(data_path)
original_columns = data.columns.tolist()  # 保存原始列的顺序

# 准备数据列
analyst_columns = ['Numpstisten_A', 'Numneusten_A', 'Numnegasten_A', 'Numsten_A']
gb_columns = ['Tpostnum_G', 'Pospostnum_G', 'Negpostnum_G', 'Readnum_G', 'Commentnum_G']

# 股吧情绪得分计算
def calculate_neutral_weight(pos_count, neg_count):
    return 0.3 if pos_count > neg_count else -0.3 if pos_count < neg_count else 0.1

data['GB_emotion'] = np.nan
if data[gb_columns].notna().all(axis=1).any():
    data.loc[data[gb_columns].notna().all(axis=1), 'NeutralWeight'] = data.apply(lambda x: calculate_neutral_weight(x['Pospostnum_G'], x['Negpostnum_G']), axis=1)
    data['GB_emotion'] = (3 * data['Pospostnum_G'] + data['NeutralWeight'] * data['Neupostnum_G'] - 3 * data['Negpostnum_G']) / \
                         (data['Pospostnum_G'] + data['Negpostnum_G'] + data['Neupostnum_G'])

# 准备PCA分析
#transformer = FunctionTransformer(np.log1p, validate=True)
#X_transformed = transformer.fit_transform(data.loc[data[gb_columns].notna().all(axis=1), gb_columns])
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X_transformed)
#pca = PCA(n_components=2)
#PCs = pca.fit_transform(X_scaled)

# 创建exog
#exog = pd.DataFrame(PCs, columns=['PC1', 'PC2'], index=data.loc[data[gb_columns].notna().all(axis=1)].index)
#exog = sm.add_constant(exog)

# 筛选数据
#data_with_analyst = data[data[analyst_columns].notna().all(axis=1) & data['Wretnd'].notna()]
#if not data_with_analyst.empty:
 #   X, y = exog.loc[data_with_analyst.index], data_with_analyst['Wretnd']

    # 梯度提升机
#   gbm_model.fit(X, y)
#    data['Analyst_emotion_GBM'] = np.nan
#    data.loc[data_with_analyst.index, 'Analyst_emotion_GBM'] = gbm_model.predict(X)

# 输出到新的CSV文件
output_columns = original_columns + ['GB_emotion', 'Analyst_emotion_GBM']  # 使用原始列顺序，添加新列至末尾
output_data = data[output_columns]
output_data.to_csv('study1.csv', index=False)

