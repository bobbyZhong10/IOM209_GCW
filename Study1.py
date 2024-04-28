import pandas as pd
import numpy as np  # 导入 numpy 用于处理 NaN

# 加载数据集
file_path = 'panel data.csv'
data = pd.read_csv(file_path)

# 定义一个函数来计算中性帖子的权重
def calculate_neutral_weight(pos_count, neg_count):
    """根据积极和消极帖子的数量，决定中性帖子的权重。"""
    if pos_count > neg_count:
        return 0.3
    elif pos_count < neg_count:
        return -0.3
    else:
        return 0.1

# 计算股吧情绪得分GB_emotion
C_gb = 1
D_gb = 1

# Calculate the GB_emotion score using the provided formula
data['NeutralWeight'] = data.apply(lambda row: calculate_neutral_weight(row['Pospostnum_G'], row['Negpostnum_G']), axis=1)
data['GB_emotion'] = ((3 * C_gb * data['Pospostnum_G'] +
                       data['NeutralWeight'] * data['Neupostnum_G'] -
                       3 * D_gb * data['Negpostnum_G']) /
                      (data['Pospostnum_G'] + data['Negpostnum_G'] + data['Neupostnum_G']))

# 计算分析师情绪得分JG_emotion score
C_jg = 0.343
D_jg = 0.396

# Calculate the JG_emotion score using the provided formula
data['JG_emotion'] = ((3 * C_jg * data['Numpstisten_A'] +
                       data['NeutralWeight'] * data['Numneusten_A'] -
                       3 * D_jg * data['Numnegasten_A']) /
                      (data['Numpstisten_A'] + data['Numneusten_A'] + data['Numnegasten_A']))

# Replacing infinities with np.nan for better handling in both emotion scores
data['GB_emotion'].replace([np.inf, -np.inf], np.nan, inplace=True)
data['JG_emotion'].replace([np.inf, -np.inf], np.nan, inplace=True)

# 查看更新后的数据，包括新的 GB_emotion 和 JG_emotion 列
print(data[['Scode', 'ShortName', 'Date', 'Pospostnum_G', 'Negpostnum_G', 'Neupostnum_G', 'Tpostnum_G', 'GB_emotion', 'JG_emotion']].head())

# 将处理后的数据保存到新文件
data.to_csv('output_with_emotions.csv', index=False)
