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

# 分析师情绪得分计算
a = 0.342855001  # 无用积极词汇比
b = 0.3956291    # 无用消极词汇比

def calculate_adjusted_emotion_score_handle_zero_neutral(A, B, C, a, b):
    neutral_sentiment = 0
    if C > 0:
        if A/C > 1:
            neutral_sentiment = 0.3
        elif A/C < 1:
            neutral_sentiment = -0.3
        elif A/C == 1:
            neutral_sentiment = 0.1
    A_adjusted = 3 * A * (1 - a)
    B_adjusted = -3 * C * (1 - b)
    total_count = A + C + (B if B > 0 else 0)
    emotion_score = (A_adjusted + neutral_sentiment * C + B_adjusted) / total_count if total_count > 0 else 0
    return emotion_score

data['Analyst_emotion'] = data.apply(
    lambda row: calculate_adjusted_emotion_score_handle_zero_neutral(
        row['Numpstisten_A'], row['Numneusten_A'], row['Numnegasten_A'], a, b
    ) if not pd.isnull(row['Numpstisten_A']) and not pd.isnull(row['Numnegasten_A']) and not pd.isnull(row['Numneusten_A']) else np.nan,
    axis=1
)

# 输出到新的CSV文件
output_columns = original_columns + ['GB_emotion', 'Analyst_emotion']  # 使用原始列顺序，添加新列至末尾
output_data = data[output_columns]
output_data.to_csv('study1_final_all.csv', index=False)