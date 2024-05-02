import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv('study1_final_all.csv')

# 将'Date'转换为日期格式，并设置为索引
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# 将数据按周进行重采样，计算数值列的均值，并将'Analyst_emotion'中的空值填充为0
weekly_data = data.resample('W').mean()
weekly_data['Analyst_emotion'].fillna(0, inplace=True)

# # 绘制情绪得分和股指回报率的折线图
# plt.figure(figsize=(14, 8))
# plt.plot(weekly_data.index, weekly_data['GB_emotion'], label='GB_emotion（普通投资者情绪得分）', color='blue')
# plt.plot(weekly_data.index, weekly_data['Analyst_emotion'], label='Analyst_emotion（分析师情绪得分）', color='red')
# plt.plot(weekly_data.index, weekly_data['sz_all'], label='Shanghai Index Return（上证指数回报率）', color='green')
# plt.plot(weekly_data.index, weekly_data['sz_50'], label='Shanghai 50 Return（上证50回报率）', color='purple')
#
# plt.title('2014-2016年每周投资者情绪与上证指数回报率的趋势')
# plt.xlabel('日期')
# plt.ylabel('值')
# plt.legend()
# plt.grid(True)
# plt.show()


# #延后一周
# 按周进行重采样，计算平均值
weekly_avg = data.resample('W').mean()

# 放大GB_emotion的值
weekly_avg['GB_emotion_scaled'] = weekly_avg['GB_emotion'] * 10

# # 将sz_50延后一周
# weekly_avg['sz_50_lagged'] = weekly_avg['sz_50'].shift(-1)

# 绘制GB_emotion与延后的sz_50
plt.figure(figsize=(14, 7))
plt.plot(weekly_avg.index, weekly_avg['sz_50_lagged'], label='Lagged Average sz_50', color='blue')
plt.plot(weekly_avg.index, weekly_avg['GB_emotion_scaled'], label='Scaled Average GB_emotion (x10)', color='red')
plt.title('Comparison of Scaled GB_emotion vs. Lagged sz_50 (2014-2016)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# 保存图表为PNG文件
plt.savefig('fig2.png', format='png')
plt.show()
