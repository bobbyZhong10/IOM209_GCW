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

# 绘制情绪得分和股指回报率的折线图
plt.figure(figsize=(14, 8))
plt.plot(weekly_data.index, weekly_data['GB_emotion'], label='GB_emotion（普通投资者情绪得分）', color='blue')
plt.plot(weekly_data.index, weekly_data['Analyst_emotion'], label='Analyst_emotion（分析师情绪得分）', color='red')
plt.plot(weekly_data.index, weekly_data['sz_all'], label='Shanghai Index Return（上证指数回报率）', color='green')
plt.plot(weekly_data.index, weekly_data['sz_50'], label='Shanghai 50 Return（上证50回报率）', color='purple')

plt.title('2014-2016年每周投资者情绪与上证指数回报率的趋势')
plt.xlabel('日期')
plt.ylabel('值')
plt.legend()
plt.grid(True)
plt.show()
