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

# 按周进行重采样，计算平均值
weekly_avg = data.resample('W').mean()

# 放大GB_emotion的值
weekly_avg['GB_emotion_scaled'] = weekly_avg['GB_emotion'] * 10

weekly_avg['sz_50_lagged'] = weekly_avg['sz_50']

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
