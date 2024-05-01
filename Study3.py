import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv('study1_final_all.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
weekly_data = data.resample('W').mean()


# Calculate the correlation matrix
correlation_matrix = weekly_data[['SVI_code', 'SVI_All', 'GB_emotion', 'Analyst_emotion']].corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap of Correlation Coefficients between Emotions and Search Volumes')
plt.show()

# Perform Engle-Granger cointegration test
y = weekly_data['SVI_code'].dropna()
X = weekly_data['GB_emotion'].dropna()
result = coint(X, y)

# Print the results of the cointegration test
print('Cointegration test t-statistic:', result[0])
print('Critical values (1%, 5%, 10%):', result[1])
print('Number of lags used:', result[2])

# Assess the robustness of the original correlation coefficient using Bootstrap
original_corr = weekly_data['GB_emotion'].corr(weekly_data['SVI_code'])
bootstrapped_corrs = []
for _ in range(1000):  # Perform 1000 resampling
    sample = resample(weekly_data[['GB_emotion', 'SVI_code']])
    bootstrapped_corr = sample['GB_emotion'].corr(sample['SVI_code'])
    bootstrapped_corrs.append(bootstrapped_corr)

# Calculate the 95% confidence interval
conf_interval = np.percentile(bootstrapped_corrs, [2.5, 97.5])
print('Original correlation coefficient:', original_corr)
print('Bootstrap 95% confidence interval:', conf_interval)