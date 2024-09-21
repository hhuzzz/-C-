import pandas as pd
import numpy as np
from scipy.fftpack import fft

# 假设数据存储在csv文件中，先读取数据
# df = pd.read_csv('data.csv')  # 如果是csv文件可以用此方法读取
df = pd.DataFrame(np.random.randn(13000, 1024))  # 这里用随机数据模拟

# 定义计算峰值因子的函数
def crest_factor(series):
    peak_value = np.max(np.abs(series))
    rms_value = np.sqrt(np.mean(series**2))
    return peak_value / rms_value if rms_value != 0 else np.nan

# 定义计算自相关系数的函数
def autocorr(series, lag=1):
    n = len(series)
    series_mean = np.mean(series)
    autocov = np.sum((series[:n-lag] - series_mean) * (series[lag:] - series_mean)) / n
    autocorr_coef = autocov / np.var(series)
    return autocorr_coef

# 使用列表存储所有特征行
features_list = []

# 对1024列进行特征提取
for col in df.columns:
    col_data = df[col]

    # 计算特征
    mean_val = col_data.mean()
    std_val = col_data.std()
    max_val = col_data.max()
    min_val = col_data.min()
    crest_fact = crest_factor(col_data)
    autocorr_coef = autocorr(col_data)
    
    # 傅里叶变换，只取前几个主要频率特征
    fft_vals = fft(col_data)
    fft_magnitude = np.abs(fft_vals)[:10]  # 取前10个频率成分

    # 将特征存入字典中
    feature_dict = {
        'mean': mean_val,
        'std_dev': std_val,
        'max': max_val,
        'min': min_val,
        'crest_factor': crest_fact,
        'autocorr_coef': autocorr_coef,
        'fft_1': fft_magnitude[0],
        'fft_2': fft_magnitude[1],
        'fft_3': fft_magnitude[2],
        'fft_4': fft_magnitude[3],
        'fft_5': fft_magnitude[4]
    }

    # 添加特征字典到列表中
    features_list.append(feature_dict)

# 将特征列表转换为DataFrame
features_df = pd.DataFrame(features_list)

# 显示最终的特征表
print(features_df)

# 可以将结果保存为csv
# features_df.to_csv('extracted_features.csv', index=False)
