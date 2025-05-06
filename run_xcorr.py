import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. 读取CSV文件
df = pd.read_csv('data/solar/solar_AL.csv')

# 2. 取第2-8列（即索引1-7）前96行
data = df.iloc[1:97, 1:30]

# 创建保存图片的文件夹
output_dir = 'pic'
os.makedirs(output_dir, exist_ok=True)

# 3. 对7列数据两两组合进行互相关计算
num_cols = data.shape[1]
column_names = data.columns

for i in range(num_cols):
    for j in range(i + 1, num_cols):  # 避免重复与自身
        x = data.iloc[:, i].values
        y = data.iloc[:, j].values

        # 计算互相关
        max_lag = 3  # 最大滞后值
        corr = np.correlate(x - np.mean(x), y - np.mean(y), mode='full')
        corr /= (np.std(x) * np.std(y) * len(x))  # 标准化
        lags = np.arange(-len(x) + 1, len(x))  # 计算滞后的范围

        # 只取中间部分 [-max_lag, max_lag]
        mid = len(corr) // 2
        corr = corr[mid - max_lag: mid + max_lag + 1]
        lags = lags[mid - max_lag: mid + max_lag + 1]

        # 绘制柱状图
        plt.figure(figsize=(10, 5))
        plt.bar(lags, corr, color='skyblue', edgecolor='k')
        plt.title(f"Cross-correlation (Bar): {column_names[i]} vs {column_names[j]}")
        plt.xlabel("Lag")
        plt.ylabel("Correlation")
        #plt.ylim(-1.1, 1.1)
        plt.grid(True, linestyle='--', alpha=0.5)

        # 保存图像
        filename = f"{column_names[i]}_vs_{column_names[j]}_bar.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath)
        plt.close()  # 关闭当前图，准备下一张
