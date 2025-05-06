import pandas as pd

# 读取txt文件（假设数据由制表符分隔）
df = pd.read_csv('data/solar/solar_AL.txt', delimiter=',', header=None)

# 将数据保存为csv文件
df.to_csv('data/solar/solar_AL.csv', index=False, header=False)  # 如果不需要索引和标题行

