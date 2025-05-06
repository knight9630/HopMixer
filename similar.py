import pandas as pd
import numpy as np

# 1. 读取 Excel 文件
df = pd.read_excel('data/solar/spart.xlsx', engine='openpyxl')

# 2. 计算列之间的相关系数矩阵
corr_matrix = df.corr()

# 3. 计算每列与其他列的平均相似度
# 忽略对角线上的相关性（每列与自己是完全相关的）
mean_similarity = corr_matrix.apply(lambda col: col[~col.index.isin([col.name])].mean(), axis=0)

# 4. 选择与其他列平均相似度较高的列
# 设置一个阈值，只保留相似度较高的列。我们可以选择保留前30列或者选择平均相似度高于某个值的列
top_30_columns = mean_similarity.nlargest(30).index

# 5. 只保留最相似的30列
df_filtered = df[top_30_columns]

# 6. 将结果保存到新的 Excel 文件
df_filtered.to_excel('data/solar/filtered_top_30_columns.xlsx', index=False)

print("已保存最相关的30列到 filtered_top_30_columns.xlsx 文件。")
