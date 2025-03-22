import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv("juhe.csv")

# 计算每列忽略缺失值后的均值
mean_values = df.mean(skipna=True)
print("每列忽略缺失值后的均值:")
print(mean_values)

# 计算每列忽略缺失值后的标准差
std_values = df.std(skipna=True, ddof=1)  # 使用无偏估计 (N-1)
print("\n每列忽略缺失值后的标准差:")
print(std_values)

# 计算每列非缺失值的数量和缺失值的数量
non_null_counts = df.count()
null_counts = df.isnull().sum()
print("\n每列非缺失值的数量:")
print(non_null_counts)
print("\n每列缺失值的数量:")
print(null_counts)

# 计算每列缺失值的百分比
missing_percentage = (null_counts / len(df)) * 100
print("\n每列缺失值的百分比:")
print(missing_percentage.round(2), "%")

# 将结果保存到CSV文件
results = pd.DataFrame({
    '列名': df.columns,
    '均值': mean_values,
    '标准差': std_values,
    '非缺失值数量': non_null_counts,
    '缺失值数量': null_counts,
    '缺失值百分比(%)': missing_percentage.round(2)
})

# 保存结果
results.to_csv('column_statistics.csv', index=False, encoding='utf-8-sig')
print("\n统计结果已保存到 column_statistics.csv 文件")

# 显示结果表格
print(results)