import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('imputed_data_maxabs.csv', header=None)

# 计算每列的缺失值数量和比例
nan_count = df.isna().sum()
nan_percentage = df.isna().mean() * 100  # 转换为百分比

# 创建一个存储结果的DataFrame
result = pd.DataFrame(index=['mean', 'variance', 'IQR', 'Q1', 'Q3', 'nan_count', 'nan_percentage'])

# 对每列进行统计计算
for col in df.columns:
    # 忽略NaN值
    col_data = df[col].dropna()
    
    # 如果列中有数据，则计算统计量
    if len(col_data) > 0:
        mean = col_data.mean()
        variance = col_data.var()
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        
        # 获取该列的缺失值统计
        nan_cnt = nan_count[col]
        nan_pct = nan_percentage[col]
        
        # 将结果添加到结果DataFrame中
        result[col] = [mean, variance, iqr, q1, q3, nan_cnt, nan_pct]
    else:
        # 如果列中没有数据，则填充NaN
        result[col] = [np.nan, np.nan, np.nan, np.nan, np.nan, nan_count[col], nan_percentage[col]]

# 打印结果
print("CSV文件统计结果（包含NaN值统计）:")
print("\n每列的均值、方差、四分位距和缺失值统计:")
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', 1000)  # 设置显示宽度
print(result)

# 保存结果到CSV文件
result.to_csv('fenxi_4_statistics_with_nan.csv')
print("\n结果已保存到 'fenxi_4_statistics_with_nan.csv'")