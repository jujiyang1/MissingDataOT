import pandas as pd
import numpy as np
import os

# 读取Excel文件
try:
    # 检查文件是否存在
    file_path = 'fenxi_4.2.xlsx'
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        exit(1)
    
    print(f"尝试读取文件: {file_path}")
    print(f"文件大小: {os.path.getsize(file_path)} 字节")
    
    # 使用openpyxl引擎读取Excel文件
    df = pd.read_excel(file_path, engine='openpyxl')
    print("成功读取Excel文件")
    print(f"数据形状: {df.shape}")
    
    # 查看前几行数据
    print("\n数据前5行:")
    print(df.head())
    
    # 检查第7列(索引为6)的数据
    print("\n第7列(索引为6)的数据类型和前10个值:")
    print(f"数据类型: {df.iloc[:, 6].dtype}")
    print(df.iloc[:, 6].head(10))
    
    # 创建化学元素到数值的映射
    unique_elements = df.iloc[:, 6].dropna().unique()
    element_to_num = {element: i+1 for i, element in enumerate(unique_elements)}
    num_to_element = {i+1: element for i, element in enumerate(unique_elements)}
    
    # 保存映射关系
    with open('element_mapping.txt', 'w') as f:
        f.write("化学元素到数值的映射关系:\n")
        for element, num in element_to_num.items():
            f.write(f"{element} -> {num}\n")
    
    # 创建数据副本并转换第7列
    df_converted = df.copy()
    df_converted.iloc[:, 6] = df.iloc[:, 6].map(element_to_num)
    
    # 计算每列的统计指标
    stats = pd.DataFrame()
    stats['列名'] = df.columns
    stats['缺失值数量'] = df.isna().sum().values
    stats['缺失值比率'] = df.isna().mean().values
    stats['均值'] = df.mean(numeric_only=True).reindex(index=df.columns).values
    stats['方差'] = df.var(numeric_only=True).reindex(index=df.columns).values
    stats['标准差'] = df.std(numeric_only=True).reindex(index=df.columns).values
    stats['最小值'] = df.min(numeric_only=True).reindex(index=df.columns).values
    stats['25%分位数'] = df.quantile(0.25, numeric_only=True).reindex(index=df.columns).values
    stats['中位数'] = df.median(numeric_only=True).reindex(index=df.columns).values
    stats['75%分位数'] = df.quantile(0.75, numeric_only=True).reindex(index=df.columns).values
    stats['最大值'] = df.max(numeric_only=True).reindex(index=df.columns).values
    
    # 保存统计结果
    stats.to_csv('column_statistics_with_elements.csv', index=False, encoding='utf-8-sig')
    print("\n已保存列统计信息到 column_statistics_with_elements.csv")
    
    # 保存转换后的数据
    df_converted.to_csv('converted_data.csv', index=False, encoding='utf-8-sig')
    print("已保存转换后的数据到 converted_data.csv")
    
    # 打印统计信息
    print("\n列统计信息:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(stats)
    
except Exception as e:
    print(f"处理数据时出错: {e}")