import pandas as pd
import numpy as np

# 读取juhe.csv文件
df = pd.read_csv('juhe.csv')

# 检测NaN值的位置
nan_mask = df.isna()

# 获取所有NaN值的位置（行索引和列索引）
nan_positions = []
for col in df.columns:
    # 获取该列中所有NaN值的行索引
    nan_rows = df.index[df[col].isna()].tolist()
    if nan_rows:
        for row in nan_rows:
            nan_positions.append((row, col))

# 打印NaN值的位置信息
print(f"在juhe.csv中共发现{len(nan_positions)}个NaN值")
print("\nNaN值的位置（行索引，列名）:")
for pos in nan_positions:
    print(f"行 {pos[0]}, 列 '{pos[1]}'")

# 统计每列的NaN值数量
nan_counts = nan_mask.sum()
print("\n每列的NaN值数量:")
for col, count in nan_counts.items():
    if count > 0:
        print(f"列 '{col}': {count}个NaN值")

# 计算NaN值的百分比
total_cells = df.shape[0] * df.shape[1]
nan_percentage = (len(nan_positions) / total_cells) * 100
print(f"\n数据集中NaN值的百分比: {nan_percentage:.2f}%")

# 将NaN位置信息保存到文件
with open('nan_positions.txt', 'w') as f:
    f.write(f"在juhe.csv中共发现{len(nan_positions)}个NaN值\n\n")
    f.write("NaN值的位置（行索引，列名）:\n")
    for pos in nan_positions:
        f.write(f"行 {pos[0]}, 列 '{pos[1]}'\n")
    
    f.write("\n每列的NaN值数量:\n")
    for col, count in nan_counts.items():
        if count > 0:
            f.write(f"列 '{col}': {count}个NaN值\n")
    
    f.write(f"\n数据集中NaN值的百分比: {nan_percentage:.2f}%\n")

print("\nNaN位置信息已保存到 'nan_positions.txt' 文件中")