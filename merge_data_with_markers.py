import pandas as pd
import numpy as np

# 读取两个CSV文件
df_juhe = pd.read_csv("juhe.csv")
df_imputed = pd.read_csv("imputed_data1.csv")

# 确保两个数据框的形状相同
if df_juhe.shape != df_imputed.shape:
    print(f"警告：两个文件的形状不同！juhe.csv: {df_juhe.shape}, imputed_data1.csv: {df_imputed.shape}")

# 创建一个标记矩阵，记录哪些值是从imputed_data1.csv填充的
markers = pd.DataFrame(False, index=df_juhe.index, columns=df_juhe.columns)

# 创建合并后的数据框，初始值为juhe.csv的值
df_merged = df_juhe.copy()

# 对于juhe.csv中的每个NaN值，用imputed_data1.csv中对应位置的值填充
# 并在markers矩阵中标记这些位置
for col in df_juhe.columns:
    # 找出juhe.csv中该列的NaN值的索引
    nan_indices = df_juhe[col].isna()
    
    # 用imputed_data1.csv中对应位置的值填充
    df_merged.loc[nan_indices, col] = df_imputed.loc[nan_indices, col]
    
    # 在markers矩阵中标记这些位置
    markers.loc[nan_indices, col] = True

# 保存合并后的数据到CSV文件
df_merged.to_csv('merged_data.csv', index=False)

# 创建一个带有标记的HTML文件，其中被填充的值会被加粗显示
def style_cell(val, is_filled):
    if is_filled:
        return f'<b>{val}</b>'
    return val

# 将数据框转换为HTML表格，并根据markers矩阵添加样式
html_table = '<table border="1" cellspacing="0" cellpadding="5">'

# 添加表头
html_table += '<tr>'
for col in df_merged.columns:
    html_table += f'<th>{col}</th>'
html_table += '</tr>'

# 添加数据行
for i in range(len(df_merged)):
    html_table += '<tr>'
    for j, col in enumerate(df_merged.columns):
        val = df_merged.iloc[i, j]
        is_filled = markers.iloc[i, j]
        styled_val = style_cell(val, is_filled)
        html_table += f'<td>{styled_val}</td>'
    html_table += '</tr>'

html_table += '</table>'

# 保存HTML表格到文件
with open('imputed_data_with_markers.html', 'w') as f:
    f.write('<html><body>')
    f.write(html_table)
    f.write('</body></html>')

# 创建一个带有标记的CSV文件
# 不再添加特殊字符，直接保存原始数据，保持float属性
# 保存原始合并数据到CSV文件，不添加任何标记
df_merged.to_csv('imputed_data_with_markers.csv', index=False)

# 额外创建一个带有标记信息的CSV文件，用于参考
# 创建一个标记数据框，其中True表示填充的值，False表示原始值
markers.to_csv('imputed_data_markers_reference.csv', index=False)

print("处理完成！")
print(f"1. 合并后的数据已保存到 merged_data.csv")
print(f"2. 带有标记的HTML表格已保存到 imputed_data_with_markers.html (填充的值以加粗显示)")
print(f"3. 原始数据格式的CSV文件已保存到 imputed_data_with_markers.csv (保持float属性)")
print(f"4. 标记信息已保存到 imputed_data_markers_reference.csv (True表示填充的值，False表示原始值)")