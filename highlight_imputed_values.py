import pandas as pd
import numpy as np

# 读取原始数据和填充后的数据
df_original = pd.read_csv('juhe.csv')
df_imputed = pd.read_csv('imputed_data2.csv')

# 检测原始数据中的NaN值位置
nan_mask = df_original.isna()

# 创建一个新的DataFrame用于输出，初始值与原始数据相同
df_output = df_original.copy()

# 将DataFrame转换为字符串类型，以便后续处理
df_output = df_output.astype(str)

# 替换'nan'为空字符串
df_output = df_output.replace('nan', '')

# 遍历所有单元格，检查是否为NaN，如果是，则用加粗的填充值替换
for i in range(df_original.shape[0]):
    for j in range(df_original.shape[1]):
        if nan_mask.iloc[i, j]:
            # 获取填充后的值
            imputed_value = df_imputed.iloc[i, j]
            # 将填充值加粗（使用HTML标签）
            df_output.iloc[i, j] = f'<b>{imputed_value}</b>'

# 将结果保存为HTML文件
html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>填充值加粗显示</title>
    <style>
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
    </style>
</head>
<body>
    <h2>juhe.csv中NaN值的填充结果（填充值已加粗）</h2>
    {df_output.to_html(escape=False)}
</body>
</html>
'''

with open('imputed_data_bold.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print('已生成imputed_data_bold.html文件，其中填充值已加粗显示')