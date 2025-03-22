import pandas as pd
import re
from bs4 import BeautifulSoup
import os

# 读取HTML文件
with open('imputed_data_bold.html', 'r', encoding='utf-8') as f:
    html_content = f.read()

# 使用BeautifulSoup解析HTML
soup = BeautifulSoup(html_content, 'html.parser')

# 获取表格数据
table = soup.find('table')
rows = table.find_all('tr')

# 提取表头
headers = [th.text.strip() for th in rows[0].find_all('th')]

# 提取数据行
data = []
imputed_mask = []

for row in rows[1:]:  # 跳过表头行
    cols = row.find_all('td')
    row_data = []
    row_mask = []
    
    for col in cols:
        # 检查是否有加粗标签
        bold_tag = col.find('b')
        if bold_tag:
            # 这是一个填充值
            value = bold_tag.text.strip()
            row_data.append(value)
            row_mask.append(True)  # 标记为填充值
        else:
            # 这是原始值
            value = col.text.strip()
            row_data.append(value)
            row_mask.append(False)  # 标记为原始值
    
    data.append(row_data)
    imputed_mask.append(row_mask)

# 创建DataFrame
df = pd.DataFrame(data, columns=headers[1:])  # 跳过索引列

# 将数据转换为适当的类型
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except ValueError:
        # 如果无法转换为数字，保持为字符串
        pass

# 创建填充值掩码DataFrame
df_mask = pd.DataFrame(imputed_mask, columns=headers[1:])

# 保存为CSV文件
df.to_csv('imputed_data_from_html.csv', index=False)
print(f"数据已保存为CSV: {os.path.abspath('imputed_data_from_html.csv')}")

# 创建一个Excel writer对象
with pd.ExcelWriter('imputed_data_from_html.xlsx', engine='openpyxl') as writer:
    # 保存数据
    df.to_excel(writer, sheet_name='Data', index=False)
    
    # 获取工作簿和工作表
    workbook = writer.book
    worksheet = writer.sheets['Data']
    
    # 设置填充值的单元格格式（加粗）
    from openpyxl.styles import Font
    bold_font = Font(bold=True)
    
    # 应用格式
    for row_idx, row in enumerate(imputed_mask):
        for col_idx, is_imputed in enumerate(row):
            if is_imputed:
                # Excel行列索引从1开始，且要考虑表头行
                cell = worksheet.cell(row=row_idx+2, column=col_idx+1)
                cell.font = bold_font

print(f"数据已保存为Excel: {os.path.abspath('imputed_data_from_html.xlsx')}")

# 创建一个带有填充标记的CSV
df_with_markers = df.copy()

# 添加标记列，指示哪些值是填充的
for col in df.columns:
    marker_col = f"{col}_is_imputed"
    df_with_markers[marker_col] = df_mask[col]

# 保存带有标记的CSV
df_with_markers.to_csv('imputed_data_with_markers.csv', index=False)
print(f"带有填充标记的数据已保存为CSV: {os.path.abspath('imputed_data_with_markers.csv')}")

print("\n转换完成！生成了三个文件：")
print("1. imputed_data_from_html.csv - 基本CSV格式")
print("2. imputed_data_from_html.xlsx - Excel格式，填充值以加粗显示")
print("3. imputed_data_with_markers.csv - CSV格式，带有标记列指示哪些值是填充的")