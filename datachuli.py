import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 读取CSV文件
data_1 = pd.read_csv('chabu.csv')

# 使用LabelEncoder将dapant列转换为数值编码
label_encoder = LabelEncoder()
data_1['dapant'] = label_encoder.fit_transform(data_1['dapant'])

print("\n转换为数值编码后的数据前5行：")
print(data_1.head())
print(data_1.shape)

# 打印编码映射关系
print("\n元素符号到数值的映射关系：")
for i, label in enumerate(label_encoder.classes_):
    print(f"{label}: {i}")

# 将处理后的数据导出为新的CSV文件
data_1.to_csv('processed_data.csv', index=False)
print("\n数据已成功导出到 processed_data.csv 文件中")
