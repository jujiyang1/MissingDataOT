import pandas as pd
import numpy as np

# 元素符号到原子序数的映射字典
element_to_atomic_number = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
    'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}

def convert_csv_file():
    try:
        # 读取CSV文件
        print("正在读取fenxi_4.csv文件...")
        df = pd.read_csv('fenxi_4.csv', header=None)
        
        # 第一行作为索引行
        index_row = df.iloc[0].copy()
        
        # 从第二行开始处理数据（跳过索引行）
        data_df = df.iloc[1:].copy()
        
        # 将第7列（索引为6）的化学元素符号转换为原子序数
        print("正在将化学元素符号转换为原子序数...")
        
        def convert_element(element):
            if pd.isna(element) or element == '':
                return element
            return element_to_atomic_number.get(element, element)
        
        # 应用转换函数到第7列
        data_df.iloc[:, 6] = data_df.iloc[:, 6].apply(convert_element)
        
        # 将所有数据转换为float类型
        print("正在将所有数据转换为float类型...")
        for col in range(len(data_df.columns)):
            data_df.iloc[:, col] = pd.to_numeric(data_df.iloc[:, col], errors='coerce')
        
        # 创建结果DataFrame，保留原始索引行
        result_df = pd.concat([pd.DataFrame([index_row]), data_df], ignore_index=False)
        
        # 保存结果到新文件
        output_file = 'fenxi_4_converted.csv'
        result_df.to_csv(output_file, index=False, header=False)
        print(f"处理完成，结果已保存到 {output_file}")
        
        return True
    except Exception as e:
        print(f"处理过程中出错: {e}")
        return False

if __name__ == "__main__":
    convert_csv_file()