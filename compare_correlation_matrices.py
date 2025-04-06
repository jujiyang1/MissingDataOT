#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
比较插补方法的相关系数矩阵

这个脚本比较imputed_data_maxabs.csv、imputed_data_mice.csv与imputed_data_rf_mice.csv三种插补方法的效果，
通过计算它们与cell_performance.csv中基于成对完整观测值计算的相关系数矩阵的差异。

评估方法：
1. 使用成对完整观测值计算原始数据的相关系数矩阵
2. 计算插补后数据的相关系数矩阵
3. 使用Frobenius范数计算差异
4. 比较哪种插补方法更好地保留了原始数据的相关结构
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def load_data(file_path):
    """
    加载CSV文件数据
    
    参数
    ----------
    file_path : str
        CSV文件路径
        
    返回
    -------
    pandas.DataFrame
        加载的数据框
    """
    try:
        print(f"读取数据文件: {file_path}")
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return None


def calculate_pairwise_correlation(data):
    """
    使用成对完整观测值计算相关系数矩阵
    
    参数
    ----------
    data : pandas.DataFrame
        包含缺失值的数据框
        
    返回
    -------
    pandas.DataFrame
        相关系数矩阵
    """
    # 只使用数值列
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # 使用成对完整观测值计算相关系数矩阵（pandas默认会使用成对完整观测值）
    return data[numeric_cols].corr(method='pearson')


def calculate_correlation_matrix(data):
    """
    计算完整数据的相关系数矩阵
    
    参数
    ----------
    data : pandas.DataFrame
        完整数据框（插补后的数据）
        
    返回
    -------
    pandas.DataFrame
        相关系数矩阵
    """
    # 只使用数值列
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # 计算相关系数矩阵
    return data[numeric_cols].corr(method='pearson')


def calculate_frobenius_norm(matrix1, matrix2):
    """
    计算两个矩阵之间的Frobenius范数差异
    
    参数
    ----------
    matrix1 : pandas.DataFrame
        第一个矩阵
    matrix2 : pandas.DataFrame
        第二个矩阵
        
    返回
    -------
    float
        Frobenius范数差异
    """
    # 确保两个矩阵有相同的列和行
    common_cols = matrix1.columns.intersection(matrix2.columns)
    
    # 计算Frobenius范数差异
    diff = np.linalg.norm(matrix1.loc[common_cols, common_cols] - matrix2.loc[common_cols, common_cols])
    
    return diff


def visualize_correlation_matrices(original_corr, maxabs_corr, mice_corr, rf_mice_corr):
    """
    可视化相关系数矩阵
    
    参数
    ----------
    original_corr : pandas.DataFrame
        原始数据的相关系数矩阵
    maxabs_corr : pandas.DataFrame
        MaxAbs插补后数据的相关系数矩阵
    mice_corr : pandas.DataFrame
        MICE插补后数据的相关系数矩阵
    rf_mice_corr : pandas.DataFrame
        RF-MICE插补后数据的相关系数矩阵
    """
    plt.figure(figsize=(20, 10))
    
    # 原始数据相关系数矩阵
    plt.subplot(2, 2, 1)
    sns.heatmap(original_corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('原始数据相关系数矩阵（成对完整观测值）')
    
    # MaxAbs插补数据相关系数矩阵
    plt.subplot(2, 2, 2)
    sns.heatmap(maxabs_corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('MaxAbs插补数据相关系数矩阵')
    
    # MICE插补数据相关系数矩阵
    plt.subplot(2, 2, 3)
    sns.heatmap(mice_corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('MICE插补数据相关系数矩阵')
    
    # RF-MICE插补数据相关系数矩阵
    plt.subplot(2, 2, 4)
    sns.heatmap(rf_mice_corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('RF-MICE插补数据相关系数矩阵')
    
    plt.tight_layout()
    plt.savefig('correlation_matrices_comparison.png')
    print("相关系数矩阵可视化已保存到 'correlation_matrices_comparison.png'")


def compare_correlation_matrices(original_file, maxabs_file, mice_file, rf_mice_file):
    """
    比较MaxAbs、MICE和RF-MICE三种插补方法的相关系数矩阵
    
    参数
    ----------
    original_file : str
        原始数据文件路径
    maxabs_file : str
        MaxAbs处理后的数据文件路径
    mice_file : str
        MICE处理后的数据文件路径
    rf_mice_file : str
        RF-MICE处理后的数据文件路径
    """
    try:
        # 加载数据
        original_df = load_data(original_file)
        maxabs_df = load_data(maxabs_file)
        mice_df = load_data(mice_file)
        rf_mice_df = load_data(rf_mice_file)
        
        if original_df is None or maxabs_df is None or mice_df is None or rf_mice_df is None:
            print("无法加载所有必要的数据文件")
            return
        
        # 使用成对完整观测值计算原始数据的相关系数矩阵
        print("\n使用成对完整观测值计算原始数据的相关系数矩阵")
        original_corr = calculate_pairwise_correlation(original_df)
        
        # 计算插补后数据的相关系数矩阵
        print("计算MaxAbs插补后数据的相关系数矩阵")
        maxabs_corr = calculate_correlation_matrix(maxabs_df)
        
        print("计算MICE插补后数据的相关系数矩阵")
        mice_corr = calculate_correlation_matrix(mice_df)
        
        print("计算RF-MICE插补后数据的相关系数矩阵")
        rf_mice_corr = calculate_correlation_matrix(rf_mice_df)
        
        # 计算Frobenius范数差异
        print("\n计算Frobenius范数差异")
        maxabs_diff = calculate_frobenius_norm(original_corr, maxabs_corr)
        mice_diff = calculate_frobenius_norm(original_corr, mice_corr)
        rf_mice_diff = calculate_frobenius_norm(original_corr, rf_mice_corr)
        
        print(f"MaxAbs插补方法与原始数据的Frobenius范数差异: {maxabs_diff:.4f}")
        print(f"MICE插补方法与原始数据的Frobenius范数差异: {mice_diff:.4f}")
        print(f"RF-MICE插补方法与原始数据的Frobenius范数差异: {rf_mice_diff:.4f}")
        
        # 判断哪种方法更好
        min_diff = min(maxabs_diff, mice_diff, rf_mice_diff)
        
        if min_diff == maxabs_diff:
            conclusion = "MaxAbs插补方法在保持相关系数矩阵方面表现最好，更接近原始数据分布"
            winner = "MaxAbs"
        elif min_diff == mice_diff:
            conclusion = "MICE插补方法在保持相关系数矩阵方面表现最好，更接近原始数据分布"
            winner = "MICE"
        elif min_diff == rf_mice_diff:
            conclusion = "RF-MICE插补方法在保持相关系数矩阵方面表现最好，更接近原始数据分布"
            winner = "RF-MICE"
        else:
            conclusion = "三种方法在保持相关系数矩阵方面表现相当"
            winner = "平局"
        
        print(f"\n结论: {conclusion}")
        
        # 创建详细比较表格
        comparison_table = [{
            '评估指标': '相关系数矩阵Frobenius范数差异',
            'MaxAbs差异': maxabs_diff,
            'MICE差异': mice_diff,
            'RF-MICE差异': rf_mice_diff,
            '胜者': winner
        }]
        
        # 创建比较表格DataFrame并保存
        comparison_df = pd.DataFrame(comparison_table)
        comparison_df.to_csv('correlation_matrices_comparison_results.csv', index=False, encoding='utf-8-sig')
        print("\n详细比较结果已保存到 'correlation_matrices_comparison_results.csv'")
        
        # 可视化相关系数矩阵
        print("\n生成相关系数矩阵可视化...")
        visualize_correlation_matrices(original_corr, maxabs_corr, mice_corr, rf_mice_corr)
        
        # 返回结果字典
        return {
            'original_corr': original_corr,
            'maxabs_corr': maxabs_corr,
            'mice_corr': mice_corr,
            'rf_mice_corr': rf_mice_corr,
            'maxabs_diff': maxabs_diff,
            'mice_diff': mice_diff,
            'rf_mice_diff': rf_mice_diff,
            'winner': winner
        }
        
    except Exception as e:
        print(f"比较过程中出错: {e}")
        return None




if __name__ == "__main__":
    # 设置文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    original_file = os.path.join(current_dir, "cell_performance.csv")
    maxabs_file = os.path.join(current_dir, "imputed_data_maxabs.csv")
    mice_file = os.path.join(current_dir, "imputed_data_mice.csv")
    rf_mice_file = os.path.join(current_dir, "imputed_data_rf_mice.csv")
    
    # 比较相关系数矩阵
    print("比较插补方法的相关系数矩阵...\n")
    compare_correlation_matrices(original_file, maxabs_file, mice_file, rf_mice_file)