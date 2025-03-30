#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
比较Sinkhorn与MICE插补方法的统计指标

这个脚本实现以下功能：
1. 读取原始数据(cell_performance.csv)和两种插补方法处理后的数据(imputed_data_maxabs.csv和imputed_data_mice.csv)
2. 提取原始数据中的非缺失值作为基准
3. 计算每种插补方法在各列上的统计指标(均值、中位数、方差、四分位距、偏度)
4. 比较插补后数据与原始数据的统计指标差异
5. 生成比较报告
"""

import pandas as pd
import numpy as np
import os
import logging
from scipy import stats

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


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
        logger.info(f"读取数据文件: {file_path}")
        return pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"读取文件 {file_path} 时出错: {e}")
        return None


def calculate_statistics(data):
    """
    计算数据框每列的统计指标
    
    参数
    ----------
    data : pandas.DataFrame
        需要计算统计指标的数据框
        
    返回
    -------
    pandas.DataFrame
        包含每列统计指标的数据框
    """
    stats_dict = {}
    
    for col in data.columns:
        # 跳过非数值列
        if not pd.api.types.is_numeric_dtype(data[col]):
            continue
            
        # 获取非缺失值
        valid_data = data[col].dropna()
        
        # 如果列中没有有效数据，则跳过
        if len(valid_data) == 0:
            continue
            
        # 计算统计指标
        stats_dict[col] = {
            '均值': valid_data.mean(),
            '中位数': valid_data.median(),
            '方差': valid_data.var(),
            '四分位距': valid_data.quantile(0.75) - valid_data.quantile(0.25),
            '偏度': stats.skew(valid_data)
        }
    
    # 转换为DataFrame
    return pd.DataFrame(stats_dict).T


def calculate_differences(original_stats, imputed_stats):
    """
    计算插补后数据与原始数据的统计指标差异
    
    参数
    ----------
    original_stats : pandas.DataFrame
        原始数据的统计指标
    imputed_stats : pandas.DataFrame
        插补后数据的统计指标
        
    返回
    -------
    pandas.DataFrame
        统计指标差异
    """
    # 确保两个DataFrame有相同的索引
    common_indices = original_stats.index.intersection(imputed_stats.index)
    
    # 计算绝对差异
    abs_diff = (imputed_stats.loc[common_indices] - original_stats.loc[common_indices]).abs()
    
    # 计算相对差异 (百分比)
    # 添加一个小的epsilon值，避免除以0的情况
    epsilon = 1e-10
    denominator = original_stats.loc[common_indices].abs() + epsilon
    rel_diff = abs_diff.div(denominator) * 100
    
    # 对于原始值接近0的情况，如果绝对差异也很小，则将相对差异设为0
    # 这样可以避免在原始值接近0但差异也很小的情况下出现很大的相对差异
    small_threshold = 1e-8
    for col in rel_diff.columns:
        for idx in rel_diff.index:
            if abs(original_stats.loc[idx, col]) < small_threshold and abs_diff.loc[idx, col] < small_threshold:
                rel_diff.loc[idx, col] = 0
    
    return abs_diff, rel_diff


def extract_observed_values(data):
    """
    从原始数据中提取非缺失值作为基准
    
    参数
    ----------
    data : pandas.DataFrame
        原始数据
        
    返回
    -------
    pandas.DataFrame
        只包含非缺失值的数据框
    """
    # 创建一个与原始数据相同的DataFrame
    observed_data = data.copy()
    
    # 只保留非缺失值
    for col in observed_data.columns:
        # 跳过非数值列
        if not pd.api.types.is_numeric_dtype(observed_data[col]):
            continue
            
        # 获取该列的非缺失值的索引
        valid_indices = observed_data[col].notna()
        
        # 对于缺失值的行，将整行设为NaN
        observed_data.loc[~valid_indices, :] = np.nan
    
    return observed_data


def compare_imputation_methods(original_file, sinkhorn_file, mice_file, output_file):
    """
    比较Sinkhorn和MICE两种插补方法的统计指标
    
    参数
    ----------
    original_file : str
        原始数据文件路径
    sinkhorn_file : str
        Sinkhorn处理后的数据文件路径
    mice_file : str
        MICE处理后的数据文件路径
    output_file : str
        输出结果文件路径
    """
    try:
        # 加载数据
        original_df = load_data(original_file)
        sinkhorn_df = load_data(sinkhorn_file)
        mice_df = load_data(mice_file)
        
        if original_df is None or sinkhorn_df is None or mice_df is None:
            logger.error("无法加载所有必要的数据文件")
            return
        
        # 提取原始数据中的非缺失值作为基准
        logger.info("提取原始数据中的非缺失值作为基准")
        observed_df = extract_observed_values(original_df)
        
        # 计算统计指标
        logger.info("计算原始数据的统计指标")
        observed_stats = calculate_statistics(observed_df)
        
        logger.info("计算Sinkhorn插补后数据的统计指标")
        sinkhorn_stats = calculate_statistics(sinkhorn_df)
        
        logger.info("计算MICE插补后数据的统计指标")
        mice_stats = calculate_statistics(mice_df)
        
        # 计算差异
        logger.info("计算Sinkhorn插补方法与原始数据的统计指标差异")
        sinkhorn_abs_diff, _ = calculate_differences(observed_stats, sinkhorn_stats)
        
        logger.info("计算MICE插补方法与原始数据的统计指标差异")
        mice_abs_diff, _ = calculate_differences(observed_stats, mice_stats)
        
        # 创建结果报告
        logger.info("创建比较报告")
        
        # 只保留需要的统计指标（均值、中位数、方差、四分位距）
        metrics_to_keep = ['均值', '中位数', '方差', '四分位距']
        sinkhorn_abs_diff = sinkhorn_abs_diff[metrics_to_keep]
        mice_abs_diff = mice_abs_diff[metrics_to_keep]
        
        # 保存结果到Excel文件
        with pd.ExcelWriter(output_file) as writer:
            observed_stats[metrics_to_keep].to_excel(writer, sheet_name='原始数据统计指标')
            sinkhorn_stats[metrics_to_keep].to_excel(writer, sheet_name='Sinkhorn统计指标')
            mice_stats[metrics_to_keep].to_excel(writer, sheet_name='MICE统计指标')
            sinkhorn_abs_diff.to_excel(writer, sheet_name='Sinkhorn绝对误差')
            mice_abs_diff.to_excel(writer, sheet_name='MICE绝对误差')
        
        logger.info(f"比较结果已保存到: {output_file}")
        
        # 打印每列的绝对误差对比
        print("\nSinkhorn方法的绝对误差:")
        print(sinkhorn_abs_diff)
        
        print("\nMICE方法的绝对误差:")
        print(mice_abs_diff)
        
    except Exception as e:
        logger.error(f"比较过程中出错: {e}")


if __name__ == "__main__":
    # 设置文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    original_file = os.path.join(current_dir, "cell_performance.csv")
    sinkhorn_file = os.path.join(current_dir, "imputed_data_maxabs.csv")
    mice_file = os.path.join(current_dir, "imputed_data_mice.csv")
    output_file = os.path.join(current_dir, "imputation_comparison_results.xlsx")
    
    # 比较插补方法
    compare_imputation_methods(original_file, sinkhorn_file, mice_file, output_file)