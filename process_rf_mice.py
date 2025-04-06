#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据处理脚本：使用MaxAbsScaler归一化和随机森林MICE填补缺失值

这个脚本实现以下功能：
1. 读取cell_performance.csv数据文件
2. 对数据进行MaxAbsScaler归一化（忽略NaN值）
3. 使用随机森林MICE算法填补缺失值
4. 将填补后的数据还原回原始尺度
5. 保存结果到CSV文件
"""

import pandas as pd
import numpy as np
import torch
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# 设置PyTorch默认张量类型
torch.set_default_tensor_type('torch.DoubleTensor')

# 导入MICE相关模块和随机森林
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


class MaxAbsScaler:
    """
    MaxAbsScaler实现
    将每个特征缩放到[-1, 1]范围内，通过除以每个特征的最大绝对值
    """
    
    def __init__(self):
        """
        初始化MaxAbsScaler类
        """
        self.max_abs_ = None  # 存储每个特征的最大绝对值
        self.mean_ = None    # 与DataScaler兼容
        self.std_ = None     # 与DataScaler兼容
        self.original_shape = None  # 存储原始数据的形状
        
    def fit(self, data):
        """
        计算并存储每个特征的最大绝对值，用于后续的缩放操作
        
        参数
        ----------
        data : numpy.ndarray 或 torch.Tensor
            需要进行归一化的原始数据
            
        返回
        -------
        self : 返回自身实例，支持链式调用
        """
        # 保存原始数据形状
        self.original_shape = data.shape
        
        # 转换为numpy数组进行计算
        if torch.is_tensor(data):
            data_np = data.detach().cpu().numpy()
        else:
            data_np = np.array(data)
            
        # 计算每个特征的最大绝对值
        self.max_abs_ = np.nanmax(np.abs(data_np), axis=0)
        
        # 处理最大绝对值为0的情况
        self.max_abs_[self.max_abs_ == 0] = 1.0
        
        # 保存均值和标准差，用于与DataScaler兼容
        self.mean_ = np.zeros_like(self.max_abs_)
        self.std_ = self.max_abs_
        
        return self
    
    def transform(self, data):
        """
        使用最大绝对值缩放对数据进行归一化处理
        
        参数
        ----------
        data : numpy.ndarray 或 torch.Tensor
            需要进行归一化的原始数据
            
        返回
        -------
        scaled_data : 与输入相同类型的归一化后的数据
        """
        # 检查是否已经拟合
        if self.max_abs_ is None:
            raise ValueError("必须先调用fit方法计算最大绝对值")
        
        # 判断输入类型
        is_torch = torch.is_tensor(data)
        
        # 转换为numpy数组进行处理
        if is_torch:
            data_np = data.detach().cpu().numpy()
        else:
            data_np = np.array(data)
            
        # 使用最大绝对值进行缩放
        scaled_data = data_np / self.max_abs_
        
        # 转换回原始类型
        if is_torch:
            return torch.from_numpy(scaled_data)
        else:
            return scaled_data
    
    def fit_transform(self, data):
        """
        结合fit和transform操作，先拟合再转换
        
        参数
        ----------
        data : numpy.ndarray 或 torch.Tensor
            需要进行归一化的原始数据
            
        返回
        -------
        scaled_data : 与输入相同类型的归一化后的数据
        """
        return self.fit(data).transform(data)
    
    def inverse_transform(self, scaled_data):
        """
        将归一化后的数据还原回原始尺度
        
        参数
        ----------
        scaled_data : numpy.ndarray 或 torch.Tensor
            归一化后的数据
            
        返回
        -------
        original_data : 与输入相同类型的还原后的数据
        """
        # 检查是否已经拟合
        if self.max_abs_ is None:
            raise ValueError("必须先调用fit方法计算最大绝对值")
        
        # 判断输入类型
        is_torch = torch.is_tensor(scaled_data)
        
        # 转换为numpy数组进行处理
        if is_torch:
            data_np = scaled_data.detach().cpu().numpy()
        else:
            data_np = np.array(scaled_data)
            
        # 反向转换: X_original = X_scaled * max_abs
        original_data = data_np * self.max_abs_
        
        # 转换回原始类型
        if is_torch:
            return torch.from_numpy(original_data)
        else:
            return original_data


def process_data(input_file, output_file):
    """
    处理数据的主函数
    
    参数
    ----------
    input_file : str
        输入CSV文件路径
    output_file : str
        输出CSV文件路径
    """
    try:
        # 读取CSV文件
        logger.info(f"读取数据文件: {input_file}")
        df = pd.read_csv(input_file)
        data = df.to_numpy().astype(float)
        
        # 创建MaxAbsScaler并拟合数据
        logger.info("创建MaxAbsScaler并拟合数据")
        scaler = MaxAbsScaler()
        
        # 直接按列计算最大绝对值（忽略NaN值）
        logger.info("按列计算最大绝对值")
        max_abs_values = np.zeros(data.shape[1])
        for j in range(data.shape[1]):
            col_data = data[:, j]
            valid_data = col_data[~np.isnan(col_data)]
            if len(valid_data) > 0:
                max_abs_values[j] = np.max(np.abs(valid_data))
            else:
                max_abs_values[j] = 1.0  # 如果整列都是NaN，则设为1
        
        # 手动设置scaler的参数
        scaler.max_abs_ = max_abs_values
        scaler.mean_ = np.zeros_like(max_abs_values)
        scaler.std_ = max_abs_values
        scaler.original_shape = data.shape
        
        # 对整个数据集进行转换
        logger.info("对整个数据集进行归一化")
        data_for_im_np = scaler.transform(data)
        
        # 设置随机森林MICE参数
        logger.info("开始使用随机森林MICE填补缺失值...")
        # 配置随机森林回归器
        rf_regressor = RandomForestRegressor(
            n_estimators=100,  # 森林中树的数量
            max_depth=10,      # 树的最大深度
            min_samples_split=2,  # 分裂内部节点所需的最小样本数
            min_samples_leaf=1,   # 叶节点所需的最小样本数
            random_state=42       # 随机种子，确保结果可重现
        )
        
        # 创建使用随机森林作为估计器的MICE插补器
        rf_mice_imputer = IterativeImputer(
            estimator=rf_regressor,  # 使用随机森林作为估计器
            random_state=42,         # 随机种子
            max_iter=50,             # 最大迭代次数
            verbose=1                # 显示进度信息
        )
        
        imputed_data = rf_mice_imputer.fit_transform(data_for_im_np)
        logger.info("缺失值填补完成")
        
        # 将填补后的数据转换回原始尺度
        logger.info("将填补后的数据还原回原始尺度")
        df_imp = scaler.inverse_transform(imputed_data)
        df_imp = pd.DataFrame(df_imp, columns=df.columns) 
        
        # 保存为CSV文件
        logger.info(f"保存结果到: {output_file}")
        df_imp.to_csv(output_file, index=False)
        
        # 验证结果
        logger.info(f"原始数据中的缺失值数量: {np.isnan(data).sum()}")
        logger.info(f"填补后的数据中的缺失值数量: {np.isnan(df_imp.values).sum()}")
        logger.info("数据处理完成")
        
    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 设置文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "cell_performance.csv")
    output_file = os.path.join(current_dir, "imputed_data_rf_mice.csv")
    
    # 处理数据
    process_data(input_file, output_file)