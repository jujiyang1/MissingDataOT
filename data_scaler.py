#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from sklearn.preprocessing import scale

class DataScaler:
    """
    数据缩放处理类，提供数据归一化和反归一化功能。
    
    主要功能：
    1. 对原始数据进行scale归一化处理
    2. 将归一化后的数据还原回原始尺度
    
    这个类作为数据预处理和后处理的桥梁，与其他文件中实现的数据填补算法协同工作。
    """
    
    def __init__(self):
        """
        初始化DataScaler类，设置内部状态变量
        """
        self.mean_ = None  # 存储原始数据的均值
        self.std_ = None   # 存储原始数据的标准差
        self.original_shape = None  # 存储原始数据的形状
        
    def fit(self, data):
        """
        计算并存储数据的均值和标准差，用于后续的缩放操作
        
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
            
        # 计算均值和标准差
        self.mean_ = np.mean(data_np, axis=0)
        self.std_ = np.std(data_np, axis=0, ddof=1)  # 使用无偏估计 (N-1)
        
        # 处理标准差为0的情况
        self.std_[self.std_ == 0] = 1.0
        
        return self
    
    def transform(self, data):
        """
        使用sklearn的scale函数对数据进行归一化处理
        
        参数
        ----------
        data : numpy.ndarray 或 torch.Tensor
            需要进行归一化的原始数据
            
        返回
        -------
        scaled_data : 与输入相同类型的归一化后的数据
        """
        # 检查是否已经拟合
        if self.mean_ is None or self.std_ is None:
            raise ValueError("必须先调用fit方法计算均值和标准差")
        
        # 判断输入类型
        is_torch = torch.is_tensor(data)
        
        # 转换为numpy数组进行处理
        if is_torch:
            data_np = data.detach().cpu().numpy()
        else:
            data_np = np.array(data)
            
        # 使用sklearn的scale函数进行归一化
        scaled_data = scale(data_np)
        
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
        if self.mean_ is None or self.std_ is None:
            raise ValueError("必须先调用fit方法计算均值和标准差")
        
        # 判断输入类型
        is_torch = torch.is_tensor(scaled_data)
        
        # 转换为numpy数组进行处理
        if is_torch:
            data_np = scaled_data.detach().cpu().numpy()
        else:
            data_np = np.array(scaled_data)
            
        # 反向转换: X_original = X_scaled * std + mean
        original_data = data_np * self.std_ + self.mean_
        
        # 转换回原始类型
        if is_torch:
            return torch.from_numpy(original_data)
        else:
            return original_data

# 使用示例
if __name__ == "__main__":
    # 示例数据
    data = np.random.randn(100, 5)
    
    # 创建缩放器实例
    scaler = DataScaler()
    
    # 归一化数据
    scaled_data = scaler.fit_transform(data)
    print("归一化后数据均值:", np.mean(scaled_data, axis=0))
    print("归一化后数据标准差:", np.std(scaled_data, axis=0))
    from sklearn.metrics import mean_squared_error
    # 计算还原后数据与原始数据的均方误差
    #mse = mean_squared_error(data, scaler.inverse_transform(scaled_data))
    
    # 还原数据
    restored_data = scaler.inverse_transform(scaled_data)
    mse = mean_squared_error(data, scaler.inverse_transform(scaled_data))
    print("MSE:", mse)
    print("还原后数据与原始数据的最大差异:", np.max(np.abs(restored_data - data)))
   