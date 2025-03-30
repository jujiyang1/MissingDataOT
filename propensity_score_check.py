#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
使用倾向性得分平衡检查比较插补方法

这个脚本实现以下功能：
1. 读取原始数据(cell_performance.csv)和两种插补方法处理后的数据(imputed_data_maxabs.csv和imputed_data_mice.csv)
2. 创建缺失值掩码矩阵，标记原始数据中的缺失值
3. 构建分类问题：
   a. 区分观测值和插补值
   b. 区分两种插补方法的结果
4. 使用机器学习模型计算AUC分数
5. 评估哪种插补方法产生的数据与原始数据分布更接近
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler

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


def create_missing_mask(data):
    """
    创建缺失值掩码矩阵
    
    参数
    ----------
    data : pandas.DataFrame
        原始数据框
        
    返回
    -------
    pandas.DataFrame
        缺失值掩码矩阵，1表示缺失，0表示非缺失
    """
    return data.isna().astype(int)


def prepare_comparison_dataset(original_df, maxabs_df, mice_df, mask):
    """
    准备用于比较两种插补方法的数据集
    
    参数
    ----------
    original_df : pandas.DataFrame
        原始数据框
    maxabs_df : pandas.DataFrame
        MaxAbs插补后的数据框
    mice_df : pandas.DataFrame
        MICE插补后的数据框
    mask : pandas.DataFrame
        缺失值掩码矩阵
        
    返回
    -------
    pandas.DataFrame
        用于比较的数据集
    """
    # 获取数值列
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 创建空的数据集
    comparison_data = []
    
    # 遍历每一行和每一列
    for i in range(len(original_df)):
        for j, col in enumerate(numeric_cols):
            # 如果原始数据中该位置是缺失的
            if mask.iloc[i, j] == 1:
                # 获取该行的其他特征（不包括当前列）
                row_features = {}
                for k, other_col in enumerate(numeric_cols):
                    if k != j:  # 不包括当前列
                        # 使用插补后的值作为特征
                        maxabs_value = maxabs_df.iloc[i, k]
                        mice_value = mice_df.iloc[i, k]
                        row_features[f'maxabs_{other_col}'] = maxabs_value
                        row_features[f'mice_{other_col}'] = mice_value
                
                # 添加两种方法的插补值
                maxabs_value = maxabs_df.iloc[i, j]
                mice_value = mice_df.iloc[i, j]
                
                # 只有当两个值都不是NaN时才添加记录
                if not (pd.isna(maxabs_value) or pd.isna(mice_value)):
                    # 创建两条记录，一条标记为MaxAbs，一条标记为MICE
                    maxabs_record = row_features.copy()
                    maxabs_record['value'] = maxabs_value
                    maxabs_record['method'] = 'MaxAbs'
                    maxabs_record['column'] = col
                    maxabs_record['row'] = i
                    
                    mice_record = row_features.copy()
                    mice_record['value'] = mice_value
                    mice_record['method'] = 'MICE'
                    mice_record['column'] = col
                    mice_record['row'] = i
                    
                    comparison_data.append(maxabs_record)
                    comparison_data.append(mice_record)
    
    return pd.DataFrame(comparison_data)


def prepare_observed_vs_imputed_dataset(original_df, imputed_df, mask, method_name):
    """
    准备用于比较观测值和插补值的数据集
    
    参数
    ----------
    original_df : pandas.DataFrame
        原始数据框
    imputed_df : pandas.DataFrame
        插补后的数据框
    mask : pandas.DataFrame
        缺失值掩码矩阵
    method_name : str
        插补方法名称
        
    返回
    -------
    pandas.DataFrame
        用于比较的数据集
    """
    # 获取数值列
    numeric_cols = original_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 创建空的数据集
    comparison_data = []
    
    # 遍历每一行和每一列
    for i in range(len(original_df)):
        for j, col in enumerate(numeric_cols):
            # 获取该行的其他特征（不包括当前列）
            row_features = {}
            has_missing = False
            
            for k, other_col in enumerate(numeric_cols):
                if k != j:  # 不包括当前列
                    # 使用插补后的值作为特征
                    value = imputed_df.iloc[i, k]
                    if pd.isna(value):
                        has_missing = True
                        break
                    row_features[f'{other_col}'] = value
            
            if has_missing:
                continue
            
            # 添加当前值和标签
            if mask.iloc[i, j] == 1:  # 如果原始数据中该位置是缺失的
                # 使用插补值
                value = imputed_df.iloc[i, j]
                if pd.isna(value):
                    continue
                is_imputed = 1  # 标记为插补值
            else:  # 如果原始数据中该位置不是缺失的
                # 使用原始值
                value = original_df.iloc[i, j]
                if pd.isna(value):
                    continue
                is_imputed = 0  # 标记为观测值
            
            # 创建记录
            record = row_features.copy()
            record['value'] = value
            record['is_imputed'] = is_imputed
            record['column'] = col
            record['row'] = i
            
            comparison_data.append(record)
    
    df = pd.DataFrame(comparison_data)
    df['method'] = method_name
    return df


def train_and_evaluate_classifier(data, target_col, feature_cols, test_size=0.3, random_state=42):
    """
    训练分类器并评估性能
    
    参数
    ----------
    data : pandas.DataFrame
        数据集
    target_col : str
        目标变量列名
    feature_cols : list
        特征列名列表
    test_size : float, 可选
        测试集比例，默认为0.3
    random_state : int, 可选
        随机种子，默认为42
        
    返回
    -------
    dict
        包含模型、AUC分数和预测概率的字典
    """
    # 准备特征和目标变量
    X = data[feature_cols].copy()
    y = data[target_col].copy()
    
    # 处理缺失值
    X = X.fillna(X.mean())
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
    
    # 训练随机森林分类器
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train, y_train)
    
    # 预测概率
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    # 计算AUC分数
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    return {
        'model': clf,
        'auc_score': auc_score,
        'y_test': y_test,
        'y_pred_proba': y_pred_proba
    }


def plot_roc_curve(results_dict, title):
    """
    绘制ROC曲线
    
    参数
    ----------
    results_dict : dict
        包含模型评估结果的字典
    title : str
        图表标题
    """
    plt.figure(figsize=(10, 8))
    
    for method, result in results_dict.items():
        fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
        roc_auc = result['auc_score']
        plt.plot(fpr, tpr, lw=2, label=f'{method} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f"{title.replace(' ', '_')}.png")
    print(f"ROC曲线已保存到 '{title.replace(' ', '_')}.png'")


def compare_imputation_methods_with_propensity_score(original_file, maxabs_file, mice_file):
    """
    使用倾向性得分平衡检查比较插补方法
    
    参数
    ----------
    original_file : str
        原始数据文件路径
    maxabs_file : str
        MaxAbs处理后的数据文件路径
    mice_file : str
        MICE处理后的数据文件路径
    """
    try:
        # 加载数据
        original_df = load_data(original_file)
        maxabs_df = load_data(maxabs_file)
        mice_df = load_data(mice_file)
        
        if original_df is None or maxabs_df is None or mice_df is None:
            print("无法加载所有必要的数据文件")
            return
        
        # 创建缺失值掩码
        print("\n创建缺失值掩码矩阵")
        mask = create_missing_mask(original_df)
        
        # 获取数值列
        numeric_cols = original_df.select_dtypes(include=[np.number]).columns.tolist()
        mask = mask[numeric_cols]
        
        # 准备比较两种插补方法的数据集
        print("\n准备比较两种插补方法的数据集")
        method_comparison_df = prepare_comparison_dataset(original_df, maxabs_df, mice_df, mask)
        
        # 准备比较观测值和插补值的数据集
        print("\n准备比较观测值和插补值的数据集")
        maxabs_obs_imp_df = prepare_observed_vs_imputed_dataset(original_df, maxabs_df, mask, 'MaxAbs')
        mice_obs_imp_df = prepare_observed_vs_imputed_dataset(original_df, mice_df, mask, 'MICE')
        
        # 合并观测值和插补值的数据集
        obs_imp_df = pd.concat([maxabs_obs_imp_df, mice_obs_imp_df])
        
        # 训练和评估分类器 - 比较两种插补方法
        print("\n训练和评估分类器 - 比较两种插补方法")
        # 排除非特征列
        feature_cols = [col for col in method_comparison_df.columns if col not in ['method', 'value', 'column', 'row']]
        method_results = train_and_evaluate_classifier(method_comparison_df, 'method', feature_cols)
        
        # 训练和评估分类器 - 比较观测值和插补值
        print("\n训练和评估分类器 - 比较观测值和插补值")
        # 分别对每种方法进行评估
        obs_imp_results = {}
        for method in ['MaxAbs', 'MICE']:
            method_df = obs_imp_df[obs_imp_df['method'] == method]
            # 排除非特征列
            feature_cols = [col for col in method_df.columns if col not in ['is_imputed', 'value', 'column', 'row', 'method']]
            obs_imp_results[method] = train_and_evaluate_classifier(method_df, 'is_imputed', feature_cols)
        
        # 输出结果
        print("\n===== 倾向性得分平衡检查结果 =====")
        print("\n1. 区分两种插补方法的AUC分数")
        method_auc = method_results['auc_score']
        print(f"AUC分数: {method_auc:.4f}")
        if method_auc > 0.6:
            print("结论: 两种插补方法产生的数据有明显差异")
        else:
            print("结论: 两种插补方法产生的数据差异不大")
        
        print("\n2. 区分观测值和插补值的AUC分数")
        for method, result in obs_imp_results.items():
            print(f"{method}方法 - AUC分数: {result['auc_score']:.4f}")
        
        # 比较哪种方法更好
        maxabs_auc = obs_imp_results['MaxAbs']['auc_score']
        mice_auc = obs_imp_results['MICE']['auc_score']
        
        print("\n3. 插补方法评估")
        if abs(maxabs_auc - 0.5) < abs(mice_auc - 0.5):
            conclusion = "MaxAbs插补方法产生的数据与原始数据分布更接近"
            winner = "MaxAbs"
        elif abs(mice_auc - 0.5) < abs(maxabs_auc - 0.5):
            conclusion = "MICE插补方法产生的数据与原始数据分布更接近"
            winner = "MICE"
        else:
            conclusion = "两种方法在保持数据分布方面表现相当"
            winner = "平局"
        
        print(f"结论: {conclusion}")
        
        # 创建详细比较表格
        comparison_table = [{
            '评估指标': '区分两种插补方法的AUC分数',
            'AUC分数': method_auc,
            '解释': '越接近0.5表示两种方法越相似'
        }, {
            '评估指标': 'MaxAbs方法区分观测值和插补值的AUC分数',
            'AUC分数': maxabs_auc,
            '解释': '越接近0.5表示插补值与观测值越相似'
        }, {
            '评估指标': 'MICE方法区分观测值和插补值的AUC分数',
            'AUC分数': mice_auc,
            '解释': '越接近0.5表示插补值与观测值越相似'
        }, {
            '评估指标': '总体评估',
            'AUC分数': None,
            '解释': conclusion
        }]
        
        # 创建比较表格DataFrame并保存
        comparison_df = pd.DataFrame(comparison_table)
        comparison_df.to_csv('propensity_score_comparison_results.csv', index=False, encoding='utf-8-sig')
        print("\n详细比较结果已保存到 'propensity_score_comparison_results.csv'")
        
        # 绘制ROC曲线
        print("\n绘制ROC曲线...")
        plot_roc_curve(obs_imp_results, "观测值与插补值区分的ROC曲线")
        
        # 返回结果字典
        return {
            'method_auc': method_auc,
            'maxabs_auc': maxabs_auc,
            'mice_auc': mice_auc,
            'winner': winner
        }
        
    except Exception as e:
        print(f"比较过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # 设置文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    original_file = os.path.join(current_dir, "cell_performance.csv")
    maxabs_file = os.path.join(current_dir, "imputed_data_maxabs.csv")
    mice_file = os.path.join(current_dir, "imputed_data_mice.csv")
    
    # 使用倾向性得分平衡检查比较插补方法
    print("使用倾向性得分平衡检查比较插补方法...\n")
    compare_imputation_methods_with_propensity_score(original_file, maxabs_file, mice_file)