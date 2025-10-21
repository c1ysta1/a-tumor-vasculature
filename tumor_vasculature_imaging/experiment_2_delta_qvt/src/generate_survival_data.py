#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成模拟生存分析数据的脚本

本脚本用于生成用于生存分析的数据文件，包括delta QVT特征和响应标签。
这些数据将被survival_analysis.py脚本使用来生成生存曲线和热图。
"""

import os
import numpy as np
import pandas as pd
import argparse

def generate_mock_delta_features(num_patients=100, num_features=20):
    """
    生成模拟的delta QVT特征数据
    
    Args:
        num_patients: 患者数量
        num_features: 特征数量
    
    Returns:
        delta_features: delta QVT特征数据框
    """
    # 设置随机种子以确保结果可重现
    np.random.seed(42)
    
    # 生成随机特征数据
    data = np.random.randn(num_patients, num_features)
    
    # 创建特征名称
    feature_names = []
    # 生成角度相关特征
    for i in range(int(num_features * 0.4)):
        feature_names.append(f'delta_angle_acute_{i+1}')
    # 生成血管密度相关特征
    for i in range(int(num_features * 0.3)):
        feature_names.append(f'delta_vessel_density_{i+1}')
    # 生成血管长度相关特征
    for i in range(int(num_features * 0.2)):
        feature_names.append(f'delta_vessel_length_{i+1}')
    # 生成其他特征
    for i in range(num_features - len(feature_names)):
        feature_names.append(f'delta_feature_{i+1}')
    
    # 创建数据框
    delta_features = pd.DataFrame(data, columns=feature_names)
    
    # 添加患者ID作为索引
    delta_features.index = [f'patient_{i+1}' for i in range(num_patients)]
    
    return delta_features

def generate_mock_response_labels(num_patients=100, positive_rate=0.4):
    """
    生成模拟的响应标签数据
    
    Args:
        num_patients: 患者数量
        positive_rate: 阳性（响应）患者的比例
    
    Returns:
        response_labels: 响应标签序列
    """
    # 设置随机种子以确保结果可重现
    np.random.seed(43)
    
    # 生成响应标签（0表示非响应，1表示响应）
    num_positive = int(num_patients * positive_rate)
    num_negative = num_patients - num_positive
    
    labels = np.concatenate([np.ones(num_positive), np.zeros(num_negative)])
    np.random.shuffle(labels)
    
    # 创建标签序列
    response_labels = pd.Series(labels, dtype=int)
    response_labels.index = [f'patient_{i+1}' for i in range(num_patients)]
    response_labels.name = 'response'
    
    return response_labels

def generate_dataset(data_dir, dataset_name, num_patients=50, num_features=20, positive_rate=0.4):
    """
    生成特定数据集的模拟数据
    
    Args:
        data_dir: 数据保存目录
        dataset_name: 数据集名称
        num_patients: 患者数量
        num_features: 特征数量
        positive_rate: 阳性（响应）患者的比例
    """
    # 生成delta QVT特征
    delta_features = generate_mock_delta_features(num_patients, num_features)
    
    # 生成响应标签
    response_labels = generate_mock_response_labels(num_patients, positive_rate)
    
    # 确保患者ID一致
    delta_features = delta_features.loc[response_labels.index]
    
    # 保存数据
    dataset_dir = os.path.join(data_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    delta_features.to_csv(os.path.join(dataset_dir, 'delta_angle_features.csv'))
    response_labels.to_csv(os.path.join(dataset_dir, 'response_labels.csv'))
    
    print(f"已生成 {dataset_name} 数据集，包含 {num_patients} 个患者和 {num_features} 个特征")
    print(f"阳性患者比例: {positive_rate}")
    print(f"数据保存至: {dataset_dir}")
    
    return delta_features, response_labels

def generate_all_datasets(data_dir, results_dir):
    """
    生成所有需要的模拟数据集
    
    Args:
        data_dir: 数据保存目录
        results_dir: 结果保存目录
    """
    # 确保目录存在
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 定义数据集配置
    datasets = [
        {'name': 'D1', 'num_patients': 40, 'num_features': 15, 'positive_rate': 0.4},
        {'name': 'D2', 'num_patients': 25, 'num_features': 18, 'positive_rate': 0.45},
        {'name': 'D3', 'num_patients': 18, 'num_features': 12, 'positive_rate': 0.35},
        {'name': 'D4', 'num_patients': 23, 'num_features': 16, 'positive_rate': 0.42},
    ]
    
    # 生成每个数据集
    all_delta_features = []
    all_response_labels = []
    
    for dataset in datasets:
        delta_features, response_labels = generate_dataset(
            data_dir, 
            dataset['name'], 
            dataset['num_patients'], 
            dataset['num_features'], 
            dataset['positive_rate']
        )
        
        all_delta_features.append(delta_features)
        all_response_labels.append(response_labels)
    
    # 合并所有数据集
    combined_delta_features = pd.concat(all_delta_features)
    combined_response_labels = pd.concat(all_response_labels)
    
    # 保存合并后的数据集
    combined_delta_features.to_csv(os.path.join(results_dir, 'all_delta_angle_features.csv'))
    combined_response_labels.to_csv(os.path.join(results_dir, 'all_response_labels.csv'))
    
    print(f"\n已生成合并数据集，包含 {len(combined_delta_features)} 个患者")
    print(f"合并数据保存至: {results_dir}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成模拟生存分析数据')
    parser.add_argument('--data-dir', type=str, 
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'),
                        help='数据目录路径')
    parser.add_argument('--results-dir', type=str, 
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'),
                        help='结果目录路径')
    
    args = parser.parse_args()
    
    # 生成所有数据集
    generate_all_datasets(args.data_dir, args.results_dir)

if __name__ == "__main__":
    main()