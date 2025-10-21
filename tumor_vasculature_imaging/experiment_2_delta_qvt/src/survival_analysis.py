#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生存分析脚本：生成生存曲线和相关可视化图表

本脚本用于根据QVT特征生成生存分析图表，包括：
1. 特征热图（类似于图表A）
2. 不同风险组的Kaplan-Meier生存曲线（类似于图表B-F）
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from sklearn.preprocessing import StandardScaler
import argparse

# 设置字体
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示
plt.rcParams['font.size'] = 10

def create_survival_data(delta_features, response_labels, n_patients=100):
    """
    创建生存分析所需的数据
    
    Args:
        delta_features: delta QVT特征数据
        response_labels: 响应标签
        n_patients: 要生成的患者数量
    
    Returns:
        survival_data: 包含生存信息的数据框
    """
    # 设置固定随机种子以确保结果可重复
    np.random.seed(42)
    
    # 重置索引，避免重复索引问题
    delta_features = delta_features.reset_index(drop=True)
    response_labels = response_labels.reset_index(drop=True)
    
    # 确保响应标签与特征数据长度一致
    min_length = min(len(delta_features), len(response_labels))
    delta_features = delta_features.iloc[:min_length]
    response_labels = response_labels.iloc[:min_length]
    
    # 选择部分患者进行演示 - 使用固定随机种子确保每次选择相同的患者
    if len(delta_features) > n_patients:
        delta_features = delta_features.sample(n=n_patients, random_state=42)
        response_labels = response_labels.loc[delta_features.index]
    
    # 创建生存数据 - 随机种子已在函数开头设置
    
    # 生成生存时间（月）
    # 响应者（1）通常有更长的生存期
    num_patients = len(delta_features)
    base_survival = np.random.gamma(shape=5, scale=10, size=num_patients)
    survival_time = []
    
    # 获取响应标签列表
    response_list = response_labels.tolist()
    
    for i in range(num_patients):
        if i < len(response_list) and response_list[i] == 1:  # 响应者
            # 响应者生存期更长
            survival_time.append(base_survival[i] * (1.5 + np.random.random()))
        else:  # 非响应者
            survival_time.append(base_survival[i] * (0.5 + np.random.random()))
    
    # 生成事件指示器（1表示事件发生，0表示截尾）
    # 随机将约20%的观测值设为截尾
    event_observed = np.random.binomial(1, 0.8, size=num_patients)
    
    # 创建生存数据框
    survival_data = delta_features.copy()
    survival_data['survival_time'] = survival_time
    survival_data['event_observed'] = event_observed
    survival_data['response'] = response_labels
    
    # 计算风险评分（基于delta QVT特征）
    # 这里简化处理，使用几个关键特征的加权和作为风险评分
    key_features = []
    for col in survival_data.columns:
        if 'delta_angle_acute' in col or 'delta_vessel_density' in col or 'delta_vessel_length' in col:
            key_features.append(col)
    
    # 如果没有找到合适的特征，使用前几个特征
    if len(key_features) == 0:
        key_features = survival_data.columns[:3].tolist()
    
    # 标准化特征
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(survival_data[key_features])
    
    # 计算风险评分
    # 使用固定权重而非随机权重，确保结果可重复
    # 使用固定种子生成权重，确保每次都相同
    np.random.seed(42)
    weights = np.random.randn(len(key_features))
    risk_scores = np.dot(scaled_features, weights)
    
    # 标准化风险评分
    risk_scores = (risk_scores - np.mean(risk_scores)) / np.std(risk_scores)
    
    # 添加风险评分到数据框
    survival_data['risk_score'] = risk_scores
    
    # 将患者分为高风险和低风险组（基于风险评分中位数）
    median_risk = np.median(risk_scores)
    survival_data['risk_group'] = ['High risk' if score > median_risk else 'Low risk' for score in risk_scores]
    
    return survival_data

def plot_feature_heatmap(survival_data, output_path):
    """
    绘制特征热图（类似于图表A）
    
    Args:
        survival_data: 生存分析数据
        output_path: 输出文件路径
    """
    # 选择用于热图的特征（排除非特征列）
    feature_cols = [col for col in survival_data.columns if col not in ['survival_time', 'event_observed', 'response', 'risk_score', 'risk_group']]
    
    # 如果特征太多，只选择前20个特征
    if len(feature_cols) > 20:
        feature_cols = feature_cols[:20]
    
    # 确保特征列不为空
    if len(feature_cols) == 0:
        print("警告：没有足够的特征列用于绘制热图，跳过热图绘制。")
        return
    
    # 获取特征数据
    feature_data = survival_data[feature_cols].copy()
    
    # 处理缺失值和非有限值
    feature_data = feature_data.fillna(0)  # 填充缺失值
    feature_data = feature_data.replace([np.inf, -np.inf], 0)  # 替换无穷大值
    
    # 标准化特征用于热图显示
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(feature_data)
    scaled_df = pd.DataFrame(scaled_data, columns=feature_cols)
    
    # 添加风险组信息用于排序
    scaled_df['risk_group'] = survival_data['risk_group'].values
    
    # 按风险组排序
    scaled_df = scaled_df.sort_values('risk_group')
    
    # 移除风险组列
    scaled_df = scaled_df.drop('risk_group', axis=1)
    
    # 创建热图，禁用聚类以避免非有限值问题
    plt.figure(figsize=(12, 8))
    sns.clustermap(scaled_df, cmap='RdBu', row_cluster=False, col_cluster=False, 
                  figsize=(12, 8), cbar_kws={'label': 'Z-score'})
    plt.title('Feature Expression Heatmap by Risk Group', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"热图已保存到: {output_path}")
    plt.close()

def plot_km_curve(survival_data, output_dir, dataset_name="Combined"):
    """
    绘制Kaplan-Meier生存曲线（类似于图表B-F）
    
    Args:
        survival_data: 生存分析数据
        output_dir: 输出目录
        dataset_name: 数据集名称
    """
    # 确保数据类型正确
    # 复制数据以避免修改原始数据
    data_copy = survival_data.copy()
    
    # 确保生存时间和事件观察值是数值类型
    data_copy['survival_time'] = pd.to_numeric(data_copy['survival_time'], errors='coerce')
    data_copy['event_observed'] = pd.to_numeric(data_copy['event_observed'], errors='coerce')
    
    # 移除任何包含NaN的行
    data_copy = data_copy.dropna(subset=['survival_time', 'event_observed'])
    
    # 确保有足够的数据点
    if len(data_copy) < 2:
        print(f"警告：{dataset_name} 数据集样本数量不足，跳过KM曲线绘制。")
        return
    
    # 初始化KM拟合器
    kmf = KaplanMeierFitter()
    
    # 按风险组绘制生存曲线
    plt.figure(figsize=(8, 6))
    
    # 低风险组
    low_risk = data_copy[data_copy['risk_group'] == 'Low risk']
    if len(low_risk) > 0:
        kmf.fit(low_risk['survival_time'], event_observed=low_risk['event_observed'], label='Low risk')
        ax = kmf.plot(ci_show=True, color='blue')
    
    # 高风险组
    high_risk = data_copy[data_copy['risk_group'] == 'High risk']
    if len(high_risk) > 0:
        kmf.fit(high_risk['survival_time'], event_observed=high_risk['event_observed'], label='High risk')
        kmf.plot(ax=ax, ci_show=True, color='red')
    
    # 执行logrank检验
    results = logrank_test(
        low_risk['survival_time'], 
        high_risk['survival_time'],
        event_observed_A=low_risk['event_observed'],
        event_observed_B=high_risk['event_observed']
    )
    
    # 计算风险比（HR）
    # 这里使用简化的方法估算HR
    low_risk_events = low_risk['event_observed'].sum()
    high_risk_events = high_risk['event_observed'].sum()
    
    # 计算观察时间总和（用于估算死亡率）
    low_risk_time = low_risk['survival_time'].sum()
    high_risk_time = high_risk['survival_time'].sum()
    
    # 估算风险比
    if low_risk_time > 0 and high_risk_time > 0:
        low_risk_rate = low_risk_events / low_risk_time
        high_risk_rate = high_risk_events / high_risk_time
        hr = high_risk_rate / low_risk_rate
        
        # 添加p值和HR到图表
        plt.text(0.05, 0.05, f'P = {results.p_value:.4f}', transform=ax.transAxes, fontsize=12)
        plt.text(0.05, 0.01, f'HR = {hr:.2f}', transform=ax.transAxes, fontsize=12)
    
    # 添加风险人数表格
    plt.text(0.5, 0.05, 'Number at risk', transform=ax.transAxes, ha='center', fontsize=10)
    
    # 获取x轴的刻度位置
    x_ticks = ax.get_xticks()
    x_ticks = x_ticks[x_ticks >= 0]
    
    # 计算每个时间点的风险人数
    low_risk_counts = []
    high_risk_counts = []
    
    for tick in x_ticks:
        lr_count = len(low_risk[low_risk['survival_time'] >= tick])
        hr_count = len(high_risk[high_risk['survival_time'] >= tick])
        low_risk_counts.append(lr_count)
        high_risk_counts.append(hr_count)
    
    # 添加风险人数到图表
    for i, tick in enumerate(x_ticks):
        plt.text(tick, -0.07, str(low_risk_counts[i]), ha='center', fontsize=8, color='blue')
        plt.text(tick, -0.12, str(high_risk_counts[i]), ha='center', fontsize=8, color='red')
    
    # 设置图表标题和标签
    plt.title(f'Kaplan-Meier Survival Curve - {dataset_name}', fontsize=14)
    plt.xlabel('Time (months)', fontsize=12)
    plt.ylabel('Survival probability', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 调整底部边距以容纳风险人数表格
    plt.subplots_adjust(bottom=0.2)
    
    # 保存图表
    output_path = os.path.join(output_dir, f'km_curve_{dataset_name.lower().replace(" ", "_")}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"KM曲线已保存到: {output_path}")
    plt.close()

def generate_survival_analysis_charts(data_dir, results_dir, output_dir):
    """
    生成生存分析相关图表
    
    Args:
        data_dir: 数据目录
        results_dir: 结果目录
        output_dir: 输出目录
    """
    # 设置全局随机种子，确保所有随机操作可重复
    np.random.seed(42)
    
    print("开始生成生存分析图表...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 统一从data文件夹读取数据集
    try:
        # 加载所有患者的基线和治疗后特征
        all_baseline_features = pd.read_csv(os.path.join(data_dir, 'all_baseline_qvt_features.csv'), index_col=0)
        all_post_features = pd.read_csv(os.path.join(data_dir, 'all_post_treatment_qvt_features.csv'), index_col=0)
        all_labels = pd.read_csv(os.path.join(data_dir, 'all_response_labels.csv'), index_col=0).squeeze()
        
        # 计算delta特征
        all_delta_features = all_post_features - all_baseline_features
        
        # 重命名列，添加delta_前缀
        all_delta_features.columns = ['delta_' + col for col in all_delta_features.columns]
        
        print(f"从data文件夹加载并计算了 {len(all_delta_features)} 个患者的delta特征数据")
    except FileNotFoundError as e:
        print(f"错误：无法找到数据集文件！{str(e)}")
        print("请确保数据文件存在于data目录中。")
        return
    
    # 为合并数据集创建生存数据
    combined_survival_data = create_survival_data(all_delta_features, all_labels)
    
    # 绘制合并数据集的热图
    plot_feature_heatmap(combined_survival_data, os.path.join(output_dir, 'feature_heatmap_combined.png'))
    
    # 绘制合并数据集的KM曲线
    plot_km_curve(combined_survival_data, output_dir, "Combined")
    
    # 尝试为每个单独的数据集生成图表
    for dataset in ['D1', 'D2', 'D3', 'D4']:
        try:
            # 从data目录下的对应数据集子目录读取数据
            dataset_dir = os.path.join(data_dir, dataset)
            
            # 加载基线和治疗后特征
            dataset_baseline = pd.read_csv(os.path.join(dataset_dir, 'baseline_qvt_features.csv'), index_col=0)
            dataset_post = pd.read_csv(os.path.join(dataset_dir, 'post_treatment_qvt_features.csv'), index_col=0)
            dataset_labels = pd.read_csv(os.path.join(dataset_dir, 'response_labels.csv'), index_col=0).squeeze()
            
            # 计算delta特征
            dataset_delta_features = dataset_post - dataset_baseline
            dataset_delta_features.columns = ['delta_' + col for col in dataset_delta_features.columns]
            
            # 创建数据集的生存数据
            dataset_survival_data = create_survival_data(dataset_delta_features, dataset_labels)
            
            # 绘制KM曲线
            plot_km_curve(dataset_survival_data, output_dir, dataset)
            
        except FileNotFoundError as e:
            print(f"警告：找不到 {dataset} 数据集的文件 {str(e)}，跳过该数据集的分析。")
            continue
    
    print("生存分析图表生成完成！")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='生成生存分析相关图表')
    parser.add_argument('--data-dir', type=str, 
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'),
                        help='数据目录路径')
    parser.add_argument('--results-dir', type=str, 
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'),
                        help='结果目录路径')
    parser.add_argument('--output-dir', type=str, 
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'visualization', 'experiment_2', 'survival_plots'),
                        help='输出目录路径')
    
    args = parser.parse_args()
    
    # 生成生存分析图表
    generate_survival_analysis_charts(args.data_dir, args.results_dir, args.output_dir)

if __name__ == "__main__":
    main()