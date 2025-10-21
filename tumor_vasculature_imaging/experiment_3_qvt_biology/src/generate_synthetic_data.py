#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实验三：生成模拟数据

本模块用于生成QVT特征、PD-L1表达、TIL密度和通路富集的模拟数据
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import random

class SyntheticDataGenerator:
    """合成数据生成器类"""
    
    def __init__(self, output_dir: str):
        """
        初始化数据生成器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置随机种子以保证可重复性
        np.random.seed(42)
        random.seed(42)
        
        # 数据集参数
        self.total_patients = 162  # 总患者数，与论文一致
        self.d5_patients = 100  # D5数据集（PD-L1分析）患者数
        self.d6_patients = 92   # D6数据集（通路分析）患者数
        self.til_subset_size = 31  # TIL分析子集大小
        
        # 特征参数
        self.num_qvt_features = 74  # QVT特征数量
        self.num_til_features = 12  # TIL特征数量
        self.key_pathways = ['WNT', 'FGF', 'VEGF', 'EGFR', 'NOTCH', 'PI3K', 'MAPK']  # 关键通路
        
    def generate_qvt_features(self) -> pd.DataFrame:
        """
        生成QVT特征数据
        
        Returns:
            pd.DataFrame: QVT特征数据
        """
        print("生成QVT特征模拟数据...")
        
        # 创建患者ID
        patient_ids = [f'Patient_{i:03d}' for i in range(1, self.total_patients + 1)]
        
        # 生成基础QVT特征
        feature_data = np.random.normal(loc=0, scale=1, size=(self.total_patients, self.num_qvt_features))
        
        # 创建特征名称（与论文保持一致的命名风格）
        feature_names = []
        # 角度特征（约20个）
        for i in range(1, 21):
            feature_names.append(f'angle_feature_{i}')
        # 曲率特征（约20个）
        for i in range(1, 21):
            feature_names.append(f'curvature_feature_{i}')
        # 分支特征（约15个）
        for i in range(1, 16):
            feature_names.append(f'branch_feature_{i}')
        # 统计特征（约19个）
        for i in range(1, 20):
            feature_names.append(f'statistical_feature_{i}')
        
        # 创建DataFrame
        df = pd.DataFrame(feature_data, columns=feature_names)
        df['patient_id'] = patient_ids
        df['response'] = np.random.binomial(1, 0.38, size=self.total_patients)  # 约38%响应率
        
        # 确保高QVT表型组有特定的特征分布（模拟论文发现）
        high_qvt_mask = df[feature_names[:10]].mean(axis=1) > 0
        # 在高QVT表型组中增强某些特征的表达
        df.loc[high_qvt_mask, feature_names[:5]] += 0.8
        
        # 保存数据
        output_file = os.path.join(self.output_dir, 'qvt_features.csv')
        df.to_csv(output_file, index=False)
        print(f"QVT特征数据已保存到: {output_file}")
        
        return df
    
    def generate_pd_l1_data(self) -> pd.DataFrame:
        """
        生成PD-L1表达数据
        
        Returns:
            pd.DataFrame: PD-L1表达数据
        """
        print("生成PD-L1表达模拟数据...")
        
        # 为D5数据集生成患者ID
        patient_ids = [f'Patient_{i:03d}' for i in range(1, self.d5_patients + 1)]
        
        # 生成PD-L1表达值（模拟论文发现：QVT与PD-L1表达相关）
        # 约40%的患者PD-L1高表达（>50%）
        pd_l1_expression = np.zeros(self.d5_patients)
        high_pd_l1_indices = np.random.choice(self.d5_patients, size=int(self.d5_patients * 0.4), replace=False)
        pd_l1_expression[high_pd_l1_indices] = np.random.uniform(50, 100, size=len(high_pd_l1_indices))
        low_pd_l1_indices = [i for i in range(self.d5_patients) if i not in high_pd_l1_indices]
        pd_l1_expression[low_pd_l1_indices] = np.random.uniform(0, 50, size=len(low_pd_l1_indices))
        
        # 创建DataFrame
        df = pd.DataFrame({
            'patient_id': patient_ids,
            'pd_l1_expression': pd_l1_expression,
            'pd_l1_high': (pd_l1_expression > 50).astype(int)
        })
        
        # 保存数据
        output_file = os.path.join(self.output_dir, 'pd_l1_expression.csv')
        df.to_csv(output_file, index=False)
        print(f"PD-L1表达数据已保存到: {output_file}")
        
        return df
    
    def generate_til_data(self) -> pd.DataFrame:
        """
        生成TIL密度数据
        
        Returns:
            pd.DataFrame: TIL密度数据
        """
        print("生成TIL密度模拟数据...")
        
        # 为TIL分析子集生成患者ID
        patient_ids = random.sample([f'Patient_{i:03d}' for i in range(1, self.total_patients + 1)], self.til_subset_size)
        
        # 生成TIL特征（12个特征，如论文所述）
        til_features = np.random.normal(loc=0, scale=1, size=(self.til_subset_size, self.num_til_features))
        
        # 创建特征名称
        til_feature_names = []
        for i in range(1, self.num_til_features + 1):
            til_feature_names.append(f'til_feature_{i}')
        
        # 创建DataFrame
        df = pd.DataFrame(til_features, columns=til_feature_names)
        df['patient_id'] = patient_ids
        
        # 添加一个综合TIL密度指标
        df['til_density_score'] = df[til_feature_names].mean(axis=1)
        
        # 保存数据
        output_file = os.path.join(self.output_dir, 'til_density.csv')
        df.to_csv(output_file, index=False)
        print(f"TIL密度数据已保存到: {output_file}")
        
        return df
    
    def generate_pathway_data(self) -> pd.DataFrame:
        """
        生成通路富集数据
        
        Returns:
            pd.DataFrame: 通路富集数据
        """
        print("生成通路富集模拟数据...")
        
        # 为D6数据集生成患者ID
        patient_ids = [f'Patient_{i:03d}' for i in range(1, self.d6_patients + 1)]
        
        # 创建通路富集分数矩阵
        pathway_scores = np.random.normal(loc=0, scale=1, size=(self.d6_patients, len(self.key_pathways)))
        
        # 模拟论文发现：WNT和FGF通路在高QVT表型组中上调
        # 随机选择40%的患者作为高QVT表型组
        high_qvt_indices = np.random.choice(self.d6_patients, size=int(self.d6_patients * 0.4), replace=False)
        
        # 在高QVT表型组中上调WNT和FGF通路分数
        pathway_scores[high_qvt_indices, 0] += 1.2  # WNT通路上调
        pathway_scores[high_qvt_indices, 1] += 0.8  # FGF通路上调
        
        # 创建DataFrame
        df = pd.DataFrame(pathway_scores, columns=self.key_pathways)
        df['patient_id'] = patient_ids
        df['qvt_phenotype'] = 0
        df.loc[high_qvt_indices, 'qvt_phenotype'] = 1
        
        # 保存数据
        output_file = os.path.join(self.output_dir, 'pathway_enrichment.csv')
        df.to_csv(output_file, index=False)
        print(f"通路富集数据已保存到: {output_file}")
        
        return df
    
    def generate_all_data(self, force_regenerate: bool = False):
        """
        生成所有模拟数据
        
        Args:
            force_regenerate: 是否强制重新生成数据，即使文件已存在
        
        Returns:
            dict: 包含所有生成数据的字典
        """
        # 检查所有数据文件是否已存在且不需要强制重新生成
        qvt_file = os.path.join(self.output_dir, 'qvt_features.csv')
        pd_l1_file = os.path.join(self.output_dir, 'pd_l1_expression.csv')
        til_file = os.path.join(self.output_dir, 'til_density.csv')
        pathway_file = os.path.join(self.output_dir, 'pathway_enrichment.csv')
        
        all_files_exist = all(os.path.exists(f) for f in [qvt_file, pd_l1_file, til_file, pathway_file])
        
        if all_files_exist and not force_regenerate:
            print(f"所有数据文件已存在于目录: {self.output_dir}")
            print("直接使用现有数据，跳过重新生成。")
            # 读取现有数据以返回
            qvt_data = pd.read_csv(qvt_file)
            pd_l1_data = pd.read_csv(pd_l1_file)
            til_data = pd.read_csv(til_file)
            pathway_data = pd.read_csv(pathway_file)
            
            print("\n已加载现有数据：")
            print(f"- QVT特征数据: {len(qvt_data)} 名患者，{self.num_qvt_features} 个特征")
            print(f"- PD-L1表达数据: {len(pd_l1_data)} 名患者")
            print(f"- TIL密度数据: {len(til_data)} 名患者，{self.num_til_features} 个特征")
            print(f"- 通路富集数据: {len(pathway_data)} 名患者，{len(self.key_pathways)} 个通路")
            
            return {
                'qvt_features': qvt_data,
                'pd_l1_expression': pd_l1_data,
                'til_density': til_data,
                'pathway_enrichment': pathway_data
            }
        
        print("开始生成实验三的模拟数据...")
        
        # 生成各类数据
        qvt_data = self.generate_qvt_features()
        pd_l1_data = self.generate_pd_l1_data()
        til_data = self.generate_til_data()
        pathway_data = self.generate_pathway_data()
        
        print("\n所有模拟数据生成完成！")
        print(f"- QVT特征数据: {len(qvt_data)} 名患者，{self.num_qvt_features} 个特征")
        print(f"- PD-L1表达数据: {len(pd_l1_data)} 名患者")
        print(f"- TIL密度数据: {len(til_data)} 名患者，{self.num_til_features} 个特征")
        print(f"- 通路富集数据: {len(pathway_data)} 名患者，{len(self.key_pathways)} 个通路")
        
        return {
            'qvt_features': qvt_data,
            'pd_l1_expression': pd_l1_data,
            'til_density': til_data,
            'pathway_enrichment': pathway_data
        }

def main():
    """主函数"""
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 设置输出目录为data目录
    output_dir = os.path.join(script_dir, '..', 'data')
    
    # 初始化并运行数据生成器
    generator = SyntheticDataGenerator(output_dir)
    generator.generate_all_data()

if __name__ == "__main__":
    main()