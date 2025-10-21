import numpy as np
import pandas as pd
import os
from typing import Dict, List, Tuple

class DeltaQVTCalculator:
    """
    计算治疗前后QVT特征的绝对变化量（delta QVT）
    根据论文：基线与治疗后QVT特征的绝对变化
    支持单个数据集和多个数据集的处理
    """
    
    def __init__(self):
        # 论文中提到的关键QVT特征类别
        self.angle_feature_prefixes = ['angle_', 'acute_', 'obtuse_']
        self.curvature_feature_prefixes = ['curvature_', 'tortuosity_']
        self.statistical_feature_prefixes = ['mean_', 'std_', 'min_', 'max_']
    
    def calculate_delta_qvt_single(self, baseline_features: pd.DataFrame, post_treatment_features: pd.DataFrame) -> pd.DataFrame:
        """
        计算单个数据集的基线和治疗后QVT特征的绝对变化量
        
        Args:
            baseline_features: 基线QVT特征
            post_treatment_features: 治疗后QVT特征
        
        Returns:
            delta_qvt_features: 特征绝对变化量
        """
        # 确保两个数据框具有相同的索引（患者ID）和列（特征）
        common_indices = baseline_features.index.intersection(post_treatment_features.index)
        common_columns = baseline_features.columns.intersection(post_treatment_features.columns)
        
        # 计算绝对变化量 |治疗后 - 基线|
        delta_qvt = np.abs(
            post_treatment_features.loc[common_indices, common_columns] - 
            baseline_features.loc[common_indices, common_columns]
        )
        
        # 重命名列，添加delta_前缀
        delta_qvt.columns = [f'delta_{col}' for col in delta_qvt.columns]
        
        return delta_qvt
    
    def calculate_delta_qvt_multiple(self, baseline_datasets: Dict[str, pd.DataFrame], 
                                   post_treatment_datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        计算多个数据集的delta QVT特征
        
        Args:
            baseline_datasets: 各数据集的基线数据字典
            post_treatment_datasets: 各数据集的治疗后数据字典
            
        Returns:
            delta_datasets: 各数据集的delta QVT特征字典
        """
        delta_datasets = {}
        
        for dataset_name in baseline_datasets.keys():
            if dataset_name in post_treatment_datasets:
                delta_datasets[dataset_name] = self.calculate_delta_qvt_single(
                    baseline_datasets[dataset_name], 
                    post_treatment_datasets[dataset_name]
                )
                print(f"已计算数据集 {dataset_name} 的delta QVT特征")
            else:
                print(f"警告: 数据集 {dataset_name} 在治疗后数据中不存在")
        
        return delta_datasets
    
    # 保留旧接口以保持向后兼容性
    def calculate_delta_qvt(self, baseline_features, post_treatment_features):
        """兼容旧接口的计算方法"""
        if isinstance(baseline_features, dict) and isinstance(post_treatment_features, dict):
            return self.calculate_delta_qvt_multiple(baseline_features, post_treatment_features)
        else:
            return self.calculate_delta_qvt_single(baseline_features, post_treatment_features)
    
    def extract_angle_features(self, delta_features: pd.DataFrame) -> pd.DataFrame:
        """
        提取与角度相关的delta QVT特征
        根据论文：响应者治疗后血管锐角数量显著减少
        """
        angle_features = []
        for col in delta_features.columns:
            if any(prefix in col for prefix in self.angle_feature_prefixes):
                angle_features.append(col)
        
        return delta_features[angle_features]
    
    def load_features(self, file_path: str) -> pd.DataFrame:
        """
        从CSV文件加载QVT特征
        
        Args:
            file_path: 特征文件路径
        
        Returns:
            features: 加载的特征数据框
        """
        return pd.read_csv(file_path, index_col=0)
    
    def save_delta_features_single(self, delta_features: pd.DataFrame, output_path: str) -> None:
        """
        保存单个数据集计算的delta QVT特征
        
        Args:
            delta_features: 计算得到的delta QVT特征
            output_path: 输出文件路径
        """
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        delta_features.to_csv(output_path)
        print(f"Delta QVT特征已保存至: {output_path}")
        print(f"共 {len(delta_features)} 患者, {len(delta_features.columns)} 个delta特征")
    
    def save_delta_features_multiple(self, delta_datasets: Dict[str, pd.DataFrame], output_dir: str) -> None:
        """
        保存多个数据集的delta QVT特征
        
        Args:
            delta_datasets: 各数据集的delta QVT特征字典
            output_dir: 输出目录
        """
        # 创建主输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 为每个数据集保存delta特征
        for dataset_name, delta_data in delta_datasets.items():
            dataset_path = os.path.join(output_dir, f'{dataset_name}_delta_qvt_features.csv')
            delta_data.to_csv(dataset_path)
            print(f"数据集 {dataset_name} 的Delta QVT特征已保存至: {dataset_path}")
        
        # 同时保存所有数据的合并结果
        all_delta = pd.concat(delta_datasets.values())
        all_delta.to_csv(os.path.join(output_dir, 'all_delta_qvt_features.csv'))
        print(f"\n合并后的Delta QVT特征已保存至 {os.path.join(output_dir, 'all_delta_qvt_features.csv')}")
        print(f"总患者数: {len(all_delta)} 患者, {len(all_delta.columns)} 个delta特征")
    
    # 保留旧接口以保持向后兼容性
    def save_delta_features(self, delta_features, output_path):
        """兼容旧接口的保存方法"""
        if isinstance(delta_features, dict):
            # 如果是字典，output_path被视为目录
            self.save_delta_features_multiple(delta_features, output_path)
        else:
            # 如果不是字典，保持原有行为
            self.save_delta_features_single(delta_features, output_path)

if __name__ == "__main__":
    # 示例用法
    calculator = DeltaQVTCalculator()
    print("Delta QVT特征计算器已初始化")
    print("请使用以下命令行参数运行:")
    print("python delta_qvt_calculator.py --baseline <baseline_file.csv> --post <post_treatment_file.csv> --output <output_file.csv>")