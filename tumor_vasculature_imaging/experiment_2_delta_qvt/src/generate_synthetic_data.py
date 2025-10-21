import numpy as np
import pandas as pd
import os
from typing import Tuple, Dict, List

class SyntheticDataGenerator:
    """
    生成模拟的肿瘤血管数据，包括基线和治疗后数据
    根据论文：响应者治疗后血管锐角数量显著减少
    """
    
    def __init__(self, dataset_config: Dict[str, int] = None):
        # 如果没有提供配置，使用默认配置模拟论文中的数据集分布
        if dataset_config is None:
            # 模拟论文中的D1-D4数据集分布
            self.dataset_config = {
                'D1': 62,  # 训练集
                'D2': 50,  # 内部验证集
                'D3': 27,  # 第一个独立测试集
                'D4': 23   # 第二个独立测试集
            }
        else:
            self.dataset_config = dataset_config
        
        # 计算总患者数
        self.total_patients = sum(self.dataset_config.values())
        
        # 响应者比例约为50%，与论文中的分布大致一致
        self.num_responsive = int(self.total_patients * 0.5)
        self.num_non_responsive = self.total_patients - self.num_responsive
        
        # 根据论文，关键的QVT特征包括角度、曲率和统计特征
        self.feature_generators = {
            # 血管角度特征 - 响应者治疗后锐角显著减少
            'angle_acute_count': {'baseline': (20, 5), 'responsive_post': (10, 3), 'non_responsive_post': (22, 6)},  # 锐角数量
            'angle_obtuse_count': {'baseline': (15, 4), 'responsive_post': (18, 5), 'non_responsive_post': (14, 4)},  # 钝角数量
            'angle_mean': {'baseline': (60, 15), 'responsive_post': (75, 18), 'non_responsive_post': (58, 16)},  # 平均角度
            
            # 血管曲率和曲折度特征
            'curvature_mean': {'baseline': (0.05, 0.02), 'responsive_post': (0.03, 0.01), 'non_responsive_post': (0.055, 0.025)},  # 平均曲率
            'tortuosity_mean': {'baseline': (1.2, 0.3), 'responsive_post': (1.0, 0.2), 'non_responsive_post': (1.25, 0.35)},  # 平均曲折度
            
            # 血管长度特征
            'vessel_length_mean': {'baseline': (10.0, 2.0), 'responsive_post': (9.0, 1.8), 'non_responsive_post': (10.5, 2.2)},  # 平均血管长度
            
            # 分支特征
            'branch_count': {'baseline': (8, 3), 'responsive_post': (6, 2), 'non_responsive_post': (9, 4)},  # 分支数量
        }
    
    def generate_data(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.Series], pd.Series]:
        """
        生成模拟数据
        
        Returns:
            baseline_datasets: 各数据集的基线QVT特征数据
            post_treatment_datasets: 各数据集的治疗后QVT特征数据
            response_label_datasets: 各数据集的响应标签
            all_response_labels: 所有患者的响应标签
        """
        # 初始化数据集字典
        baseline_datasets = {}
        post_treatment_datasets = {}
        response_label_datasets = {}
        # 生成患者ID，按照数据集分组
        patient_ids = []
        current_id = 1
        for dataset_name, dataset_size in self.dataset_config.items():
            for i in range(dataset_size):
                patient_ids.append(f'{dataset_name}_patient_{current_id}')
                current_id += 1
        
        # 创建响应标签
        response_labels = pd.Series([1] * self.num_responsive + [0] * self.num_non_responsive, 
                                  index=patient_ids, name='response')
        
        # 随机打乱标签顺序
        response_labels = response_labels.sample(frac=1).reset_index(drop=True)
        response_labels.index = patient_ids
        
        # 初始化所有患者的数据框
        all_baseline_data = pd.DataFrame(index=patient_ids)
        all_post_treatment_data = pd.DataFrame(index=patient_ids)
        
        # 生成每个特征
        for feature_name, distributions in self.feature_generators.items():
            # 生成基线数据
            mu, sigma = distributions['baseline']
            all_baseline_data[feature_name] = np.random.normal(mu, sigma, self.total_patients)
            
            # 为响应者和非响应者分别生成治疗后数据
            post_treatment_values = np.zeros(self.total_patients)
            
            # 响应者数据
            responsive_indices = response_labels[response_labels == 1].index
            mu_r, sigma_r = distributions['responsive_post']
            post_treatment_values[response_labels[response_labels == 1].index.get_indexer(responsive_indices)] = \
                np.random.normal(mu_r, sigma_r, len(responsive_indices))
            
            # 非响应者数据
            non_responsive_indices = response_labels[response_labels == 0].index
            mu_nr, sigma_nr = distributions['non_responsive_post']
            post_treatment_values[response_labels[response_labels == 0].index.get_indexer(non_responsive_indices)] = \
                np.random.normal(mu_nr, sigma_nr, len(non_responsive_indices))
            
            all_post_treatment_data[feature_name] = post_treatment_values
        
        # 确保所有值为正数（实际特征不应该为负）
        for df in [all_baseline_data, all_post_treatment_data]:
            for col in df.columns:
                df[col] = df[col].apply(lambda x: max(0, x))
        
        # 按照数据集配置划分数据
        for dataset_name, _ in self.dataset_config.items():
            # 筛选该数据集的患者ID
            dataset_patient_ids = [pid for pid in patient_ids if pid.startswith(dataset_name)]
            
            # 为每个数据集创建子数据框
            baseline_datasets[dataset_name] = all_baseline_data.loc[dataset_patient_ids].copy()
            post_treatment_datasets[dataset_name] = all_post_treatment_data.loc[dataset_patient_ids].copy()
            response_label_datasets[dataset_name] = response_labels.loc[dataset_patient_ids].copy()
        
        return baseline_datasets, post_treatment_datasets, response_label_datasets, response_labels
    
    def save_data(self, baseline_datasets: Dict[str, pd.DataFrame], 
                  post_treatment_datasets: Dict[str, pd.DataFrame], 
                  response_label_datasets: Dict[str, pd.Series],
                  all_response_labels: pd.Series,
                  output_dir: str, force_regenerate: bool = False) -> bool:
        """
        保存生成的数据到CSV文件，按照论文中的数据集分布保存
        
        Args:
            baseline_datasets: 各数据集的基线数据
            post_treatment_datasets: 各数据集的治疗后数据
            response_label_datasets: 各数据集的响应标签
            all_response_labels: 所有患者的响应标签
            output_dir: 输出目录
            force_regenerate: 是否强制重新生成数据
        
        Returns:
            bool: 是否已使用现有数据（未重新生成）
        """
        # 检查汇总数据是否已存在且不需要强制重新生成
        all_baseline_file = os.path.join(output_dir, 'all_baseline_qvt_features.csv')
        all_post_file = os.path.join(output_dir, 'all_post_treatment_qvt_features.csv')
        all_labels_file = os.path.join(output_dir, 'all_response_labels.csv')
        
        # 检查是否所有汇总文件都存在
        if all(os.path.exists(f) for f in [all_baseline_file, all_post_file, all_labels_file]) and not force_regenerate:
            print(f"数据文件已存在于目录: {output_dir}")
            print("直接使用现有数据，跳过重新生成。")
            # 读取现有数据以验证
            baseline_df = pd.read_csv(all_baseline_file)
            post_df = pd.read_csv(all_post_file)
            labels_df = pd.read_csv(all_labels_file)
            print(f"- 基线QVT特征: {len(baseline_df)} 条记录")
            print(f"- 治疗后QVT特征: {len(post_df)} 条记录")
            print(f"- 响应标签: {len(labels_df)} 条记录")
            return True
        
        # 创建主输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 为每个数据集创建子目录并保存数据
        for dataset_name in baseline_datasets.keys():
            dataset_dir = os.path.join(output_dir, dataset_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            # 保存该数据集的数据
            baseline_datasets[dataset_name].to_csv(os.path.join(dataset_dir, f'baseline_qvt_features.csv'))
            post_treatment_datasets[dataset_name].to_csv(os.path.join(dataset_dir, f'post_treatment_qvt_features.csv'))
            response_label_datasets[dataset_name].to_csv(os.path.join(dataset_dir, f'response_labels.csv'))
            
            print(f"数据集 {dataset_name} 已保存至 {dataset_dir}")
            print(f"  - 患者数量: {len(baseline_datasets[dataset_name])}")
            print(f"  - 响应者数量: {sum(response_label_datasets[dataset_name] == 1)}")
            print(f"  - 非响应者数量: {sum(response_label_datasets[dataset_name] == 0)}")
        
        # 同时保存整体数据集（便于某些分析）
        all_baseline = pd.concat(baseline_datasets.values())
        all_post = pd.concat(post_treatment_datasets.values())
        
        all_baseline.to_csv(all_baseline_file)
        all_post.to_csv(all_post_file)
        all_response_labels.to_csv(all_labels_file)
        
        print(f"\n汇总数据已保存至 {output_dir}")
        print(f"总患者数: {len(all_baseline)} 患者, {len(all_baseline.columns)} 特征")
        print(f"总响应者数量: {sum(all_response_labels == 1)}, 总非响应者数量: {sum(all_response_labels == 0)}")
        return False

if __name__ == "__main__":
    # 使用默认配置，模拟论文中的数据集分布（D1-D4）
    # 总患者数：62 + 50 + 27 + 23 = 162，与论文中的ICI治疗患者总数一致
    generator = SyntheticDataGenerator()
    baseline_datasets, post_treatment_datasets, response_label_datasets, all_labels = generator.generate_data()
    
    # 保存到默认目录
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    generator.save_data(baseline_datasets, post_treatment_datasets, response_label_datasets, all_labels, output_dir)