import os
import pandas as pd
import numpy as np
from delta_qvt_calculator import DeltaQVTCalculator
from generate_synthetic_data import SyntheticDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from scipy import stats

# Set matplotlib font
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # Ensure proper display of negative signs

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 创建结果目录
results_dir = os.path.join(current_dir, '..', 'results')
os.makedirs(results_dir, exist_ok=True)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Set font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # Ensure proper display of negative signs

class DeltaQVTExperiment:
    """
    Experiment 2: Delta QVT Feature Calculation and Analysis
    Based on the paper: Analyze delta QVT features (absolute changes in QVT features before and after treatment) and their association with ICI response
    """
    
    def __init__(self):
        # Set data and results directories
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.results_dir = os.path.join(self.base_dir, 'results')
        
        # Ensure results directory exists
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        
        # Initialize components
        self.calculator = DeltaQVTCalculator()
    
    def load_data(self) -> tuple:
        """
        Load baseline and post-treatment data
        
        Returns:
            baseline_features: Baseline QVT features
            post_treatment_features: Post-treatment QVT features
            response_labels: Response labels
        """
        baseline_path = os.path.join(self.data_dir, 'baseline_qvt_features.csv')
        post_treatment_path = os.path.join(self.data_dir, 'post_treatment_qvt_features.csv')
        labels_path = os.path.join(self.data_dir, 'response_labels.csv')
        
        baseline_features = pd.read_csv(baseline_path, index_col=0)
        post_treatment_features = pd.read_csv(post_treatment_path, index_col=0)
        response_labels = pd.read_csv(labels_path, index_col=0)['response']
        
        return baseline_features, post_treatment_features, response_labels
    
    def calculate_delta_features(self, baseline_features: pd.DataFrame, 
                               post_treatment_features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate delta QVT features
        
        Args:
            baseline_features: Baseline features
            post_treatment_features: Post-treatment features
        
        Returns:
            delta_features: Calculated delta QVT features
        """
        delta_features = self.calculator.calculate_delta_qvt(
            baseline_features, post_treatment_features
        )
        
        # 保存delta特征
        delta_path = os.path.join(self.data_dir, 'delta_qvt_features.csv')
        self.calculator.save_delta_features(delta_features, delta_path)
        
        return delta_features
    
    def analyze_angle_features(self, delta_features: pd.DataFrame, 
                             response_labels: pd.Series) -> None:
        """
        Analyze relationship between angle-related features and response
        According to the paper: Responders show significant reduction in acute vessel angles after treatment
        """
        # 提取角度相关特征
        angle_features = self.calculator.extract_angle_features(delta_features)
        
        # 添加响应标签
        analysis_data = pd.concat([angle_features, response_labels], axis=1)
        
        # Analyze feature differences between responders and non-responders
        print("\nAngle feature analysis:")
        print("-" * 50)
        
        for feature in angle_features.columns:
            # Check if feature contains 'angle_acute_count'
            if 'angle_acute_count' in feature:
                print(f"Feature: {feature}")
                print(f"  Responder mean: {analysis_data[analysis_data['response']==1][feature].mean():.3f}")
                print(f"  Non-responder mean: {analysis_data[analysis_data['response']==0][feature].mean():.3f}")
                non_resp_mean = analysis_data[analysis_data['response']==0][feature].mean()
                resp_mean = analysis_data[analysis_data['response']==1][feature].mean()
                print(f"  Difference ratio: {non_resp_mean / resp_mean:.2f}x")
        
        # Visualize angle feature differences
        self.plot_feature_comparison(analysis_data)
    
    def plot_feature_comparison(self, analysis_data: pd.DataFrame) -> None:
        """
        Visualize feature differences between responders and non-responders
        """
        try:
            # Dynamic import of visualization libraries
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set font to Times New Roman
            plt.rcParams['font.family'] = 'Times New Roman'
            plt.rcParams['axes.unicode_minus'] = False  # Ensure proper display of negative signs
            
            # Select angle-related feature
            angle_acute_feature = 'delta_angle_acute_count' if 'delta_angle_acute_count' in analysis_data.columns else None
            
            if angle_acute_feature:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x='response', y=angle_acute_feature, data=analysis_data)
                plt.title('Difference in Acute Angle Count Changes Between Responders and Non-Responders')
                plt.xlabel('Response Status (0=Non-responder, 1=Responder)')
                plt.ylabel('Delta Acute Angle Count')
                plt.xticks([0, 1], ['Non-responder', 'Responder'])
            
                # Add statistical information
                resp_mean = analysis_data[analysis_data['response']==1][angle_acute_feature].mean()
                non_resp_mean = analysis_data[analysis_data['response']==0][angle_acute_feature].mean()
                plt.text(0, non_resp_mean + 0.1, f'mean: {non_resp_mean:.2f}', ha='center')
                plt.text(1, resp_mean + 0.1, f'mean: {resp_mean:.2f}', ha='center')
                
                output_path = os.path.join(self.results_dir, 'acute_angle_comparison.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"\nVisualization plot saved to: {output_path}")
                
                # Clean up plot resources
                plt.close()
        except Exception as e:
            print(f"\nWarning: Visualization functionality temporarily unavailable ({str(e)})")
            print("Data processing and analysis will continue to execute")
    
    def run_experiment(self):
        """
        Run the complete Experiment 2 workflow
        """
        print("Starting Experiment 2: Delta QVT Feature Calculation and Analysis")
        print("-" * 60)
        
        # Step 1: Load data
        print("\nStep 1: Loading data...")
        baseline_features, post_treatment_features, response_labels = self.load_data()
        print(f"  Loaded: {len(baseline_features)} patients, {len(baseline_features.columns)} features")
        
        # Step 2: Calculate delta QVT features
        print("\nStep 2: Calculating delta QVT features...")
        delta_features = self.calculate_delta_features(baseline_features, post_treatment_features)
        print(f"  Calculated: {len(delta_features)} delta features")
        print(f"  Feature list: {', '.join(delta_features.columns)}")
        
        # Step 3: Analyze relationship between angle features and response
        print("\nStep 3: Analyzing relationship between angle features and response...")
        self.analyze_angle_features(delta_features, response_labels)
        
        print("\n" + "-" * 60)
        print("Experiment 2 completed!")
        print(f"Results saved in: {self.results_dir}")

import os
import pandas as pd
from generate_synthetic_data import SyntheticDataGenerator
from delta_qvt_calculator import DeltaQVTCalculator


def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Experiment 2: Delta QVT Feature Calculation and Analysis')
    parser.add_argument('--force-regenerate', action='store_true', help='Force regenerate data even if files already exist')
    args = parser.parse_args()
    
    # 设置数据和结果目录
    base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    results_dir = os.path.join(base_dir, 'results')
    
    # Generate synthetic data (dataset distribution from the paper)
    print("Generating synthetic data (dataset distribution D1-D4 from the paper)...")
    print("Total patients: 162, closer to actual conditions in the paper")
    
    # 检查数据是否存在且不需要强制重新生成
    baseline_file = os.path.join(data_dir, 'all_baseline_qvt_features.csv')
    post_treatment_file = os.path.join(data_dir, 'all_post_treatment_qvt_features.csv')
    labels_file = os.path.join(data_dir, 'all_response_labels.csv')
    
    if all(os.path.exists(f) for f in [baseline_file, post_treatment_file, labels_file]) and not args.force_regenerate:
        # Use existing data directly
        print(f"Data files already exist in directory: {data_dir}")
        print("Using existing data directly, skipping regeneration.")
        
        # Read existing data
        all_baseline = pd.read_csv(baseline_file, index_col=0)
        all_post = pd.read_csv(post_treatment_file, index_col=0)
        all_labels = pd.read_csv(labels_file, index_col=0).squeeze()
        
        # Split data by dataset
        baseline_datasets = {}
        post_treatment_datasets = {}
        response_label_datasets = {}
        
        for dataset in ['D1', 'D2', 'D3', 'D4']:
            # Identify datasets based on patient IDs (ID format like 'D1_patient_1')
            mask = all_baseline.index.str.startswith(dataset)
            baseline_datasets[dataset] = all_baseline[mask]
            post_treatment_datasets[dataset] = all_post[mask]
            response_label_datasets[dataset] = all_labels[mask]
    else:
        # Regenerate data
        generator = SyntheticDataGenerator()  # Use default configuration to simulate dataset distribution from the paper
        baseline_datasets, post_treatment_datasets, response_label_datasets, all_labels = generator.generate_data()
        
        # Save generated data
        generator.save_data(baseline_datasets, post_treatment_datasets, response_label_datasets, all_labels, data_dir)
    
    # Calculate delta QVT features
    print("\nCalculating delta QVT features...")
    calculator = DeltaQVTCalculator()
    delta_qvt_datasets = calculator.calculate_delta_qvt(baseline_datasets, post_treatment_datasets)
    
    # Extract angle features for each dataset
    print("\nExtracting angle features...")
    angle_feature_datasets = {}
    for dataset_name, delta_features in delta_qvt_datasets.items():
        angle_feature_datasets[dataset_name] = calculator.extract_angle_features(delta_features)
        print(f"Extracted angle features for dataset {dataset_name}")
    
    # Save delta QVT features
    print("\nSaving results...")
    calculator.save_delta_features(delta_qvt_datasets, results_dir)
    
    # Save angle features
    for dataset_name, angle_features in angle_feature_datasets.items():
        dataset_results_dir = os.path.join(results_dir, dataset)
        if not os.path.exists(dataset_results_dir):
            os.makedirs(dataset_results_dir, exist_ok=True)
        calculator.save_delta_features_single(
            angle_features, 
            os.path.join(dataset_results_dir, 'delta_angle_features.csv')
        )
    
    # Save combined angle features
    all_angle_features = pd.concat(angle_feature_datasets.values())
    all_angle_features.to_csv(os.path.join(results_dir, 'all_delta_angle_features.csv'))
    
    # Save labels for further analysis
    for dataset_name, labels in response_label_datasets.items():
        dataset_results_dir = os.path.join(results_dir, dataset_name)
        labels.to_csv(os.path.join(dataset_results_dir, 'response_labels.csv'))
    
    # Save combined labels
    all_labels.to_csv(os.path.join(results_dir, 'all_response_labels.csv'))
    
    print("\nProcessing complete!")
    print(f"Data saved in directory: {data_dir}")
    print(f"Results saved in directory: {results_dir}")
    print("\nDataset distribution:")
    for dataset_name in baseline_datasets.keys():
        n_patients = len(baseline_datasets[dataset_name])
        n_responsive = sum(response_label_datasets[dataset_name] == 1)
        print(f"  - {dataset_name}: {n_patients} patients ({n_responsive} responders, {n_patients - n_responsive} non-responders)")


if __name__ == "__main__":
    main()