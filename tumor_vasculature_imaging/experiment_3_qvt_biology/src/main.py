"""
实验三：QVT生物标记物研究主入口
1. QVT特征与PD-L1表达的关联
2. QVT特征与TIL密度的关联
3. QVT表型与分子通路的关联
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Dict, List, Tuple

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块
from generate_synthetic_data import SyntheticDataGenerator
from biology_analyzer import QVTBiologyAnalyzer
from validate_results import ResultsValidator

def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Experiment 3: QVT Biomarker Study')
    parser.add_argument('--action', type=str, 
                        choices=['generate_data', 'analyze', 'validate', 'all'],
                        default='all',
                        help='Operation to perform: generate data, analyze data, validate results, or all')
    parser.add_argument('--data-dir', type=str, 
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'),
                        help='Data directory path')
    parser.add_argument('--results-dir', type=str, 
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'),
                        help='Results directory path')
    parser.add_argument('--force-regenerate', action='store_true', help='强制重新生成数据，即使文件已存在')
    
    args = parser.parse_args()
    
    # Set English font
    plt.rcParams['font.family'] = ['Times New Roman']
    plt.rcParams['axes.unicode_minus'] = False  # Ensure correct display of minus signs
    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置数据和结果目录
    data_dir = args.data_dir
    results_dir = args.results_dir
    
    # 确保目录存在
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'pd_l1_analysis'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'til_analysis'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'pathway_analysis'), exist_ok=True)
    
    print(f"Experiment 3: QVT Biomarker Study")
    print(f"Data directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Operation: {args.action}")
    
    # 执行指定操作
    if args.action == 'generate_data' or args.action == 'all':
        print("\n1. Generating synthetic data...")
        data_generator = SyntheticDataGenerator(data_dir)
        data_generator.generate_all_data(force_regenerate=args.force_regenerate)
    
    if args.action == 'analyze' or args.action == 'all':
        print("\n2. Performing biological association analysis...")
        biology_analyzer = QVTBiologyAnalyzer(data_dir, results_dir)
        biology_analyzer.analyze_all()
    
    if args.action == 'validate' or args.action == 'all':
        print("\n3. Validating results against paper...")
        validator = ResultsValidator(data_dir, results_dir)
        validation_results = validator.run_validation()
    
    print("\nExperiment 3 completed!")
    if args.action == 'validate' or args.action == 'all':
        if validation_results['overall']['passed']:
            print("✅ All validations passed! Simulation results are consistent with paper findings.")
        else:
            print("⚠️ Some validations failed, please check the results.")

if __name__ == "__main__":
    main()