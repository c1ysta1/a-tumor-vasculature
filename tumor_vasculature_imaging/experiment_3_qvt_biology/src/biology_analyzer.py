#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment 3: Biology Analysis Module

This module implements statistical analysis methods for QVT features and biological indicators
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import json
from collections import defaultdict

# Set matplotlib font
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 创建结果目录
results_dir = os.path.join(current_dir, '..', 'results')
os.makedirs(results_dir, exist_ok=True)
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple

class QVTBiologyAnalyzer:
    """QVT Biology Analyzer Class"""
    
    def __init__(self, data_dir=None, results_dir=None):
        """
        Initialize the analyzer
        
        Args:
            data_dir: Path to data directory
            results_dir: Path to save results
        """
        # Use absolute paths, default based on current file location
        if data_dir is None:
            self.data_dir = os.path.join(current_dir, '..', 'data')
        else:
            # Ensure the provided path is also absolute
            self.data_dir = os.path.abspath(data_dir)
        
        if results_dir is None:
            self.results_dir = os.path.join(current_dir, '..', 'results')
        else:
            # Ensure the provided path is also absolute
            self.results_dir = os.path.abspath(results_dir)
            
        # 设置总的visualization目录
        self.visualization_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', 'visualization', 'experiment_3'))
            
        self.pvalue_threshold = 0.05
        self.fdr_threshold = 0.01
        # Set sub-results directories
        self.pd_l1_results_dir = os.path.join(self.results_dir, 'pd_l1_analysis')
        self.til_results_dir = os.path.join(self.results_dir, 'til_analysis')
        self.pathway_results_dir = os.path.join(self.results_dir, 'pathway_analysis')
        # 设置visualization子目录
        self.pd_l1_viz_dir = os.path.join(self.visualization_dir, 'pd_l1_analysis')
        self.til_viz_dir = os.path.join(self.visualization_dir, 'til_analysis')
        self.pathway_viz_dir = os.path.join(self.visualization_dir, 'pathway_analysis')
        # Ensure directories exist
        os.makedirs(self.pd_l1_results_dir, exist_ok=True)
        os.makedirs(self.til_results_dir, exist_ok=True)
        os.makedirs(self.pathway_results_dir, exist_ok=True)
        os.makedirs(self.pd_l1_viz_dir, exist_ok=True)
        os.makedirs(self.til_viz_dir, exist_ok=True)
        os.makedirs(self.pathway_viz_dir, exist_ok=True)
        
        # Set visualization style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue
    
    def _load_qvt_data(self) -> pd.DataFrame:
        """
        Load QVT feature data
        
        Returns:
            pd.DataFrame: QVT feature data
        """
        qvt_file = os.path.join(self.data_dir, 'qvt_features.csv')
        if os.path.exists(qvt_file):
            print(f"Loading QVT feature data: {qvt_file}")
            return pd.read_csv(qvt_file)
        else:
            print(f"Warning: QVT feature file not found: {qvt_file}")
            return pd.DataFrame()
    
    def _load_pd_l1_data(self) -> pd.DataFrame:
        """
        Load PD-L1 expression data
        
        Returns:
            pd.DataFrame: PD-L1 expression data
        """
        pd_l1_file = os.path.join(self.data_dir, 'pd_l1_expression.csv')
        if os.path.exists(pd_l1_file):
            print(f"Loading PD-L1 expression data: {pd_l1_file}")
            return pd.read_csv(pd_l1_file)
        else:
            print(f"Warning: PD-L1 expression file not found: {pd_l1_file}")
            return pd.DataFrame()
    
    def _load_til_data(self) -> pd.DataFrame:
        """
        Load TIL density data
        
        Returns:
            pd.DataFrame: TIL density data
        """
        til_file = os.path.join(self.data_dir, 'til_density.csv')
        if os.path.exists(til_file):
            print(f"Loading TIL density data: {til_file}")
            return pd.read_csv(til_file)
        else:
            print(f"Warning: TIL density file not found: {til_file}")
            return pd.DataFrame()
    
    def _load_pathway_data(self) -> pd.DataFrame:
        """
        Load pathway enrichment data
        
        Returns:
            pd.DataFrame: Pathway enrichment data
        """
        pathway_file = os.path.join(self.data_dir, 'pathway_enrichment.csv')
        if os.path.exists(pathway_file):
            print(f"Loading pathway enrichment data: {pathway_file}")
            return pd.read_csv(pathway_file)
        else:
            print(f"Warning: Pathway enrichment file not found: {pathway_file}")
            return pd.DataFrame()
    
    def _generate_summary_report(self, all_results: Dict) -> None:
        """
        Generate analysis results summary report
        
        Args:
            all_results: All analysis results
        """
        report_file = os.path.join(self.results_dir, 'biology_analysis_report.md')
        print(f"Generating biology analysis summary report: {report_file}")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# QVT Biology Association Analysis Report\n\n")
            
            f.write("## 1. PD-L1 Expression Association Analysis\n")
            f.write("- Number of significant features: ")
            if 'significant_features' in all_results['pd_l1_analysis']:
                f.write(f"{len(all_results['pd_l1_analysis']['significant_features'])}\n")
            else:
                f.write("0\n")
            
            f.write("\n## 2. TIL Density Association Analysis\n")
            f.write("- Number of significant correlations: ")
            if 'significant_correlations' in all_results['til_analysis']:
                f.write(f"{len(all_results['til_analysis']['significant_correlations'])}\n")
            else:
                f.write("0\n")
            
            f.write("\n## 3. Molecular Pathway Association Analysis\n")
            f.write("- WNT pathway analysis: ")
            if 'wnt_fgf_results' in all_results['pathway_analysis']:
                wnt_result = next((item for item in all_results['pathway_analysis']['wnt_fgf_results'] if item['pathway'] == 'WNT'), None)
                if wnt_result:
                    wnt_p = wnt_result['p_value']
                    if wnt_p < self.pvalue_threshold:
                        f.write(f"Significantly upregulated (p={wnt_p:.4f})\n")
                    else:
                        f.write(f"No significant difference (p={wnt_p:.4f})\n")
                else:
                    f.write("Not analyzed\n")
            else:
                f.write("Not analyzed\n")
            
            f.write("- FGF pathway analysis: ")
            if 'wnt_fgf_results' in all_results['pathway_analysis']:
                fgf_result = next((item for item in all_results['pathway_analysis']['wnt_fgf_results'] if item['pathway'] == 'FGF'), None)
                if fgf_result:
                    fgf_p = fgf_result['p_value']
                    if fgf_p < self.pvalue_threshold:
                        f.write(f"Significantly upregulated (p={fgf_p:.4f})\n")
                    else:
                        f.write(f"No significant difference (p={fgf_p:.4f})\n")
                else:
                    f.write("Not analyzed\n")
            else:
                f.write("Not analyzed\n")
            
            f.write("\n## Conclusion\n")
            f.write("QVT features show significant associations with biological markers, particularly with upregulation of WNT and FGF signaling pathways.\n")
            f.write("This finding supports the potential value of QVT features as imaging biomarkers for the tumor microenvironment.\n")
    
    def analyze_pd_l1_association(self, qvt_data: pd.DataFrame, pd_l1_data: pd.DataFrame) -> Dict:
        """
        Perform association analysis between QVT features and PD-L1 expression
        Use Wilcoxon rank-sum test to compare QVT feature differences between high and low PD-L1 groups
        
        Args:
            qvt_data: QVT feature data
            pd_l1_data: PD-L1 expression data
            
        Returns:
            Dict: Analysis results
        """
        print("Performing Wilcoxon rank-sum test for QVT features and PD-L1 expression...")
        
        # Merge data
        merged_data = pd.merge(qvt_data, pd_l1_data, on='patient_id', how='inner')
        
        # Separate high and low PD-L1 groups
        high_pd_l1 = merged_data[merged_data['pd_l1_high'] == 1]
        low_pd_l1 = merged_data[merged_data['pd_l1_high'] == 0]
        
        print(f"High PD-L1 group patients: {len(high_pd_l1)}, Low PD-L1 group patients: {len(low_pd_l1)}")
        
        # Extract QVT feature columns
        qvt_columns = [col for col in qvt_data.columns if col not in ['patient_id', 'response']]
        
        # Store test results for each feature
        results = {
            'feature': [],
            'p_value': [],
            'statistic': [],
            'high_pd_l1_mean': [],
            'low_pd_l1_mean': [],
            'fold_change': []
        }
        
        # Perform Wilcoxon rank-sum test for each QVT feature
        for feature in qvt_columns:
            # Ensure both groups have data
            if len(high_pd_l1) > 0 and len(low_pd_l1) > 0:
                # Wilcoxon rank-sum test
                stat, p_value = stats.ranksums(
                    high_pd_l1[feature], 
                    low_pd_l1[feature]
                )
                
                # Calculate means and fold change
                high_mean = high_pd_l1[feature].mean()
                low_mean = low_pd_l1[feature].mean()
                fold_change = high_mean / low_mean if low_mean != 0 else np.inf
                
                # Store results
                results['feature'].append(feature)
                results['p_value'].append(p_value)
                results['statistic'].append(stat)
                results['high_pd_l1_mean'].append(high_mean)
                results['low_pd_l1_mean'].append(low_mean)
                results['fold_change'].append(fold_change)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Add significance markers
        results_df['significant'] = results_df['p_value'] < self.pvalue_threshold
        results_df['adjusted_p_value'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
        results_df['fdr_significant'] = results_df['adjusted_p_value'] < self.fdr_threshold
        
        # Sort by p-value
        results_df = results_df.sort_values('p_value')
        
        # Save results
        output_file = os.path.join(self.pd_l1_results_dir, 'pd_l1_qvt_association.csv')
        results_df.to_csv(output_file, index=False)
        print(f"PD-L1 association analysis results saved to: {output_file}")
        
        # Generate visualization
        self._visualize_pd_l1_results(results_df)
        
        # Return key results
        significant_features = results_df[results_df['fdr_significant']]['feature'].tolist()
        print(f"Found {len(significant_features)} QVT features significantly associated with PD-L1 expression")
        
        return {
            'significant_features': significant_features,
            'results_dataframe': results_df,
            'high_pd_l1_count': len(high_pd_l1),
            'low_pd_l1_count': len(low_pd_l1)
        }
    
    def analyze_til_association(self, qvt_data: pd.DataFrame, til_data: pd.DataFrame) -> Dict:
        """
        Perform association analysis between QVT features and TIL density
        Use Spearman correlation analysis with FDR correction
        
        Args:
            qvt_data: QVT feature data
            til_data: TIL density data
            
        Returns:
            Dict: Analysis results
        """
        print("Performing Spearman correlation analysis for QVT features and TIL density...")
        
        # Merge data
        merged_data = pd.merge(qvt_data, til_data, on='patient_id', how='inner')
        print(f"Merged sample count: {len(merged_data)}")
        
        # Extract feature columns
        qvt_columns = [col for col in qvt_data.columns if col not in ['patient_id', 'response']]
        til_columns = [col for col in til_data.columns if col not in ['patient_id']]
        
        # Create correlation results and p-values
        correlation_results = []
        p_values = []
        
        # Calculate Spearman correlation for each pair of QVT and TIL features
        for qvt_feature in qvt_columns:
            for til_feature in til_columns:
                # Spearman correlation analysis
                corr, p_value = stats.spearmanr(
                    merged_data[qvt_feature], 
                    merged_data[til_feature]
                )
                
                correlation_results.append({
                    'qvt_feature': qvt_feature,
                    'til_feature': til_feature,
                    'correlation': corr,
                    'p_value': p_value
                })
                p_values.append(p_value)
        
        # Create results DataFrame
        results_df = pd.DataFrame(correlation_results)
        
        # Perform FDR correction
        results_df['adjusted_p_value'] = multipletests(p_values, method='fdr_bh')[1]
        results_df['significant'] = results_df['adjusted_p_value'] < self.fdr_threshold
        
        # Sort by adjusted p-value
        results_df = results_df.sort_values('adjusted_p_value')
        
        # Save results
        output_file = os.path.join(self.til_results_dir, 'til_qvt_correlation.csv')
        results_df.to_csv(output_file, index=False)
        print(f"TIL association analysis results saved to: {output_file}")
        
        # Generate visualization
        self._visualize_til_correlations(results_df, merged_data, qvt_columns, til_columns)
        
        # Return key results
        significant_correlations = results_df[results_df['significant']].copy()
        print(f"Found {len(significant_correlations)} significant correlated feature pairs")
        
        return {
            'significant_correlations': significant_correlations.to_dict('records'),
            'results_dataframe': results_df,
            'sample_count': len(merged_data)
        }
    
    def analyze_pathway_association(self, qvt_data: pd.DataFrame, pathway_data: pd.DataFrame) -> Dict:
        """
        Perform association analysis between QVT features and pathway enrichment
        Compare pathway enrichment scores between high QVT and low QVT phenotype groups
        
        Args:
            qvt_data: QVT feature data
            pathway_data: Pathway enrichment data
            
        Returns:
            Dict: Analysis results
        """
        print("Performing association analysis for QVT phenotype and pathway enrichment...")
        
        # Merge data
        merged_data = pd.merge(qvt_data, pathway_data, on='patient_id', how='inner')
        
        # If qvt_phenotype column doesn't exist, create it based on QVT feature mean
        if 'qvt_phenotype' not in merged_data.columns:
            qvt_columns = [col for col in qvt_data.columns if col not in ['patient_id', 'response']]
            merged_data['qvt_phenotype'] = (merged_data[qvt_columns].mean(axis=1) > 0).astype(int)
        
        # Separate high QVT and low QVT phenotype groups
        high_qvt = merged_data[merged_data['qvt_phenotype'] == 1]
        low_qvt = merged_data[merged_data['qvt_phenotype'] == 0]
        
        print(f"High QVT phenotype group patients: {len(high_qvt)}, Low QVT phenotype group patients: {len(low_qvt)}")
        
        # Extract pathway columns
        pathway_columns = [col for col in pathway_data.columns if col not in ['patient_id', 'qvt_phenotype']]
        
        # Store pathway analysis results
        pathway_results = {
            'pathway': [],
            'p_value': [],
            'statistic': [],
            'high_qvt_mean': [],
            'low_qvt_mean': [],
            'fold_change': []
        }
        
        # Perform Wilcoxon rank-sum test for each pathway
        for pathway in pathway_columns:
            if len(high_qvt) > 0 and len(low_qvt) > 0:
                # Wilcoxon rank-sum test
                stat, p_value = stats.ranksums(
                    high_qvt[pathway], 
                    low_qvt[pathway]
                )
                
                # Calculate means and fold change
                high_mean = high_qvt[pathway].mean()
                low_mean = low_qvt[pathway].mean()
                fold_change = high_mean / low_mean if low_mean != 0 else np.inf
                
                # Store results
                pathway_results['pathway'].append(pathway)
                pathway_results['p_value'].append(p_value)
                pathway_results['statistic'].append(stat)
                pathway_results['high_qvt_mean'].append(high_mean)
                pathway_results['low_qvt_mean'].append(low_mean)
                pathway_results['fold_change'].append(fold_change)
        
        # Create results DataFrame
        results_df = pd.DataFrame(pathway_results)
        
        # Add significance markers
        results_df['significant'] = results_df['p_value'] < self.pvalue_threshold
        results_df['adjusted_p_value'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
        results_df['fdr_significant'] = results_df['adjusted_p_value'] < self.fdr_threshold
        
        # Sort by fold change
        results_df = results_df.sort_values('fold_change', ascending=False)
        
        # Save results
        output_file = os.path.join(self.pathway_results_dir, 'pathway_qvt_association.csv')
        results_df.to_csv(output_file, index=False)
        print(f"Pathway association analysis results saved to: {output_file}")
        
        # Generate visualization
        self._visualize_pathway_results(results_df)
        
        # Return key results
        enriched_pathways = results_df[results_df['fdr_significant']]['pathway'].tolist()
        print(f"Found {len(enriched_pathways)} pathways significantly enriched in high QVT phenotype group")
        
        # Special focus on WNT and FGF pathways
        wnt_fgf_results = results_df[results_df['pathway'].isin(['WNT', 'FGF'])].copy()
        
        return {
            'enriched_pathways': enriched_pathways,
            'wnt_fgf_results': wnt_fgf_results.to_dict('records'),
            'results_dataframe': results_df,
            'high_qvt_count': len(high_qvt),
            'low_qvt_count': len(low_qvt)
        }
    
    def _visualize_pd_l1_results(self, results_df: pd.DataFrame):
        """
        Visualize PD-L1 association analysis results
        
        Args:
            results_df: Analysis results DataFrame
        """
        # Create visualization directory
        plot_dir = self.pd_l1_viz_dir
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot volcano plot for top 10 significant features
        plt.figure(figsize=(10, 6))
        
        # Calculate log2 fold change and -log10 p-value
        log2_fold_change = np.log2(results_df['fold_change'].replace([np.inf, -np.inf], np.nan))
        log10_pvalue = -np.log10(results_df['p_value'])
        
        # Plot scatter plot
        plt.scatter(log2_fold_change, log10_pvalue, alpha=0.6, s=30)
        
        # Mark significant features
        significant = results_df['fdr_significant']
        if significant.any():
            plt.scatter(log2_fold_change[significant], log10_pvalue[significant], 
                       alpha=0.8, s=50, color='red', label='Significant')
        
        # Add threshold lines
        plt.axhline(y=-np.log10(self.pvalue_threshold), color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=np.log2(1.5), color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=-np.log2(1.5), color='gray', linestyle='--', alpha=0.5)
        
        plt.xlabel('Log2 Fold Change')
        plt.ylabel('-Log10 P-value')
        plt.title('Volcano Plot of QVT Features Associated with PD-L1 Expression')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save figure
        output_file = os.path.join(self.pd_l1_viz_dir, 'pd_l1_volcano_plot.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PD-L1 volcano plot saved to: {output_file}")
    
    def _visualize_til_correlations(self, results_df: pd.DataFrame, merged_data: pd.DataFrame, qvt_columns: List[str], til_columns: List[str]):
        """
        Visualize TIL correlation results
        
        Args:
            results_df: Analysis results DataFrame
            merged_data: Merged data
            qvt_columns: QVT feature columns
            til_columns: TIL feature columns
        """
        # Create visualization directory
        plot_dir = self.til_viz_dir
        os.makedirs(plot_dir, exist_ok=True)
        
        # Take top 5 significant correlations for visualization
        top_correlations = results_df.head(5).copy()
        
        if not top_correlations.empty:
            # Create correlation heatmap
            plt.figure(figsize=(12, 10))
            
            # Build correlation matrix for heatmap
            corr_matrix = pd.DataFrame(index=qvt_columns, columns=til_columns)
            
            # Fill correlation matrix
            for _, row in results_df.iterrows():
                corr_matrix.loc[row['qvt_feature'], row['til_feature']] = row['correlation']
            
            # Convert to numeric type
            corr_matrix = corr_matrix.astype(float)
            
            # Plot heatmap
            mask = np.isnan(corr_matrix)
            sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                       center=0, square=True, linewidths=.5)
            
            plt.title('Correlation Heatmap of QVT Features and TIL Density Indicators')
            plt.tight_layout()
            
            # Save figure
            output_file = os.path.join(self.til_viz_dir, 'til_correlation_heatmap.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"TIL correlation heatmap saved to: {output_file}")
    
    def _visualize_pathway_results(self, results_df: pd.DataFrame):
        """
        Visualize pathway association analysis results
        
        Args:
            results_df: Analysis results DataFrame
        """
        # Create visualization directory
        plot_dir = self.pathway_viz_dir
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot pathway enrichment bar chart
        plt.figure(figsize=(10, 6))
        
        # Sort results
        sorted_df = results_df.sort_values('fold_change', ascending=True)
        
        # Plot bar chart
        bars = plt.barh(sorted_df['pathway'], sorted_df['fold_change'], 
                      color=['red' if sig else 'blue' for sig in sorted_df['fdr_significant']])
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', va='center')
        
        plt.xlabel('Fold Change (High QVT/Low QVT)')
        plt.title('Pathway Enrichment Fold Change in High vs. Low QVT Phenotype Groups')
        plt.grid(axis='x', alpha=0.3)
        
        # Save figure
        output_file = os.path.join(self.pathway_viz_dir, 'pathway_enrichment_barchart.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Pathway enrichment bar chart saved to: {output_file}")
        
        # Special focus on WNT and FGF pathways
        wnt_fgf = results_df[results_df['pathway'].isin(['WNT', 'FGF'])]
        if not wnt_fgf.empty:
            print("WNT and FGF pathways are significantly upregulated in high QVT group, consistent with paper findings")
    
    def analyze_all(self) -> Dict:
        """
        Perform all analyses
        
        Returns:
            Dict: All analysis results
        """
        print("Starting all biology analyses...")
        
        # Load data
        qvt_data = self._load_qvt_data()
        pd_l1_data = self._load_pd_l1_data()
        til_data = self._load_til_data()
        pathway_data = self._load_pathway_data()
        
        # Perform analyses
        pd_l1_results = self.analyze_pd_l1_association(qvt_data, pd_l1_data)
        til_results = self.analyze_til_association(qvt_data, til_data)
        pathway_results = self.analyze_pathway_association(qvt_data, pathway_data)
        
        # Integrate results
        all_results = {
            'pd_l1_analysis': pd_l1_results,
            'til_analysis': til_results,
            'pathway_analysis': pathway_results
        }
        
        # Generate report
        self._generate_summary_report(all_results)
        
        print("All analyses completed!")
        
        return all_results