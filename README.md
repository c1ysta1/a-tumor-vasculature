# 肿瘤血管成像生物标志物研究项目

本项目基于论文《A Tumor Vasculature-Based Imaging Biomarker for Predicting Response to Immune Checkpoint Inhibitors》(Science Advances, 2022)进行复现，旨在研究肿瘤血管特征(QVT)作为免疫检查点抑制剂(ICI)治疗响应预测的生物标志物。

## 📋 项目概述

本项目通过模拟肿瘤血管数据，实现了论文中提出的三个主要实验：

1. **QVT特征分析实验**：分析肿瘤血管特征(QVT)与治疗响应的关系，并使用LDA模型进行分类预测
2. **Delta QVT特征分析实验**：分析治疗前后血管特征变化与治疗响应的关联
3. **QVT与生物学标志物关联研究**：探索QVT特征与PD-L1表达、TIL密度及通路富集的关联

## 📁 项目结构

```
├── DocuGenius/                  # 文档生成工具
├── extracted_figures/           # 从论文中提取的图表
│   ├── page_3/
│   ├── page_4/
│   ├── page_5/
│   ├── page_6/
│   ├── page_7/
│   └── page_8/
├── tumor_vasculature_imaging/   # 主项目目录
│   ├── experiment_1_qvt/        # 实验1：QVT特征分析
│   │   ├── data/                # 数据目录
│   │   ├── results/             # 结果目录
│   │   └── src/                 # 源代码目录
│   ├── experiment_2_delta_qvt/  # 实验2：Delta QVT特征分析
│   │   ├── data/                # 数据目录
│   │   ├── results/             # 结果目录
│   │   └── src/                 # 源代码目录
│   └── experiment_3_qvt_biology/# 实验3：QVT与生物学标志物关联研究
│       ├── data/                # 数据目录
│       ├── results/             # 结果目录
│       └── src/                 # 源代码目录
├── visualization/               # 可视化结果目录
│   ├── experiment_1/
│   │   └── figures/
│   ├── experiment_2/
│   │   ├── figures/
│   │   └── survival_plots/
│   └── experiment_3/
│       ├── figures/
│       ├── pathway_analysis/
│       ├── pd_l1_analysis/
│       └── til_analysis/
├── a tumor vasculature based imaging bi source sci adv so 2022.pdf  # 原始论文
└── README.md                    # 项目说明文档
```

## 🚀 安装指南

### 环境要求

- Python 3.8+
- 所需Python包：numpy, pandas, matplotlib, seaborn, scikit-learn, scipy

### 安装步骤

1. 克隆或下载本项目到本地
2. 安装依赖包：

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

## 📊 实验运行

### 实验1：QVT特征分析

```bash
cd tumor_vasculature_imaging/experiment_1_qvt/src
python generate_qvt_data.py  # 生成或加载QVT特征数据
python qvt_lda_classification.py  # 执行LDA分类分析
```

### 实验2：Delta QVT特征分析

```bash
cd tumor_vasculature_imaging/experiment_2_delta_qvt/src
python main.py  # 运行实验2主流程
```

### 实验3：QVT与生物学标志物关联研究

```bash
cd tumor_vasculature_imaging/experiment_3_qvt_biology/src
python main.py  # 运行实验3主流程
```

## 📈 结果说明

### 实验1结果
- 生成的可视化图表保存在 `visualization/experiment_1/figures/` 目录下
- 包括混淆矩阵、ROC曲线、LDA散点图、特征重要性图等
- 分类准确率指标和详细结果保存在 `experiment_1_qvt/results/` 目录下

### 实验2结果
- Delta QVT特征数据保存在 `experiment_2_delta_qvt/results/` 目录下
- 生存分析图表保存在 `visualization/experiment_2/survival_plots/` 目录下
- 数据集分布统计输出到控制台

### 实验3结果
- 各生物学标志物关联分析结果保存在 `experiment_3_qvt_biology/results/` 目录下
- 可视化结果保存在 `visualization/experiment_3/` 目录下的对应子目录中
- 验证报告以Markdown格式保存

## 💡 关键功能说明

### 数据生成
- 项目使用模拟数据生成器生成符合论文特征分布的数据
- 支持自定义数据集规模和分布参数
- 可通过 `--force-regenerate` 参数强制重新生成数据

### 特征分析
- 支持血管角度、曲率、长度等多种QVT特征的提取和分析
- 实现了治疗前后特征变化(Delta QVT)的计算
- 提供特征重要性分析和可视化功能

### 分类预测
- 实现了LDA(Latent Dirichlet Allocation)分类器
- 提供5折交叉验证评估模型性能
- 包含过拟合检测功能

### 生存分析
- 基于QVT特征生成生存数据
- 绘制KM(Kaplan-Meier)生存曲线
- 提供特征热图分析

### 生物学关联研究
- 分析QVT特征与PD-L1表达的关联
- 探索QVT特征与TIL密度的相关性
- 研究QVT表型与通路富集的关系

## 🔍 论文关键发现

1. **血管锐角减少作为响应预测指标**：响应者治疗后血管锐角数量显著减少
2. **QVT特征预测能力**：QVT特征能有效预测ICI治疗响应
3. **生物学意义**：QVT特征与免疫微环境标志物(PD-L1、TIL)存在显著关联

## 📝 注意事项

- 本项目使用模拟数据进行论文复现，实际应用时需要替换为真实临床数据
- 数据生成器已按照论文中描述的分布特征进行配置
- 所有可视化图表均按论文风格生成，便于直观比较

## 🛠️ 自定义与扩展

### 替换数据集
1. 按照现有数据格式准备自己的数据集
2. 将数据文件放置在对应实验的数据目录中
3. 确保文件名与现有命名一致

### 修改模型参数
- 可通过修改各实验中的相应配置调整模型参数
- 特征生成器参数可在 `generate_synthetic_data.py` 中自定义

## 📚 引用

如果您使用本项目进行研究或开发，请引用原始论文：

```
A Tumor Vasculature-Based Imaging Biomarker for Predicting Response to Immune Checkpoint Inhibitors
Science Advances, 2022
```

## 🔧 技术支持

如遇任何问题，请在项目Issues中提交。

---

*本项目为学术研究用途，旨在复现论文方法和结果，促进肿瘤血管成像生物标志物研究的发展。*