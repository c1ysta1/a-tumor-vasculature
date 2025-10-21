import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, classification_report
import seaborn as sns
import os
import json

# Set font to Times New Roman for all text elements
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False  # Ensure proper display of negative signs
plt.rcParams['font.size'] = 12  # Set default font size
plt.rcParams['axes.titlesize'] = 14  # Set title font size
plt.rcParams['axes.labelsize'] = 12  # Set axis label font size

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 创建结果目录
# 使用统一的visualization文件夹
results_dir = os.path.join('d:\\杂七杂八的东西\\竞赛\\论文复现\\visualization\\experiment_1\\figures')
os.makedirs(results_dir, exist_ok=True)

# 加载数据
data_path = os.path.join(current_dir, '..', 'data', 'qvt_features_data.csv')
df = pd.read_csv(data_path)

# 分离特征和标签
X = df.drop('response', axis=1)
y = df['response']

# 显示一些关键特征的重要性
print("\nKey feature statistics:")
print(f"Total number of features: {X.shape[1]}")
print(f"Number of responder samples: {np.sum(y == 1)}")
print(f"Number of non-responder samples: {np.sum(y == 0)}")

# Display example of first 10 feature names
feature_names = X.columns.tolist()
print("\nExample feature names:")
print(f"First 10 features: {feature_names[:10]}")

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection to prevent overfitting
from sklearn.feature_selection import SelectKBest, f_classif
print("\nPerforming feature selection to prevent overfitting...")
selector = SelectKBest(f_classif, k=min(20, X.shape[1]))  # Select top 20 features or all if less than 20
X_scaled_selected = selector.fit_transform(X_scaled, y)
selected_features_mask = selector.get_support()
selected_feature_names = [feature_names[i] for i, selected in enumerate(selected_features_mask) if selected]
print(f"Selected {len(selected_feature_names)} features out of {len(feature_names)}")
print(f"Selected features: {selected_feature_names}")

# 保存标准化器参数
scaler_params = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist()
}
# 数据文件仍然保存在原始results目录
original_results_dir = os.path.join(current_dir, '..', 'results')
os.makedirs(original_results_dir, exist_ok=True)
with open(os.path.join(original_results_dir, 'scaler_params.json'), 'w') as f:
    json.dump(scaler_params, f, indent=4)

# 实现三倍交叉验证
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# 存储每个折叠的评估结果
fold_results = []
all_y_true = []
all_y_pred = []
all_y_pred_proba = []

# 进行三倍交叉验证
for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
    print(f"\n=== Fold {fold + 1} ===")
    
    # 分割数据
    X_train, X_test = X_scaled_selected[train_idx], X_scaled_selected[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # 创建并训练LDA模型 with shrinkage to prevent overfitting
    # Using shrinkage parameter to regularize LDA
    lda = LDA(n_components=1, solver='eigen', shrinkage='auto')
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    
    # 预测
    y_pred = lda.predict(X_test)
    y_pred_proba = lda.predict_proba(X_test)[:, 1]  # 响应者的概率
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 存储结果
    fold_result = {
        'fold': fold + 1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm.tolist()
    }
    fold_results.append(fold_result)
    
    # 收集所有预测结果用于最终评估
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)
    all_y_pred_proba.extend(y_pred_proba)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 score: {f1:.4f}")
    print(f"AUC value: {auc:.4f}")
    print("Confusion matrix:")
    print(cm)

# 计算平均指标
avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
avg_precision = np.mean([r['precision'] for r in fold_results])
avg_recall = np.mean([r['recall'] for r in fold_results])
avg_f1 = np.mean([r['f1'] for r in fold_results])
avg_auc = np.mean([r['auc'] for r in fold_results])

# 计算总体指标
total_accuracy = accuracy_score(all_y_true, all_y_pred)
total_precision = precision_score(all_y_true, all_y_pred)
total_recall = recall_score(all_y_true, all_y_pred)
total_f1 = f1_score(all_y_true, all_y_pred)
total_auc = roc_auc_score(all_y_true, all_y_pred_proba)
total_cm = confusion_matrix(all_y_true, all_y_pred)

# Print overall results
print("\n=== Overall Results ===")
print(f"Average accuracy: {avg_accuracy:.4f}")
print(f"Average precision: {avg_precision:.4f}")
print(f"Average recall: {avg_recall:.4f}")
print(f"Average F1 score: {avg_f1:.4f}")
print(f"Average AUC value: {avg_auc:.4f}")
print(f"\nTotal accuracy: {total_accuracy:.4f}")
print(f"Total precision: {total_precision:.4f}")
print(f"Total recall: {total_recall:.4f}")
print(f"Total F1 score: {total_f1:.4f}")
print(f"Total AUC value: {total_auc:.4f}")
print("\nTotal confusion matrix:")
print(total_cm)
print("\nClassification report:")
print(classification_report(all_y_true, all_y_pred, target_names=['Non-responder', 'Responder']))

# 保存评估指标到JSON
metrics = {
    'fold_results': fold_results,
    'average_metrics': {
        'accuracy': avg_accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'auc': avg_auc
    },
    'total_metrics': {
        'accuracy': total_accuracy,
        'precision': total_precision,
        'recall': total_recall,
        'f1': total_f1,
        'auc': total_auc,
        'confusion_matrix': total_cm.tolist()
    }
}

with open(os.path.join(original_results_dir, 'lda_classification_metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=4)

# Plot confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(total_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-responder', 'Responder'], 
            yticklabels=['Non-responder', 'Responder'])
plt.title('Confusion Matrix of LDA Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'), dpi=300)
plt.close()

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(all_y_true, all_y_pred_proba)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {total_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random guess')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve of LDA Classifier')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'roc_curve.png'), dpi=300)
plt.close()

# Plot LDA dimensionality reduction scatter plot (using selected features)
lda_all = LDA(n_components=1, solver='eigen', shrinkage='auto')
X_lda_all = lda_all.fit_transform(X_scaled_selected, y)

plt.figure(figsize=(10, 6))
plt.scatter(X_lda_all[y == 0], np.zeros_like(X_lda_all[y == 0]), 
            color='blue', alpha=0.6, label='Non-responder')
plt.scatter(X_lda_all[y == 1], np.zeros_like(X_lda_all[y == 1]) + 0.1, 
            color='red', alpha=0.6, label='Responder')
plt.xlabel('LDA Dimension')
plt.title('Sample Distribution after LDA Dimensionality Reduction')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'lda_scatter.png'), dpi=300)
plt.close()

# Get LDA component weights and analyze most important features
if hasattr(lda, 'coef_'):
    coefs = lda.coef_[0]
    # Use selected feature names instead of all features
    feature_importance = pd.DataFrame({
        'Feature': selected_feature_names,
        'Coefficient': coefs
    })
    
    # Sort by absolute value
    feature_importance['Abs_Coefficient'] = np.abs(feature_importance['Coefficient'])
    feature_importance = feature_importance.sort_values('Abs_Coefficient', ascending=False)
    
    # Save feature importance
    importance_path = os.path.join(results_dir, 'feature_importance.csv')
    feature_importance.to_csv(importance_path, index=False)
    
    # Display top 10 most important features
    print("\nTop 10 most important features:")
    for i, (idx, row) in enumerate(feature_importance.head(10).iterrows()):
        print(f"{i+1}. {row['Feature']}: {row['Coefficient']:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(10)
    plt.barh(top_features['Feature'], top_features['Coefficient'], color='skyblue')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.title('Top 10 Most Important Features in LDA Model')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'feature_importance.png'), dpi=300)
    plt.close()

# Plot performance comparison across folds
metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'auc']
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC Value']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (metric, name) in enumerate(zip(metrics_list, metrics_names)):
    fold_values = [r[metric] for r in fold_results]
    axes[i].bar(range(1, len(fold_values) + 1), fold_values, color='skyblue')
    axes[i].axhline(y=metrics['average_metrics'][metric], color='red', linestyle='--', label=f'Mean: {metrics["average_metrics"][metric]:.4f}')
    axes[i].set_title(f'{name} Across Folds')
    axes[i].set_xlabel('Fold')
    axes[i].set_ylabel(name)
    axes[i].set_xticks(range(1, len(fold_values) + 1))
    axes[i].legend()
    axes[i].grid(True, linestyle='--', alpha=0.7)

plt.delaxes(axes[5])  # Remove redundant subplot
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'fold_metrics_comparison.png'), dpi=300)
plt.close()

print("\nAll results have been saved to the visualization directory")
print(f"Metrics file: {os.path.join(original_results_dir, 'lda_classification_metrics.json')}")
print(f"Confusion matrix plot: {os.path.join(results_dir, 'confusion_matrix.png')}")
print(f"ROC curve plot: {os.path.join(results_dir, 'roc_curve.png')}")
print(f"LDA scatter plot: {os.path.join(results_dir, 'lda_scatter.png')}")
print(f"Fold metrics comparison plot: {os.path.join(results_dir, 'fold_metrics_comparison.png')}")
print(f"Feature importance plot: {os.path.join(results_dir, 'feature_importance.png')}")

# Add train-test split performance comparison to detect overfitting
print("\nChecking for overfitting...")
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_scaled_selected, y, test_size=0.3, random_state=42
)

# Train model on full training set
lda_full = LDA(n_components=1, solver='eigen', shrinkage='auto')
lda_full.fit(X_train_full, y_train_full)

# Evaluate on training and test sets
train_pred = lda_full.predict(X_train_full)
train_acc = accuracy_score(y_train_full, train_pred)
test_pred = lda_full.predict(X_test_full)
test_acc = accuracy_score(y_test_full, test_pred)

train_auc = roc_auc_score(y_train_full, lda_full.predict_proba(X_train_full)[:, 1])
test_auc = roc_auc_score(y_test_full, lda_full.predict_proba(X_test_full)[:, 1])

print(f"Training set accuracy: {train_acc:.4f}")
print(f"Test set accuracy: {test_acc:.4f}")
print(f"Training set AUC: {train_auc:.4f}")
print(f"Test set AUC: {test_auc:.4f}")

# Calculate the difference to detect overfitting
acc_diff = train_acc - test_acc
auc_diff = train_auc - test_auc

print(f"\nAccuracy difference (train-test): {acc_diff:.4f}")
print(f"AUC difference (train-test): {auc_diff:.4f}")

if acc_diff > 0.1 or auc_diff > 0.1:
    print("Warning: Possible overfitting detected!")
else:
    print("No significant overfitting detected.")