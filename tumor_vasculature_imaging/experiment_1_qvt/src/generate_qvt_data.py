import numpy as np
import pandas as pd
import os
import argparse

# 创建数据目录（如果不存在）
data_dir = os.path.join('..', 'data')
os.makedirs(data_dir, exist_ok=True)

# 解析命令行参数
parser = argparse.ArgumentParser(description='生成QVT特征数据')
parser.add_argument('--force-regenerate', action='store_true', help='强制重新生成数据，即使文件已存在')
args = parser.parse_args()

# 检查文件是否已存在且不需要强制重新生成
output_file = os.path.join(data_dir, 'qvt_features_data.csv')
if os.path.exists(output_file) and not args.force_regenerate:
    print(f"数据文件已存在: {output_file}")
    print("直接使用现有数据，跳过重新生成。使用 --force-regenerate 参数可强制重新生成。")
    # 读取并显示现有数据的统计信息
    df = pd.read_csv(output_file)
    print(f"数据集形状: {df.shape}")
    print(f"响应者数量: {np.sum(df['response'] == 1)}")
    print(f"非响应者数量: {np.sum(df['response'] == 0)}")
    exit(0)

# 设置随机种子以确保可重复性
np.random.seed(42)

# 样本数量
n_samples = 162  # 论文中使用的总样本数
n_responders = 62  # 训练集大小，与论文一致
n_nonresponders = n_samples - n_responders

# 创建特征列表 - 使用更具描述性的特征名称
feature_names = []

# 1. 血管曲率统计特征
curvature_stats = [
    'curvature_mean', 'curvature_std', 'curvature_min', 'curvature_max', 'curvature_median',
    'curvature_skewness', 'curvature_kurtosis', 'curvature_q1', 'curvature_q3', 'curvature_range'
]
feature_names.extend(curvature_stats)

# 2. 血管曲折度统计特征
tortuosity_stats = [
    'tortuosity_mean', 'tortuosity_std', 'tortuosity_min', 'tortuosity_max', 'tortuosity_median',
    'tortuosity_skewness', 'tortuosity_kurtosis', 'tortuosity_q1', 'tortuosity_q3', 'tortuosity_range'
]
feature_names.extend(tortuosity_stats)

# 3. 血管分支统计特征
branch_stats = [
    'branch_count', 'branch_length_mean', 'branch_length_std', 'branch_density',
    'branch_angle_mean', 'branch_angle_std', 'branch_junction_count'
]
feature_names.extend(branch_stats)

# 4. 血管角度分布特征（15个bin）
angle_bins = [
    'angle_0-12', 'angle_12-24', 'angle_24-36', 'angle_36-48', 'angle_48-60',
    'angle_60-72', 'angle_72-84', 'angle_84-96', 'angle_96-108', 'angle_108-120',
    'angle_120-132', 'angle_132-144', 'angle_144-156', 'angle_156-168', 'angle_168-180'
]
feature_names.extend(angle_bins)

# 5. 血管体积特征
feature_names.append('vessel_volume')

# 6. 锐角分布特征
feature_names.append('acute_angle_ratio')

# 7. 钝角分布特征
feature_names.append('obtuse_angle_ratio')

# 8. 其他血管统计特征
other_stats = [
    'vessel_diameter_mean', 'vessel_diameter_std', 'vessel_length_mean', 'vessel_length_std',
    'vessel_density', 'vessel_perimeter', 'vessel_surface_area', 'vessel_complexity',
    'vessel_fractal_dimension', 'vessel_spatial_dispersion', 'vessel_branching_factor',
    'vessel_curvature_energy', 'vessel_tortuosity_index', 'vessel_continuity',
    'vessel_segment_length_mean', 'vessel_segment_length_std', 'vessel_branch_angle_min',
    'vessel_branch_angle_max', 'vessel_cross_section_area', 'vessel_conductance',
    'vessel_resistance', 'vessel_flow_velocity'
]
feature_names.extend(other_stats)

# 总特征数应该与论文中提到的74个特征相近
print(f"生成的特征数量: {len(feature_names)}")

# 为响应者生成数据（根据论文，响应者有较少的曲折血管，钝角分布几乎是非响应者的两倍）
responders_data = np.zeros((n_responders, len(feature_names)))

# 曲率特征：响应者血管较平滑，曲率较低 - 增加标准差使分布更分散
for i in range(10):
    responders_data[:, i] = np.random.normal(loc=3.0, scale=1.5, size=n_responders)

# 曲折度特征：响应者血管较直，曲折度较低 - 增加标准差
for i in range(10, 20):
    responders_data[:, i] = np.random.normal(loc=1.5, scale=0.8, size=n_responders)

# 分支统计特征 - 增加标准差
for i in range(20, 27):
    responders_data[:, i] = np.random.normal(loc=3.2, scale=1.8, size=n_responders)

# 角度分布特征：响应者有更多的钝角（bin 1-7表示钝角）- 增加标准差，减小差异
for i in range(27, 34):  # 钝角bin (1-7)
    responders_data[:, i] = np.random.normal(loc=7.0, scale=3.0, size=n_responders)
for i in range(34, 42):  # 中等角度bin (8-15)
    responders_data[:, i] = np.random.normal(loc=5.5, scale=2.5, size=n_responders)

# 血管体积 - 增加标准差
responders_data[:, 42] = np.random.normal(loc=105.0, scale=35.0, size=n_responders)

# 锐角比例较低 - 增加标准差
responders_data[:, 43] = np.random.normal(loc=0.35, scale=0.2, size=n_responders)

# 钝角比例较高 - 增加标准差
responders_data[:, 44] = np.random.normal(loc=0.55, scale=0.2, size=n_responders)

# 其他统计特征 - 增加标准差，使分布更分散
for i in range(45, len(feature_names)):
    responders_data[:, i] = np.random.normal(loc=5.5, scale=3.0, size=n_responders)

# 为非响应者生成数据（血管更曲折，锐角更多）
nonresponders_data = np.zeros((n_nonresponders, len(feature_names)))

# 曲率特征：非响应者血管更曲折，曲率较高 - 减小均值差异，增加标准差
for i in range(10):
    nonresponders_data[:, i] = np.random.normal(loc=3.8, scale=1.8, size=n_nonresponders)

# 曲折度特征：非响应者血管更曲折 - 减小均值差异，增加标准差
for i in range(10, 20):
    nonresponders_data[:, i] = np.random.normal(loc=2.0, scale=1.0, size=n_nonresponders)

# 分支统计特征 - 减小均值差异，增加标准差
for i in range(20, 27):
    nonresponders_data[:, i] = np.random.normal(loc=3.8, scale=2.0, size=n_nonresponders)

# 角度分布特征：非响应者有更多的锐角（bin 8-15表示锐角）- 减小均值差异，增加标准差
for i in range(27, 34):  # 钝角bin (1-7)
    nonresponders_data[:, i] = np.random.normal(loc=5.0, scale=3.0, size=n_nonresponders)
for i in range(34, 42):  # 中等角度bin (8-15)
    nonresponders_data[:, i] = np.random.normal(loc=6.5, scale=3.0, size=n_nonresponders)

# 血管体积 - 减小均值差异，增加标准差
nonresponders_data[:, 42] = np.random.normal(loc=115.0, scale=40.0, size=n_nonresponders)

# 锐角比例较高 - 减小均值差异，增加标准差
nonresponders_data[:, 43] = np.random.normal(loc=0.55, scale=0.2, size=n_nonresponders)

# 钝角比例较低 - 减小均值差异，增加标准差
nonresponders_data[:, 44] = np.random.normal(loc=0.35, scale=0.2, size=n_nonresponders)

# 其他统计特征 - 减小均值差异，增加标准差
for i in range(45, len(feature_names)):
    nonresponders_data[:, i] = np.random.normal(loc=6.5, scale=3.5, size=n_nonresponders)

# 添加一些特征噪声和相关性，使数据更真实
# 1. 添加随机噪声
noise_factor = 0.2
responders_noise = np.random.normal(0, noise_factor, responders_data.shape)
nonresponders_noise = np.random.normal(0, noise_factor, nonresponders_data.shape)
responders_data = responders_data + responders_noise
nonresponders_data = nonresponders_data + nonresponders_noise

# 2. 添加一些特征间的相关性（让某些特征不完全独立）
# 例如，血管密度与分支数相关
correlation_strength = 0.4
# 为响应者添加相关性
for i in range(n_responders):
    if 'vessel_density' in feature_names and 'branch_count' in feature_names:
        density_idx = feature_names.index('vessel_density')
        branch_idx = feature_names.index('branch_count')
        responders_data[i, density_idx] = responders_data[i, density_idx] * (1 - correlation_strength) + \
                                         responders_data[i, branch_idx] * correlation_strength

# 为非响应者添加相关性
for i in range(n_nonresponders):
    if 'vessel_density' in feature_names and 'branch_count' in feature_names:
        density_idx = feature_names.index('vessel_density')
        branch_idx = feature_names.index('branch_count')
        nonresponders_data[i, density_idx] = nonresponders_data[i, density_idx] * (1 - correlation_strength) + \
                                            nonresponders_data[i, branch_idx] * correlation_strength

# 合并数据
X = np.vstack((responders_data, nonresponders_data))

# 创建标签：1表示响应者，0表示非响应者
y = np.array([1] * n_responders + [0] * n_nonresponders)

# 打乱数据顺序
indices = np.arange(n_samples)
np.random.shuffle(indices)
X = X[indices]
y = y[indices]

# 创建DataFrame
columns = feature_names + ['response']
data = np.hstack((X, y.reshape(-1, 1)))
df = pd.DataFrame(data, columns=columns)

# 保存到CSV文件
output_file = os.path.join(data_dir, 'qvt_features_data.csv')
df.to_csv(output_file, index=False)

# 显示数据统计信息
print(f"数据集已保存到: {output_file}")
print(f"数据集形状: {df.shape}")
print(f"响应者数量: {np.sum(df['response'] == 1)}")
print(f"非响应者数量: {np.sum(df['response'] == 0)}")

# 显示一些关键特征的统计信息
print("\n关键特征统计信息:")
print("响应者样本:")
print(f"平均曲率统计: {df[df['response'] == 1][feature_names[:10]].mean().mean():.4f}")
print(f"平均曲折度统计: {df[df['response'] == 1][feature_names[10:20]].mean().mean():.4f}")
print(f"平均钝角比例: {df[df['response'] == 1]['obtuse_angle_ratio'].mean():.4f}")
print(f"平均锐角比例: {df[df['response'] == 1]['acute_angle_ratio'].mean():.4f}")

print("\n非响应者样本:")
print(f"平均曲率统计: {df[df['response'] == 0][feature_names[:10]].mean().mean():.4f}")
print(f"平均曲折度统计: {df[df['response'] == 0][feature_names[10:20]].mean().mean():.4f}")
print(f"平均钝角比例: {df[df['response'] == 0]['obtuse_angle_ratio'].mean():.4f}")
print(f"平均锐角比例: {df[df['response'] == 0]['acute_angle_ratio'].mean():.4f}")