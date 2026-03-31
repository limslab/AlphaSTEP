import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 加载保存的Integrated Gradients归因分数
attributions = np.load('integrated_gradients_attributions.npy')

# 加载测试集数据
X_test = np.load('X_test_encoded.npy')
y_test = np.load('y_test.npy')

# 加载原始氨基酸序列
data = pd.read_csv('result.csv')
X_original = data['X'].values

# 氨基酸编码映射
amino_acids = sorted(list(set(''.join(X_original))))
aa_to_index = {aa: i for i, aa in enumerate(amino_acids)}
index_to_aa = {i: aa for aa, i in aa_to_index.items()}

# 计算每个类的平均归因分数
def mean_attributions_by_class(attributions, y_test):
    """
    计算每个类别的平均归因分数
    :param attributions: Integrated Gradients生成的归因分数
    :param y_test: 对应的标签
    :return: 每个类别的平均归因分数
    """
    unique_classes = np.unique(y_test)
    class_mean_attributions = {}
    
    for cls in unique_classes:
        class_indices = np.where(y_test == cls)[0]
        class_attributions = attributions[class_indices]
        class_mean_attributions[cls] = class_attributions.mean(axis=0)
    
    return class_mean_attributions

# 计算并可视化类0和类1的平均归因分数
mean_attributions = mean_attributions_by_class(attributions, y_test)

# 可视化类0的平均归因分数
plt.figure(figsize=(10, 6))
plt.title(f"Mean Attributions for Class 0")
plt.bar(range(mean_attributions[0].shape[1]), mean_attributions[0].sum(axis=0))
plt.xlabel("Feature Index")
plt.ylabel("Mean Attribution Score")
plt.show()

# 可视化类1的平均归因分数
plt.figure(figsize=(10, 6))
plt.title(f"Mean Attributions for Class 1")
plt.bar(range(mean_attributions[1].shape[1]), mean_attributions[1].sum(axis=0))
plt.xlabel("Feature Index")
plt.ylabel("Mean Attribution Score")
plt.show()

def decode_sequence(encoded_seq):
    """
    将编码后的序列解码为氨基酸序列
    """
    return ''.join([index_to_aa[np.argmax(pos)] for pos in encoded_seq if np.sum(pos) != 0])

def top_features(attributions, X_test, top_k=5):
    """
    找到归因分数最高的前k个氨基酸类型
    """
    summed_attributions = np.sum(attributions, axis=1)
    
    top_features_aa = []
    for i, attr in enumerate(summed_attributions):
        decoded_seq = decode_sequence(X_test[i])
        aa_importance = {aa: 0 for aa in decoded_seq}  # 只考虑序列中存在的氨基酸
        for j, importance in enumerate(attr):
            if j < len(decoded_seq):
                aa = decoded_seq[j]
                aa_importance[aa] += importance
        
        sorted_aa = sorted(aa_importance.items(), key=lambda x: x[1], reverse=True)
        top_features_aa.append([aa for aa, _ in sorted_aa[:top_k]])
    
    return top_features_aa

# 获取类0和类1样本的top 5特征
class_0_indices = np.where(y_test == 0)[0]
class_1_indices = np.where(y_test == 1)[0]

top_k_aa_class_0 = top_features(attributions[class_0_indices], X_test[class_0_indices], top_k=5)
top_k_aa_class_1 = top_features(attributions[class_1_indices], X_test[class_1_indices], top_k=5)

# 可视化每个输入特征的归因分数
def visualize_attributions_by_class(attributions, X_test, y_test, num_samples=5):
    """
    可视化类0和类1样本的Integrated Gradients归因分数
    :param attributions: Integrated Gradients生成的归因分数
    :param X_test: 原始输入数据
    :param y_test: 对应的标签
    :param num_samples: 可视化的样本数量
    """
    sns.set(style="whitegrid")
    
    # 选择类0和类1的样本索引
    class_0_indices = np.where(y_test == 0)[0]
    class_1_indices = np.where(y_test == 1)[0]
    
    print("Visualizing Class 0 samples:")
    for i in range(min(num_samples, len(class_0_indices))):  # 确保不超出样本数量
        idx = class_0_indices[i]
        plt.figure(figsize=(10, 6))
        plt.title(f"Sample {idx+1} (Class 0)")
        plt.bar(range(attributions.shape[2]), attributions[idx].sum(axis=0))
        plt.xlabel("Feature Index")
        plt.ylabel("Attribution Score")
        plt.show()
    
    print("Visualizing Class 1 samples:")
    for i in range(min(num_samples, len(class_1_indices))):  # 确保不超出样本数量
        idx = class_1_indices[i]
        plt.figure(figsize=(10, 6))
        plt.title(f"Sample {idx+1} (Class 1)")
        plt.bar(range(attributions.shape[2]), attributions[idx].sum(axis=0))
        plt.xlabel("Feature Index")
        plt.ylabel("Attribution Score")
        plt.show()

# 可视化前5个属于类0和类1的样本的归因分数
visualize_attributions_by_class(attributions, X_test, y_test, num_samples=5)

# 输出前5个类0样本的top 5氨基酸类型
print("Class 0 samples - Top 5 Amino Acid Types:")
for i in range(min(5, len(class_0_indices))):
    idx = class_0_indices[i]
    original_seq = decode_sequence(X_test[idx])  # 使用解码后的序列
    print(f"Sample {idx+1} ({original_seq}) (Class 0) - Top 5 Amino Acid Types: {top_k_aa_class_0[i]}")

# 输出前5个类1样本的top 5氨基酸类型
print("\nClass 1 samples - Top 5 Amino Acid Types:")
for i in range(min(5, len(class_1_indices))):
    idx = class_1_indices[i]
    original_seq = decode_sequence(X_test[idx])  # 使用解码后的序列
    print(f"Sample {idx+1} ({original_seq}) (Class 1) - Top 5 Amino Acid Types: {top_k_aa_class_1[i]}")


def position_importance(attributions, y_test):
    """
    计算每个类别中序列位点的重要性（仅与位置相关）
    :param attributions: Integrated Gradients生成的归因分数
    :param y_test: 对应的标签
    :return: 每个类别中每个位点的重要性得分
    """
    unique_classes = np.unique(y_test)
    position_scores = {}
    
    for cls in unique_classes:
        # 获取当前类别的样本
        class_indices = np.where(y_test == cls)[0]
        class_attributions = attributions[class_indices]
        
        # 对每个位点的所有维度求和，再对样本求平均，得到该类别下位点的重要性
        position_scores[cls] = np.mean(np.sum(class_attributions, axis=2), axis=0)
    
    return position_scores

# 可视化位点重要性的函数
def visualize_position_importance(position_scores, max_positions=30):
    """
    可视化每个位点的重要性得分
    :param position_scores: 位点重要性得分字典
    :param max_positions: 显示的最大位点数量
    """
    plt.figure(figsize=(15, 6))
    
    # 为每个类别创建位点重要性图
    for cls in position_scores.keys():
        scores = position_scores[cls][:max_positions]  # 限制显示的位点数量
        plt.subplot(1, len(position_scores), cls + 1)
        plt.bar(range(1, len(scores) + 1), scores)
        plt.title(f'Position Importance - Class {cls}')
        plt.xlabel('Sequence Position')
        plt.ylabel('Importance Score')
        # 添加位点标签
        plt.xticks(range(1, len(scores) + 1))
        
        # 标注重要性最高的几个位点
        top_positions = np.argsort(scores)[-3:]  # 获取前3个最重要的位点
        for pos in top_positions:
            plt.text(pos + 1, scores[pos], f'Pos {pos + 1}', 
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# 输出每个类别最重要的位点
def print_top_positions(position_scores, top_k=5):
    """
    打印每个类别最重要的位点
    """
    for cls in position_scores.keys():
        scores = position_scores[cls]
        top_indices = np.argsort(scores)[-top_k:][::-1]
        print(f"\nClass {cls} - Top {top_k} important positions:")
        for i, idx in enumerate(top_indices, 1):
            print(f"Rank {i}: Position {idx + 1} (Score: {scores[idx]:.4f})")


def visualize_sample_position_importance(attributions, X_test, y_test, num_samples=3):
    """
    可视化选定样本的位点重要性
    :param attributions: Integrated Gradients归因分数
    :param X_test: 测试数据
    :param y_test: 测试标签
    :param num_samples: 每个类别要显示的样本数量
    """
    # 获取每个类别的样本索引
    class_0_indices = np.where(y_test == 0)[0]
    class_1_indices = np.where(y_test == 1)[0]
    
    # 为每个类别选择代表性样本
    for class_label, indices in [(0, class_0_indices), (1, class_1_indices)]:
        print(f"\n=== Class {class_label} Samples ===")
        
        # 选择前num_samples个样本
        for i in range(min(num_samples, len(indices))):
            idx = indices[i]
            sample_attr = attributions[idx]
            
            # 计算每个位点的总重要性（所有维度求和）
            position_importance = np.sum(sample_attr, axis=1)
            
            # 获取原始序列
            sequence = decode_sequence(X_test[idx])
            
            # 创建可视化
            plt.figure(figsize=(10, 6))
            plt.bar(range(1, len(position_importance) + 1), position_importance)
            plt.title(f'Position Importance - Class {class_label}, Sample {i+1}\nSequence: {sequence}')
            plt.xlabel('Sequence Position')
            plt.ylabel('Importance Score')
            
            # 标注重要性最高的几个位点
            top_positions = np.argsort(position_importance)[-3:]  # 获取前3个最重要的位点
            for pos in top_positions:
                plt.text(pos + 1, position_importance[pos], 
                         f'Pos {pos+1}\n{sequence[pos]}', 
                         ha='center', va='bottom', color='red')
            
            plt.tight_layout()
            plt.show()
            
            # 打印该样本的重要位点信息
            top_k = 5
            top_indices = np.argsort(position_importance)[-top_k:][::-1]
            print(f"\nSample {i+1} (Class {class_label}) - Sequence: {sequence}")
            print(f"Top {top_k} important positions:")
            for rank, idx in enumerate(top_indices, 1):
                print(f"Rank {rank}: Position {idx + 1} (AA: {sequence[idx]}, Score: {position_importance[idx]:.4f})")

position_scores_correct = position_importance(attributions, y_test)
visualize_position_importance(position_scores_correct)
print_top_positions(position_scores_correct)
visualize_sample_position_importance(attributions, X_test, y_test, num_samples=3)