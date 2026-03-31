import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# 读取CSV文件
data = pd.read_csv('result.csv')

# 将数据集分为特征和标签
X = data['X'].values
y = data['y'].values

# 将数据集以7:3的比例分为训练集和测试集
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# 将测试集以2:1的比例分为验证集和测试集
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42, shuffle=True)

# 对数据进行预处理
amino_acids = sorted(list(set(''.join(X))))
print(f"Amino acids in dataset: {amino_acids}")  # 输出氨基酸种类
encoder = OneHotEncoder(categories=[amino_acids], handle_unknown='ignore')
encoder.fit(np.array(amino_acids).reshape(-1, 1))

# 输出氨基酸及其编码顺序
print("Amino acid encoding map:")
for i, aa in enumerate(encoder.categories_[0]):
    print(f"Amino acid: {aa} -> Encoding index: {i}")

max_length = max(len(seq) for seq in X)

def preprocess_data(X):
    X_encoded = []
    for seq in X:
        encoded_seq = np.zeros((max_length, len(amino_acids)))
        for i, amino_acid in enumerate(seq):
            encoded_seq[i] = encoder.transform([[amino_acid]]).toarray()
        X_encoded.append(encoded_seq)
    X_encoded = np.array(X_encoded)
    return X_encoded

X_train_encoded = preprocess_data(X_train)
X_val_encoded = preprocess_data(X_val)
X_test_encoded = preprocess_data(X_test)

# 找出训练集中数量较少的类别，并计算差异
train_class_0_indices = np.where(y_train == 0)[0]
train_class_1_indices = np.where(y_train == 1)[0]

# 平衡训练集
train_minority_class_indices = train_class_0_indices if len(train_class_0_indices) < len(train_class_1_indices) else train_class_1_indices
train_majority_class_indices = train_class_1_indices if len(train_class_0_indices) < len(train_class_1_indices) else train_class_0_indices
train_num_to_remove = len(train_majority_class_indices) - len(train_minority_class_indices)

# 随机选择训练集多数类别中要删除的样本
train_indices_to_remove = np.random.choice(train_majority_class_indices, train_num_to_remove, replace=False)

# 删除选中的样本
X_train_balanced = np.delete(X_train_encoded, train_indices_to_remove, axis=0)
y_train_balanced = np.delete(y_train, train_indices_to_remove)

# 验证集保持原始状态，不进行平衡
X_val_balanced = X_val_encoded
y_val_balanced = y_val

# 现在X_train_balanced和y_train_balanced是平衡的，而X_val_balanced和y_val_balanced保持原始状态
print(f"平衡后的训练集样本数: {len(X_train_balanced)}")
print(f"原始验证集样本数: {len(X_val_balanced)}")

# 将预处理后的数据保存到文件中
np.save('X_train_balanced.npy', X_train_balanced)
np.save('y_train_balanced.npy', y_train_balanced)
np.save('X_val_balanced.npy', X_val_balanced)
np.save('y_val_balanced.npy', y_val_balanced)

# 假设y_test中0和1代表两个类别
class_0_indices = np.where(y_test == 0)[0]
class_1_indices = np.where(y_test == 1)[0]

# 找出数量较少的类别，并计算差异
minority_class_indices = class_0_indices if len(class_0_indices) < len(class_1_indices) else class_1_indices
majority_class_indices = class_1_indices if len(class_0_indices) < len(class_1_indices) else class_0_indices
num_to_remove = len(majority_class_indices) - len(minority_class_indices)

# 随机选择多数类别中要删除的样本
np.random.seed(42)  # 确保可重复性
indices_to_remove = np.random.choice(majority_class_indices, num_to_remove, replace=False)

# 删除选中的样本
X_test_balanced = np.delete(X_test_encoded, indices_to_remove, axis=0)
y_test_balanced = np.delete(y_test, indices_to_remove)

# 现在X_test_balanced和y_test_balanced是平衡的
print(f"平衡后的测试集样本数: {len(X_test_balanced)}")

np.save('X_test_balanced.npy', X_test_balanced)
np.save('y_test_balanced.npy', y_test_balanced)
