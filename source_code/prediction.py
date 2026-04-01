import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd

# 定义氨基酸编码字典
amino_acid_dict = {
    'A': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'C': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'D': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'E': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'F': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'G': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'H': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'I': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'K': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'L': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'M': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'N': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'P': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'Q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    'R': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    'S': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    'T': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    'V': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    'W': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    'Y': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
}

# 反转氨基酸编码字典，用于从二进制向量转换到氨基酸
binary_to_amino_acid = {tuple(v): k for k, v in amino_acid_dict.items()}

# 肽段编码为二进制向量的函数
def peptide_to_binaryvector(peptide):
    binary_vector = []
    for amino_acid in peptide:
        if amino_acid in amino_acid_dict:
            binary_vector.extend(amino_acid_dict[amino_acid])
        else:
            # 处理非标准氨基酸，这里简单填充全零向量
            binary_vector.extend([0] * 20)
    return binary_vector

# 从二进制向量转换到氨基酸的函数
def binaryvectortoaminoacid(binary_vectors):
    # 将每个二进制向量转换为氨基酸
    amino_acid_sequence = ''.join(binary_to_amino_acid.get(tuple(vector), ' ') for vector in binary_vectors)
    return amino_acid_sequence

# 模型定义
class PeptideClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(PeptideClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

# 设置超参数
input_size = 20
hidden_size = 128
num_layers = 2
num_classes = 2
batch_size = 1024

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 初始化模型
model = PeptideClassifier(input_size, hidden_size, num_layers, num_classes).to(device)

# 加载最佳模型
model.load_state_dict(torch.load('advanced best_model.pth', weights_only=True))
model.eval()

# 读取包含肽段的文件（假设是一个文本文件，每行一个肽段）
def predict_peptides(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        peptides = [line.strip() for line in f.readlines()]

    # 将肽段编码为二进制向量
    binary_vectors = [peptide_to_binaryvector(peptide) for peptide in peptides]

    # 找到最长的二进制向量长度
    max_length = max(len(vec) for vec in binary_vectors)

    # 填充二进制向量，使它们的长度一致
    padded_vectors = []
    for vec in binary_vectors:
        padding = [0] * (max_length - len(vec))
        padded_vectors.append(vec + padding)

    binary_vectors = np.array(padded_vectors)
    binary_vectors = binary_vectors.reshape(-1, max_length // input_size, input_size)

    # 创建数据集和数据加载器
    class PredictionDataset(Dataset):
        def __init__(self, X):
            self.X = X

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return torch.tensor(self.X[idx], dtype=torch.float32)

    prediction_dataset = PredictionDataset(binary_vectors)
    prediction_loader = DataLoader(prediction_dataset, batch_size=batch_size, pin_memory=True)

    # 进行预测
    predictions = []
    with torch.no_grad():
        for batch_X in tqdm(prediction_loader, desc='Predicting'):
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probabilities = torch.softmax(outputs, dim=1)  # 将 logits 转化为概率
            # 提取代表类别1的概率
            predictions.extend(probabilities[:, 1].cpu().numpy())

    # 创建DataFrame
    df_results = pd.DataFrame({
        'Peptide': peptides,
        'Predicted': predictions
    })

    # 保存结果到Excel文件
    df_results.to_excel('predictions.xlsx', index=False)
    print("预测结果已保存到 predictions1.xlsx")

# 调用预测函数，替换为你的包含肽段的文件路径
predict_peptides('unqiue peptide.txt')