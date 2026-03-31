import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 加载数据集
class PeptideDataset(Dataset):
    def __init__(self, X_file, y_file):
        self.X = np.load(X_file)
        self.y = np.load(y_file)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# 模型和超参数
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
learning_rate = 0.001
num_epochs = 1000
batch_size = 1024
patience = 30  # Early Stopping的等待epochs数
weight_decay = 0.0002  # AdamW的权重衰减系数
momentum = 0.9  # SGD的动量参数
lr_step_size = 50  # 每隔50个epoch衰减一次学习率
lr_gamma = 0.95  # 学习率衰减因子

# 加载模型
model = PeptideClassifier(input_size, hidden_size, num_layers, num_classes)
model.load_state_dict(torch.load('best_model.pth'))
model.train()  # 设置模型为评估模式

# Integrated Gradients 可解释性分析
def analyze_with_ig(model, dataloader, device):
    model.to(device)
    ig = IntegratedGradients(model)
    
    all_attributions = []
    
    for batch_X, batch_y in tqdm(dataloader, desc='Analyzing'):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Integrated Gradients需要输入和目标类
        attributions, delta = ig.attribute(batch_X, target=batch_y, return_convergence_delta=True)
        all_attributions.append(attributions.cpu().detach().numpy())

    return np.concatenate(all_attributions, axis=0)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 创建测试数据加载器
test_dataset = PeptideDataset('X_test_balanced.npy', 'y_test_balanced.npy')
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 在测试集上进行Integrated Gradients分析
attributions = analyze_with_ig(model, test_loader, device)

# 输出结果，例如保存为文件或可视化
np.save('integrated_gradients_attributions.npy', attributions)
print("Integrated Gradients analysis completed and saved.")

# 分析完成后，将模型设置回评估模式
model.eval()