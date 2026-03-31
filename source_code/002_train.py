import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class PeptideDataset(Dataset):
    def __init__(self, X_file, y_file):
        self.X = np.load(X_file)
        self.y = np.load(y_file)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)
    
# 计算类权重
def compute_class_weights(y_file):
    y = np.load(y_file)
    class_sample_count = np.bincount(y)
    weight = 1.0 / class_sample_count
    normalized_weight = weight / weight.sum()
    return torch.tensor(normalized_weight, dtype=torch.float32)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 创建数据加载器
train_dataset = PeptideDataset('X_train_balanced.npy', 'y_train_balanced.npy')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

val_dataset = PeptideDataset('X_val_balanced.npy', 'y_val_balanced.npy')
val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=True)

test_dataset = PeptideDataset('X_test_balanced.npy', 'y_test_balanced.npy')
test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True)

# 计算训练集的类权重
class_weights = compute_class_weights('y_train_balanced.npy').to(device)
print(f"The weights of different classes are {class_weights}")

# 初始化模型、损失函数和优化器
model = PeptideClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
#criterion = nn.CrossEntropyLoss()
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# 设置学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

# Early Stopping
best_val_loss = float('inf')
counter = 0
best_model = None

# 训练模型
for epoch in tqdm(range(num_epochs), desc='Training'):
    model.train()
    for batch_X, batch_y in tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # 更新学习率
    scheduler.step()

    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_weighted_correct = 0
        val_weighted_total = 0
        total_samples = len(val_dataset)
        for batch_X, batch_y in tqdm(val_loader, desc='Validation', leave=False):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)  # 使用权重计算损失
            val_loss += loss.item() * len(batch_y)  # 权重化的损失
            _, predicted = torch.max(outputs.data, 1)

            for i in range(num_classes):
                class_indices = (batch_y == i)
                val_weighted_correct += (predicted[class_indices] == batch_y[class_indices]).sum().item() * class_weights[i]
                val_weighted_total += len(predicted[class_indices]) * class_weights[i]

        val_loss /= total_samples
        val_acc = val_weighted_correct / val_weighted_total
        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model.state_dict()
        torch.save(best_model, 'best_model.pth')  # 将最佳模型保存为文件
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early Stopping at epoch {epoch+1}")
            break

    # 每10个epoch保存一次模型
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# 在测试集上评估模型
model.eval()
with torch.no_grad():
    test_correct = 0
    for batch_X, batch_y in tqdm(test_loader, desc='Testing'):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        _, predicted = torch.max(outputs.data, 1)
        test_correct += (predicted == batch_y).sum().item()
    test_acc = test_correct / len(test_dataset)
    print(f"Test Acc: {test_acc:.4f}")