import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# 训练模型
def train_epoch(loader, model, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += inputs.size(0)

    return total_loss / total_samples, total_correct / total_samples


# 验证模型
def validate_epoch(loader, model, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += inputs.size(0)

    return total_loss / total_samples, total_correct / total_samples

def load_glove_model(glove_file):
    model = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            split_line = line.split()
            word = " ".join(split_line[:-300])
            embedding = np.array([float(val) for val in split_line[-300:]])
            model[word] = embedding
    return model

def get_text_vector(text, glove_model, max_len):
    words = text.split()
    vectors = [glove_model[word] for word in words if word in glove_model]
    # Padding or truncation
    if len(vectors) < max_len:
        vectors += [np.zeros(300)] * (max_len - len(vectors))  # Zero-padding
    else:
        vectors = vectors[:max_len]  # Truncation
    return np.array(vectors)


# 读取数据
train_data = pd.read_csv('train.tsv', delimiter='\t', header=None)
test_data = pd.read_csv('test.tsv', delimiter='\t', header=None)

train_arg1, train_arg2, train_label = train_data.iloc[:, 2], train_data.iloc[:, 3], train_data.iloc[:, 1]
test_arg1, test_arg2 = test_data.iloc[:, 2], test_data.iloc[:, 3]

# 对类别标签编码
class_dict = {'Comparison': 0, 'Contingency': 1, 'Expansion': 2, 'Temporal': 3}




def save_data(filename, data):
    np.save(filename, data)


def load_data(filename):
    return np.load(filename)


# 检查数据是否已被保存
train_arg1_file = 'train_arg1_vector.npy'
train_arg2_file = 'train_arg2_vector.npy'
test_arg1_file = 'test_arg1_vector.npy'
test_arg2_file = 'test_arg2_vector.npy'
train_label_file = 'train_label.npy'

if not (os.path.exists(train_arg1_file) and os.path.exists(train_arg2_file) and
        os.path.exists(test_arg1_file) and os.path.exists(test_arg2_file) and
        os.path.exists(train_label_file)):
    # 如果文件不存在，进行GloVe向量化并保存结果
    print('File not here')
    # 加载GloVe模型
    glove_model = load_glove_model('glove.840B.300d.txt')
    max_len = 100
    train_arg1_vector = np.array([get_text_vector(text, glove_model, max_len) for text in train_arg1])
    train_arg2_vector = np.array([get_text_vector(text, glove_model, max_len) for text in train_arg2])
    test_arg1_vector = np.array([get_text_vector(text, glove_model, max_len) for text in test_arg1])
    test_arg2_vector = np.array([get_text_vector(text, glove_model, max_len) for text in test_arg2])
    train_label = np.array([class_dict[label] for label in train_label])

    save_data(train_arg1_file, train_arg1_vector)
    save_data(train_arg2_file, train_arg2_vector)
    save_data(test_arg1_file, test_arg1_vector)
    save_data(test_arg2_file, test_arg2_vector)
    save_data(train_label_file, train_label)
else:
    # 如果文件存在，直接加载数据
    train_arg1_vector = load_data(train_arg1_file)
    train_arg2_vector = load_data(train_arg2_file)
    test_arg1_vector = load_data(test_arg1_file)
    test_arg2_vector = load_data(test_arg2_file)
    train_label = load_data(train_label_file)


# 转换为Tensor
train_arg1_tensor = torch.FloatTensor(train_arg1_vector)
train_arg2_tensor = torch.FloatTensor(train_arg2_vector)
test_arg1_tensor = torch.FloatTensor(test_arg1_vector)
test_arg2_tensor = torch.FloatTensor(test_arg2_vector)

train_label_tensor = torch.LongTensor(train_label)

train_tensor = torch.cat((train_arg1_tensor, train_arg2_tensor), dim=1)
test_tensor = torch.cat((test_arg1_tensor, test_arg2_tensor), dim=1)

# 划分训练集和验证集
train_tensor, val_tensor, train_label_tensor, val_label_tensor = train_test_split(train_tensor, train_label_tensor,
                                                                                  test_size=0.2, random_state=42)

# 创建数据加载器
train_loader = DataLoader(TensorDataset(train_tensor, train_label_tensor), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(val_tensor, val_label_tensor), batch_size=64, shuffle=False)


# 定义模型
class LSTM_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(LSTM_MLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# LSTM 参数
input_dim = 300  # GloVe 向量维度
hidden_dim = 128  # LSTM 隐藏层维度
num_layers = 2  # LSTM 层数

# 创建模型实例
model = LSTM_MLP(input_dim, hidden_dim, num_layers, 4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # weight_decay参数是L2正则化项
model.to(device)

# 运行训练和验证
for epoch in range(100):
    train_loss, train_acc = train_epoch(train_loader, model, criterion, optimizer, device)
    val_loss, val_acc = validate_epoch(val_loader, model, criterion, device)
    print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
# 计算测试集预测结果并保存
model.eval()
test_tensor = test_tensor.to(device)
with torch.no_grad():
    test_pred = model(test_tensor).argmax(dim=1)
    with open('test_pred.txt', 'w') as f:
        for label in tqdm(test_pred.cpu().numpy()):
            f.write(str(label) + '\n')
