import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch.nn.init as init
from sklearn.metrics import f1_score


# 训练模型
def LSTM_train_epoch(loader, model, criterion, optimizer, device):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_preds, all_labels = [], []

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

        # 收集预测和标签
        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # 计算F1分数
    train_f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / total_samples, total_correct / total_samples, train_f1


def LSTM_validate_epoch(loader, model, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += inputs.size(0)

            # 收集预测和标签
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算F1分数
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / total_samples, total_correct / total_samples, val_f1


# 定义模型
class DualLSTM_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(DualLSTM_MLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # LSTM for the first argument
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # LSTM for the second argument
        self.lstm2 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        # Merge and classify
        self.fc1 = nn.Linear(hidden_dim * 2, 256)  # Doubled because we concatenate
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def init_weights(self):
        # 初始化LSTM权重
        for name, param in self.lstm1.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        for name, param in self.lstm2.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        # 初始化全连接层权重
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)

        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        self.fc3.bias.data.fill_(0)


    def forward(self, x):
        # Initial states
        split_x = torch.split(x, split_size_or_sections=20, dim=1)

        x1, x2 = split_x
        h0 = torch.zeros(self.num_layers, x1.size(0), self.hidden_dim).to(x1.device)
        c0 = torch.zeros(self.num_layers, x1.size(0), self.hidden_dim).to(x1.device)
        # Process each argument with its LSTM
        out1, _ = self.lstm1(x1, (h0, c0))
        out2, _ = self.lstm2(x2, (h0, c0))

        # Use the last hidden state
        out1 = out1[:, -1, :]
        out2 = out2[:, -1, :]

        # Concatenate outputs
        out = torch.cat((out1, out2), dim=1)
        out = self.dropout1(out)

        # MLP
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out



def train(train_label_tensor, train_tensor, test_tensor):
    print('start training')
    # 划分训练集和验证集
    train_tensor, val_tensor, train_label_tensor, val_label_tensor = train_test_split(train_tensor, train_label_tensor,
                                                                                      test_size=0.2, random_state=42)

    # 创建数据加载器
    train_loader = DataLoader(TensorDataset(train_tensor, train_label_tensor), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor, val_label_tensor), batch_size=64, shuffle=False)

    # LSTM 参数
    input_dim = 300  # GloVe 向量维度
    hidden_dim = 128  # LSTM 隐藏层维度
    num_layers = 3  # LSTM 层数

    # 创建模型实例
    model = DualLSTM_MLP(input_dim, hidden_dim, num_layers, 4)
    model.init_weights()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)  # weight_decay参数是L2正则化项
    model.to(device)

    # 运行训练和验证
    for epoch in range(50):
        train_loss, train_acc, train_f1 = LSTM_train_epoch(train_loader, model, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = LSTM_validate_epoch(val_loader, model, criterion, device)
        print(
            f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
    # 计算测试集预测结果并保存
    model.eval()
    test_tensor = test_tensor.to(device)
    with torch.no_grad():
        test_pred = model(test_tensor).argmax(dim=1)
        with open('test_pred.txt', 'w') as f:
            for label in (test_pred.cpu().numpy()):
                f.write(str(label) + '\n')
