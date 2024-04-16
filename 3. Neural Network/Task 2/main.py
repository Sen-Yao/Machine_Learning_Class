import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # 指定softmax操作的维度

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)  # 应用softmax
        return x

def read_file():
    train_labels = []
    train_text_name = 'train/train_texts.dat'
    with open(train_text_name, 'rb') as f:
        train_texts = pickle.load(f)
    test_text_name = 'test/test_texts.dat'
    with open(test_text_name, 'rb') as f:
        test_texts = pickle.load(f)
    train_labels_name = 'train/train_labels.txt'
    with open(train_labels_name, 'r') as file:
        for line in file:
            train_labels.append(int(line))
    return train_texts, train_labels, test_texts

def preprocess(train_texts, train_labels, test_texts):
    vectorizer = TfidfVectorizer(max_features=10000)
    vectors_train = vectorizer.fit_transform(train_texts)
    train_tensor = torch.tensor(vectors_train.todense(), dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)  # 确保标签为长整型

    # 划分训练集和验证集
    train_data, val_data, train_labels, val_labels = train_test_split(train_tensor, train_labels, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader

def train(input_size, train_loader, val_loader, epochs):
    hidden_size1 = 4096
    hidden_size2 = 512
    output_size = 32
    model = MLP(input_size, hidden_size1, hidden_size2, output_size)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 在验证集上计算损失
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print('Epoch [%d/%d], Training Loss: %.4f, Validation Loss: %.4f' % (epoch + 1, epochs, running_loss, val_loss))
    return model

def main():
    train_texts, train_labels, test_texts = read_file()
    train_loader, val_loader = preprocess(train_texts, train_labels, test_texts)
    model = train(10000, train_loader, val_loader, 200)

if __name__ == '__main__':
    main()
