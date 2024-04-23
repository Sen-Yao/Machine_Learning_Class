import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


MAX_FEATURES = 10000


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
            train_labels.append(int(line.strip()))
    return train_texts, train_labels, test_texts


def preprocess(texts, vectorizer=None, to_fit=False):
    if vectorizer is None or to_fit:
        vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
        vectors = vectorizer.fit_transform(texts)
    else:
        vectors = vectorizer.transform(texts)
    tensor = torch.tensor(vectors.todense(), dtype=torch.float32)
    return tensor, vectorizer


def prepare_loaders(train_texts, train_labels):
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    train_tensor, vectorizer = preprocess(train_texts, vectorizer, to_fit=True)
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)  # 确保标签为长整型

    # 划分训练集和验证集
    train_data, val_data, train_labels, val_labels = train_test_split(train_tensor, train_labels_tensor,
                                                                      test_size=0.2, random_state=42)

    train_dataset = TensorDataset(train_data.cuda(), train_labels.cuda())  # 放置在CUDA设备上
    val_dataset = TensorDataset(val_data.cuda(), val_labels.cuda())  # 放置在CUDA设备上

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    return train_loader, val_loader, vectorizer


def train(input_size, train_loader, val_loader, epochs):
    hidden_size1 = 256
    hidden_size2 = 64
    output_size = 20
    model = MLP(input_size, hidden_size1, hidden_size2, output_size)
    model.cuda()

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 在验证集上计算损失和正确率
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        print('Epoch [%d/%d], Training Loss: %.4f, Validation Loss: %.4f, Accuracy: %.2f%%' % (
            epoch + 1, epochs, running_loss, val_loss, val_accuracy))
    return model


def predict_and_save(model, test_texts, vectorizer):
    test_tensor, _ = preprocess(test_texts, vectorizer)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=32, shuffle=False)
    predicted_labels = []

    with torch.no_grad():
        for inputs in test_loader:
            outputs = model(inputs[0])
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels.extend(predicted.tolist())

    with open('test/test_labels.txt', 'w') as file:
        for label in predicted_labels:
            file.write(str(label) + '\n')


def main():
    print(torch.cuda.is_available())
    print(torch.__version__)
    train_texts, train_labels, test_texts = read_file()
    train_loader, val_loader, vectorizer = prepare_loaders(train_texts, train_labels)
    model = train(MAX_FEATURES, train_loader, val_loader, 50)
    predict_and_save(model, test_texts, vectorizer)


if __name__ == '__main__':
    main()

