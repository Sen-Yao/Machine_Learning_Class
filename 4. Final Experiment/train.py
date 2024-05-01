import torch
import os
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model import DualEncoderClassifier
from sklearn.metrics import f1_score


def train_epoch(loader, model, criterion, optimizer, device):
    """
    train for an epoch
    :param loader: DataLoader
    :param model: model
    :param criterion: criterion
    :param optimizer: optimizer
    :param device: cuda or cpu
    :return: loss, acc and macro F1
    """
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_preds, all_labels = [], []

    for input, labels in loader:
        input1, input2 = torch.chunk(input, 2, dim=1)
        input1, input2, labels = input1.to(device), input2.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input1, input2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

        all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / total_samples, total_correct / total_samples, train_f1


def validate_epoch(loader, model, criterion, device):
    """
    validate for an epoch
    :param loader: DataLoader
    :param model: model
    :param criterion: criterion
    :param device: cuda or cpu
    :return: loss, acc and macro F1
    """
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for input, labels in loader:
            input1, input2 = torch.chunk(input, 2, dim=1)
            input1, input2, labels = input1.to(device), input2.to(device), labels.to(device)
            outputs = model(input1, input2)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / total_samples, total_correct / total_samples, val_f1


def train(train_label_tensor, train_tensor, test_tensor, epoch_num, lr, hidden_dim, num_layers, weight_decay):
    """
    Train the model
    :param train_label_tensor: Labels of training set
    :param train_tensor: Tensors of training set
    :param test_tensor: Tensors of testing set
    :param epoch_num: number of epochs
    :param lr: learning rate
    :param hidden_dim: hidden layers' dimension of LSTM
    :param num_layers: Number of hidden layer of LSTM
    :param weight_decay: Weight Decay
    :return: model
    """
    print('start training')
    # Divide training set and validating set
    train_tensor, val_tensor, train_label_tensor, val_label_tensor = train_test_split(train_tensor, train_label_tensor,
                                                                                      test_size=0.2, random_state=42)

    # Create dataloader
    train_loader = DataLoader(TensorDataset(train_tensor, train_label_tensor), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_tensor, val_label_tensor), batch_size=64, shuffle=False)

    model = DualEncoderClassifier(300, hidden_dim, num_layers, 4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    # Start Training
    for epoch in range(epoch_num):
        train_loss, train_acc, train_f1 = train_epoch(train_loader, model, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = validate_epoch(val_loader, model, criterion, device)
        print(
            f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')

    # Apply model on testing set
    model.eval()
    test_tensor = test_tensor.to(device)
    try:
        os.mkdir('output')
    except FileExistsError:
        pass
    test_input1, test_input2 = torch.chunk(test_tensor, 2, dim=1)
    with torch.no_grad():
        test_pred = model(test_input1, test_input2).argmax(dim=1)
        # Save result
        with open('output/test_pred.txt', 'w') as f:
            for label in (test_pred.cpu().numpy()):
                f.write(str(label) + '\n')
    return model
