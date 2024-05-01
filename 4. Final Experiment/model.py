import torch
from torch import nn
import torch.nn.functional as func


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional_lstm = nn.LSTM(input_size=input_dim,
                                          hidden_size=hidden_dim,
                                          num_layers=num_layers,
                                          batch_first=True,
                                          bidirectional=True)
        self.fc = nn.Linear(2*hidden_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.bidirectional_lstm(x)
        # Pass through fully connected layer
        z = torch.tanh(self.fc(lstm_out))

        # Attention mechanism
        attn_weights = func.softmax(self.attn(z).squeeze(2), dim=1)
        attn_applied = torch.sum(z * attn_weights.unsqueeze(2), dim=1)

        return attn_applied


class Classifier(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(2*hidden_dim, num_classes)  # Combine features from both sentences

    def forward(self, arg1, arg2):

        combined = torch.cat((arg1, arg2), dim=1)
        logit = self.fc(combined)
        return func.log_softmax(logit, dim=1)


class DualEncoderClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(DualEncoderClassifier, self).__init__()
        self.encoder1 = Encoder(input_dim, hidden_dim, num_layers)
        self.encoder2 = Encoder(input_dim, hidden_dim, num_layers)
        self.classifier = Classifier(hidden_dim, num_classes)

    def forward(self, input1, input2):

        encoded_input1 = self.encoder1(input1)
        encoded_input2 = self.encoder2(input2)

        output = self.classifier(encoded_input1, encoded_input2)
        return output
