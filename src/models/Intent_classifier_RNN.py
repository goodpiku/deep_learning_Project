from model import Model
import torch.nn as nn
import torch
from typing import List, Dict


class IntentclassifierRNN(Model, ):
    def __init__(self, embedding_size: int, num_of_labels: int):
        super().__init__()
        self.rnn = nn.GRU(input_size=embedding_size, hidden_size=200, batch_first=True, bidirectional=True)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.7),
            nn.Linear(in_features=400, out_features=100),
            nn.ReLU(),
            nn.Dropout(p=0.7),
            nn.Linear(in_features=100, out_features=num_of_labels))

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_data):
        embedded_text = input_data['processed_text'].to(self.device)
        output, hidden_state = self.rnn(embedded_text)
        hidden_state = torch.cat([hidden_state[0], hidden_state[1]], dim=1)  # concatination of hidden outputs

        predicted_value = self.classifier(hidden_state)
        predicted_probabilities = self.softmax(predicted_value)
        return predicted_probabilities
