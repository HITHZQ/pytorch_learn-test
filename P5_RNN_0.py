
# RNN MODEL
from torch import nn

rnn_cell = nn.RNNCell(5, 7)

import torch

input = torch.randn(1, 5)
hidden = torch.randn(1, 7)
print(rnn_cell(input, hidden))

rnn = nn.RNN(5, 7)
inputs = torch.randn(3, 2, 5)
hiddens = torch.randn(1, 2, 7)
print(rnn(inputs, hiddens))


# LSTM MODEL
lstm_cell = nn.LSTMCell(5, 7)
print(lstm_cell)
