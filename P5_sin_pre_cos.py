import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

#steps =np.linspace(0, 2*np.pi, 100, dtype=np.float32)
#input_x = np.sin(steps)
#target_x = np.cos(steps)
#plt.plot(steps, input_x, label="input_sin", color="blue")
#plt.plot(steps, target_x, label="target_cos", color="red")
#plt.legend(loc='best')
#plt.show()

class LSTM(nn.Module):
    def __init__(self, INPUT_SIZE):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=20,
            batch_first=True,
        )
        self.out = nn.Linear(20, 1)

    def forward(self, x, h_state, c_state):
        r_out, (h_state, c_state) = self.lstm(x, (h_state, c_state))
        outputs = self.out(r_out[0, :]).unsqueeze(0)
        return outputs, h_state, c_state

    def init_hidden(self):
        h_state = torch.rand(1, 1, 20)
        c_state = torch.rand(1, 1, 20)
        return h_state, c_state

lstm = LSTM(INPUT_SIZE=1)
optimizer = torch.optim.Adam(lstm.parameters(), lr=0.001)
criterion = nn.MSELoss()

h_state, c_state = lstm.init_hidden()

plt.figure(1, figsize=(12, 5))
plt.ion()

for step in range(600):
    start, end = step * np.pi, (step+1) * np.pi
    steps = np.linspace(start, end, 100, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    x = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(-1)
    y = torch.from_numpy(y_np).unsqueeze(0).unsqueeze(-1)
    prediction, h_state, c_state = lstm(x, h_state, c_state)
    h_state = h_state.data
    c_state = c_state.data
    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plt.plot(steps, y_np.flatten(), 'b', label='y')
    plt.plot(steps, prediction.data.numpy().flatten(), 'g', label='prediction')
    plt.draw();plt.pause(0.05)

plt.ioff()
plt.show()

