import torch
import matplotlib.pyplot as plt
from torch import nn

cluster = torch.ones(500, 2)
data0 = torch.normal(4*cluster, 2)
data1 = torch.normal(-4*cluster, 2)
label0 = torch.zeros(500)
label1 = torch.ones(500)

x = torch.cat((data0, data1), ).type(torch.float)
y = torch.cat((label0, label1), ).type(torch.long)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x

cuda = torch.cuda.is_available()
if cuda :
    net = Net().cuda()
    inputs = x.cuda()
    target = y.cuda()
else :
    net = Net()
    inputs = x
    target = y

optimizer = torch.optim.SGD(net.parameters(),lr=0.02, momentum=0.9)
criterion = nn.CrossEntropyLoss()

def draw(output):
    if cuda:
        output = output.cpu()
    plt.cla()
    output = torch.max(output, 1)[1]
    pred_y = output.data.numpy().squeeze()
    target_y = y.numpy()
    plt.scatter(x.numpy()[:, 0], x.numpy()[:, 1], c=pred_y, s=10, lw=0, cmap='RdYlGn')
    accuracy = sum(pred_y == target_y)/1000.0
    plt.text(1.5, -4, 'Accuracy =%s' %accuracy, fontdict={'size':20 , 'color':'red'})
    plt.pause(0.1)


def train(model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        output = model(inputs)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 40 == 0:
            draw(output)



train(net, criterion, optimizer, 400)