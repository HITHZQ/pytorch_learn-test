import torch
import matplotlib.pyplot as plt
from torch import nn,optim
from torch.nn import functional as F


x = torch.unsqueeze(torch.linspace(-3,3,100000),dim=1)
y = x.pow(3) + 0.3*torch.randn(x.size())

class Net(nn.Module):
    def __init__(self, input_feature, num_hidden, outputs):
        super(Net, self).__init__()
        self.hidden = nn.Linear(input_feature, num_hidden)
        self.out = nn.Linear(num_hidden, outputs)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x


cuda = torch.cuda.is_available()
if cuda :
    net = Net(input_feature=1, num_hidden=20, outputs=1).cuda()
    inputs = x.cuda()
    target = y.cuda()
else:
    net = Net(input_feature=1, num_hidden=20, outputs=1)
    inputs = x
    target = y

optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
criterion = nn.MSELoss()

def train(model, criterion, optimizer, epochs):
    for epoch in range(epochs):

        output = model(inputs)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 80 == 0:
            print('Epoch [{}/{}], Loss: {:', epoch+1, epochs, loss )
            draw(output, loss)

    return model,loss

def draw(output, loss):
    if cuda :
        output = output.cpu()
    plt.cla()
    plt.scatter(x.numpy(),y.numpy())
    plt.plot(x.numpy(),output.data.numpy(),'r-',lw=5)
    plt.text(0.5,0,'Loss: {:.4f}'.format(loss),fontdict={'size':20,'color':'red'})
    plt.pause(0.005)

net,loss = train(net, criterion, optimizer, epochs= 4000)

print("Loss: {:.4f}".format(loss))

