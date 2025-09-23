import torch
from torch import nn
from torchvision import datasets, transforms
import torch.nn.functional as F


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307, ), (0.3081, ))])

trainset = datasets.MNIST('data', train=True, download=True, transform=transform)
testset = datasets.MNIST('data', train=False, download=True, transform=transform)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(1, 6, 5)
        self.c3 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c3(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


cuda = torch.cuda.is_available()
if cuda :
    net = LeNet().cuda()
else :
    net = LeNet()

optimizer = torch.optim.SGD(net.parameters(),lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss()

train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                           shuffle=True, num_workers=2)

def train(model, criterion, optimizer, epochs=1):
    for epoch in range(epochs):
        running_loss = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 1000 == 999:
                print('Epoch: %d, batch %d, loss: %.3f ', epoch + 1, i + 1, running_loss/1000)
                running_loss = 0.0


    print('Finished Training')

train(LeNet, criterion, optimizer, epochs=2)
