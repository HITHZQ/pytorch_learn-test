import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(230),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                 ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
}

data_directory = './data/DATA0'
trainset = torchvision.datasets.ImageFolder(os.path.join(data_directory, 'train'), data_transforms['train'])
testset = torchvision.datasets.ImageFolder(os.path.join(data_directory, 'test'), data_transforms['test'])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=5, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=5, shuffle=False, num_workers=0)



def imshow(inputs):
    inputs = inputs/2 + 0.5
    inputs = inputs.numpy().transpose((1, 2, 0))
    plt.imshow(inputs)
    plt.show()

inputs, classes = next(iter(trainloader))
imshow(torchvision.utils.make_grid(inputs))

from torchvision import models
alexnet = models.alexnet(pretrained=True)
print(alexnet)

import torch.nn as nn

for param in alexnet.parameters():
    param.requires_grad = False

alexnet.classifier = nn.Sequential(
    nn.Dropout(),
    nn.Linear(256*6*6,4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Linear(4096,2),
    )


cuda = torch.cuda.is_available()
if cuda:
    alexnet = alexnet.cuda()


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(alexnet.classifier.parameters(), lr=0.001, momentum=0)


def train(model, criterion, optimizer, epochs=1):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if cuda :
                inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print('[epoch:%d, batch:%d], loss:%.3f' %(epoch+1, i, running_loss/100))
                running_loss = 0.0
    print('Finished Training')

def test(model, testloader):
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        if cuda :
            images = images.cuda()
            labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


def load_param(model,path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))

def save_param(model, path):
    torch.save(model.state_dict(), path)

load_param(alexnet,'tl_model.pkl')
train(alexnet, criterion, optimizer, epochs=2)
save_param(alexnet,'tl_model.pkl')
test(alexnet, testloader)
