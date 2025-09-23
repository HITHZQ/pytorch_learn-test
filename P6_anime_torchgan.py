import os
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
import torch.utils as data
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.utils as vutils

import torchgan
from torchgan.models import *
from torchgan.losses import *
from torchgan.trainer import Trainer

data_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

trainset = dsets.ImageFolder('./data/faces', data_transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("training images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=2, value_range=(-1, 1)), (1, 2, 0)))
plt.show()

dcgan_network = {
    'generator': {
        "name": DCGANGenerator,
        "args": {
            "encoding_dims": 100,
            "out_channels": 3,
            "step_channels": 32,
            "nonlinearity": nn.LeakyReLU(0.2),
            "last_nonlinearity": nn.Tanh()
        },
        "optimizer": {
            "name": Adam,
            "args": {
                "lr": 0.0002,
                "betas": (0.5, 0.999)
            }
        }

    },
    'discriminator': {
        "name": DCGANDiscriminator,
        "args": {
            "in_channels": 3,
            "step_channels": 32,
            "nonlinearity": nn.LeakyReLU(0.2),
            "last_nonlinearity": nn.LeakyReLU(0.2)
        },
        "optimizer": {
            "name": Adam,
            "args": {
                "lr": 0.0002,
                "betas": (0.5, 0.999)
            }
        }
    }
}


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.backends.cudnn.deterministic = True
    epochs = 400
else:
    device = torch.device("cpu")
    epochs = 5
print(device)
print(epochs)

lsgan_losses = [LeastSquaresGeneratorLoss(), LeastSquaresDiscriminatorLoss()]
wgangp_losses = [WassersteinGeneratorLoss(), WassersteinDiscriminatorLoss(), WassersteinGradientPenalty()]

trainer = Trainer(dcgan_network, wgangp_losses, sample_size=64, epochs=epochs, device=device,)
trainer(dataloader)
