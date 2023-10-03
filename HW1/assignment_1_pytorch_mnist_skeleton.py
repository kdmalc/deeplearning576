# -*- coding: utf-8 -*-
"""Assignment_1_Pytorch_MNIST.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1i9KpbQyFU4zfq8zLLns8a2Kd8PRMGsaZ

Overall structure:

1) Set Pytorch metada
- seed
- tensorflow output
- whether to transfer to gpu (cuda)

2) Import data
- download data
- create data loaders with batchsie, transforms, scaling

3) Define Model architecture, loss and optimizer

4) Define Test and Training loop
    - Train:
        a. get next batch
        b. forward pass through model
        c. calculate loss
        d. backward pass from loss (calculates the gradient for each parameter)
        e. optimizer: performs weight updates

5) Perform Training over multiple epochs:
    Each epoch:
    - call train loop
    - call test loop

Acknowledgments:https://github.com/motokimura/pytorch_tensorboard/blob/master/main.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path

batch_size = 64
test_batch_size = 1000
epochs = 10
lr = 0.01
try_cuda = True
seed = 1000
logging_interval = 10 # how many batches to wait before logging
logging_dir = None

# 1) setting up the logging
[inset-code: set up logging]

#deciding whether to send to the cpu or not if available
if torch.cuda.is_available() and try_cuda:
    cuda = True
    torch.cuda.mnaual_seed(seed)
else:
    cuda = False
    torch.manual_seed(seed)

# Setting up data
transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.01307,), (0.3081,))
])

train_loader = [inset-code]

test_loader = [inset-code]

# Defining Architecture,loss and optimizer
class Net(nn.Module):
    def __init__(self):

        super(Net, self).__init__()
        self.conv1 = [inset-code]
        self.conv2 = [inset-code]
        self.conv2_drop = [inset-code]
        self.fc1 = [inset-code]
        self.fc2 = [inset-code]

    def forward(self, x):

        x = [inset-code]
        x = [inset-code]
        x = [inset-code]
        x = [inset-code]
        x = [inset-code]
        x = [inset-code]
        x = F.softmax(x,dim=1)

        return x

[inset-code: instantiate model]

optimizer = [inset-code: USE AN ADAM OPTIMIZER]


# Defining the test and trainig loops
eps=1e-13

def train(epoch):
    model.train()

    #criterion = nn.CrossEntropyLoss()
    criterion = [inset-code]

    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.[inset-code]
        output = [inset-code]
        loss = criterion(torch.log(output+eps), target) # = sum_k(-t_k * log(y_k))
        loss[inset-code]
        optimizer[inset-code]

        if batch_idx % logging_interval == 0:
            [inset-code: print and log the performance]

    # Log model parameters to TensorBoard at every epoch
    for name, param in model.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram(
            f'{layer}/{attr}',
            param.clone().cpu().data.numpy(),
            n_iter)



def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.CrossEntropyLoss(size_average = False)
    criterion = nn.NLLLoss(size_average = False)

    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data)

        test_loss += criterion(torch.log(output+eps), target,).item() # sum up batch loss (later, averaged over all test samples)
        pred = [inset-code] # get the index of the max log-probability
        correct += [inset-code]

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    [inset-code: print the performance]

    # Log test/loss and test/accuracy to TensorBoard at every epoch
    n_iter = epoch * len(train_loader)
    [inset-code: log the performance]

# Training loop

[inset-code: running test and training over epoch]

writer.close()

# Commented out IPython magic to ensure Python compatibility.
"""
#https://stackoverflow.com/questions/55970686/tensorboard-not-found-as-magic-function-in-jupyter

#seems to be working in firefox when not working in Google Chrome when running in Colab
#https://stackoverflow.com/questions/64218755/getting-error-403-in-google-colab-with-tensorboard-with-firefox


# %load_ext tensorboard
# %tensorboard --logdir [dir]

"""