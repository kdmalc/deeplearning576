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
import torch.nn.init as init

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path

batch_size = 64
test_batch_size = 1000
epochs = 7
lr = 0.01
try_cuda = False # I do not have cuda...
seed = 1000
# Initialize the random seed
logging_interval = 10 # how many batches to wait before logging
exact_folder = r"\RunX"
logging_dir = r"C:\Users\kdmen\Desktop\Fall23\ELEC576\HW1\results" + exact_folder

# 1) setting up the logging
#[inset-code: set up logging]
if logging_dir is None:
    logging_dir = os.path.join('runs', datetime.now().strftime('%b%d_%H-%M-%S'))
else:
    logging_dir = os.path.join(logging_dir, datetime.now().strftime('%b%d_%H-%M-%S'))
writer = SummaryWriter(log_dir=logging_dir)

#deciding whether to send to the cpu or not if available
if torch.cuda.is_available() and try_cuda:
    cuda = True
    torch.cuda.manual_seed(seed)
else:
    cuda = False
    torch.manual_seed(seed)

# Setting up data
given_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.01307,), (0.3081,))
])

# Both taken from the provided code at: https://github.com/motokimura/pytorch_tensorboard/blob/master/main.py
#train_loader = [inset-code]
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        'data', 
        train=True, 
        download=True,
        transform=given_transform),
    batch_size=batch_size, 
    shuffle=True
)
#test_loader = [inset-code]
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        'data', 
        train=False, 
        download=True,
        transform=given_transform),
    batch_size=test_batch_size, 
    shuffle=True
)

# Defining Architecture, loss, and optimizer
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        self.use_Xavier = False

        # Initialize the weights using Xavier initialization
        self.initialize_weights()

    def initialize_weights(self):
        if self.use_Xavier:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    init.xavier_normal_(m.weight)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        x = F.softmax(x,dim=1) #Dimension out of range (expected to be in range of [-2, 1], but got 10)
        return x

#[inset-code: instantiate model]
# Setup the network 
model = Net()

#optimizer = [inset-code: USE AN ADAM OPTIMIZER]
# Setup optimizer
#optimizer = optim.SGD(model.parameters(), lr=lr) #weight_decay=0.001
optimizer = optim.Adam(model.parameters(), lr=lr)
#optimizer = optim.Adagrad(model.parameters(), lr=lr)
#optimizer = optim.Adadelta(model.parameters(), lr=lr)
#optimizer = optim.RMSprop(model.parameters(), lr=lr)#, weight_decay=0.001, momentum=0.9)

# Defining the test and trainig loops
eps=1e-13

def train(epoch):
    model.train()

    criterion = nn.NLLLoss() 

    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data) # forward
        loss = criterion(torch.log(output+eps), target) # = sum_k(-t_k * log(y_k))
        loss.backward()
        optimizer.step()

        if batch_idx % logging_interval == 0:
            #[inset-code: print and log the performance]
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item() ) #loss.data[0]
            )

            # Log train/loss to TensorBoard at every iteration
            n_iter = (epoch - 1) * len(train_loader) + batch_idx + 1
            writer.add_scalar('train/loss', loss.data.item(), n_iter) #loss.data[0]
        if batch_idx % 100 == 0:
            n_iter = (epoch - 1) * len(train_loader) + batch_idx + 1
            # Compute statistics
            for name, param in model.named_parameters():
                #if 'weight' in name:
                writer.add_scalar(f'{name}/std', param.std(), n_iter)
                writer.add_scalar(f'{name}/min', param.min(), n_iter)
                writer.add_scalar(f'{name}/max', param.max(), n_iter)
                # Histogram
                if 'weight' in name:
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
    criterion = nn.NLLLoss(size_average = False)

    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()

        output = model(data)

        test_loss += criterion(torch.log(output+eps), target,).item() # sum up batch loss (later, averaged over all test samples)
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    #[inset-code: print the performance]
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_accuracy)
    )

    # Log test/loss and test/accuracy to TensorBoard at every epoch
    n_iter = epoch * len(train_loader)
    #[inset-code: log the performance]
    writer.add_scalar('test/loss', test_loss, n_iter)
    writer.add_scalar('test/accuracy', test_accuracy, n_iter)

# Training loop

#[inset-code: running test and training over epoch]
# Start training
for epoch in range(1, epochs + 1):
    train(epoch)
    test(epoch)

writer.close()

# Commented out IPython magic to ensure Python compatibility.
"""
#https://stackoverflow.com/questions/55970686/tensorboard-not-found-as-magic-function-in-jupyter

#seems to be working in firefox when not working in Google Chrome when running in Colab
#https://stackoverflow.com/questions/64218755/getting-error-403-in-google-colab-with-tensorboard-with-firefox

# %load_ext tensorboard
# %tensorboard --logdir [dir]

"""