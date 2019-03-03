# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:44:22 2019

@author: Berkan
"""

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class VAE(nn.Module):
    
    def __init__(self, support_z):
        super(VAE, self).__init__()
        self.support_z = support_z
        self.standard_normal = torch.distributions.MultivariateNormal(
                torch.zeros(support_z), torch.eye(support_z))
        
        self.hidden_encoder = nn.Linear(28*28, 500)
        self.output_encoder = nn.Linear(500, 2*support_z)
        
        self.hidden_decoder = nn.linear(support_z, 500)
        self.output_decoder = nn.linear(500, 28*28)
        
    def forward(self, x):
        x = F.relu(self.hidden_encoder(x))
        x = F.relu(self.output_encoder(x))
        
        mean = x[:, :, self.support_z:]
        std = x[:, :, :self.support_z]
        epsilon = self.standard_normal.sample()
        
        z = mean + std*epsilon
        
        x = F.relu(self.hidden_decoder(z))
        x = F.sigmoid(self.output_decoder)
        
        return mean, std, x
    
def ELBO(mean, std, x, batch):
    return 0.5*(mean.shape[0] + torch.sum(torch.log(std**2)) - 
                torch.sum(mean**2) - torch.sum(std**2) ) - torch.dist(x, batch, 2)
    
def train(epoxh_num):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
          print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))
          train_losses.append(loss.item())
          train_counter.append(
            (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))


if __name__ == '__main__':
    # Some parameters for training
    use_gpu = False
    n_epochs = 3
    batch_size_train = 100
    batch_size_test = 1000
    learning_rate = 0.01
    weight_decay = 0.01
    log_interval = 10
    
    random_seed = 1
    if not use_gpu:
        torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    
    # Create data handlers
    train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST\
               ('../', train=True, download=True,\
                transform=torchvision.transforms.
                Compose([torchvision.transforms.ToTensor(),
                         torchvision.transforms.
                         Normalize((0.1307,), (0.3081,))])), 
    batch_size=batch_size_train, shuffle=True, num_workers=4)
    
    test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST\
                ('../', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ])),
    batch_size=batch_size_test, shuffle=True, num_workers=4)

    # Create optimizer
    vae1 = VAE(support_z=3)
    if use_gpu:
        vae1.cuda()
        
    optim = optim.Adagrad(vae1.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = []

    for _ in range(n_epochs):
        for idx, batch, label in enumerate(train_loader):
            optim.zero_grad()
            mean, std, x = vae1(batch)
            
            # We want to minimize negative ELBO, maximize ELBO
            loss = -1*ELBO(mean, std, x, batch)
            loss.backward()
            optim.step()
            
            if idx % log_interval == 0:
                train_counter.append(train_counter[-1] + batch.shape[0]*log_interval)
                train_losses.append(loss)
                
                test_losses.append()
                
    
    
    
    
    
    
    
    
