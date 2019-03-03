# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:44:22 2019

@author: Berkan
"""
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchvision
import torch
import numpy as np
from torchvision.utils import make_grid, save_image



class VAE(nn.Module):
    
    def __init__(self, device, support_z):
        super(VAE, self).__init__()
        self.device = device
        self.support_z = support_z
        self.standard_normal = torch.distributions.MultivariateNormal(
                torch.zeros(support_z), torch.eye(support_z))
        
        self.hidden_encoder = nn.Linear(28*28, 500)
        self.output_encoder = nn.Linear(500, 2*support_z)
        
        self.hidden_decoder = nn.Linear(support_z, 500)
        self.output_decoder = nn.Linear(500, 28*28)
        
    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)  # For MNIST [100, 1, 28, 28] -> [100, 1, 784]

        x = F.relu(self.hidden_encoder(x))
        x = self.output_encoder(x)

        mean = x[:, :, :self.support_z]
        std = torch.relu(x[:, :, self.support_z:])

        epsilon = self.standard_normal.sample([x.shape[0], 1]).to(self.device)
        
        z = mean + std*epsilon
        
        x = F.relu(self.hidden_decoder(z))
        x = torch.sigmoid(self.output_decoder(x))
        
        return mean, std, x

    def only_decoder(self, z):
        with torch.no_grad():
            x = F.relu(self.hidden_decoder(z))
            x = torch.sigmoid(self.output_decoder(x))

        return x


def ELBO(mean, std, x, batch):
    M = batch.shape[0]
    batch = batch.view(batch.shape[0], 1, -1)

    neg_KL_term = 0.5*torch.sum(1 + torch.log(std**2) - mean**2 - std**2)

    likelihood_term = -1*F.binary_cross_entropy(x, batch, reduction='sum')

    return (neg_KL_term + likelihood_term)/M


def smooth_list(inp_list, degree):

    return [np.mean(inp_list[indice1:indice1 + degree])  for indice1 in range(len(inp_list) - degree + 1) ]

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    # Some parameters for training
    n_epochs = 30
    batch_size = 100
    learning_rate = 0.005
    log_interval = 3000
    dz = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # # Create data handlers
    # train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('../', train=True, download=True, \
    #                                                                       transform=torchvision.transforms.
    #                                                                       Compose([torchvision.transforms.ToTensor()])),
    #                                            batch_size=batch_size, shuffle=True, num_workers=12)
    #
    # test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST \
    #                                               ('../', train=False, download=True,
    #                                                transform=torchvision.transforms.Compose([
    #                                                    torchvision.transforms.ToTensor()])),
    #                                           batch_size=batch_size, shuffle=True, num_workers=12)
    #
    # # Create optimizer
    # vae1 = VAE(device=device, support_z=dz)
    # vae1.to(device)
    #
    # optim = optim.Adagrad([
    #     {'params': vae1.hidden_encoder.parameters()},
    #     {'params': vae1.output_encoder.parameters()},
    #     {'params': vae1.hidden_decoder.parameters(), 'weight_decay': .01},
    #     {'params': vae1.output_decoder.parameters(), 'weight_decay': .01},
    # ], lr=learning_rate)
    #
    # # optim = optim.Adagrad(vae1.parameters(), lr=learning_rate)
    #
    # train_ELBOs = []
    # test_ELBOs = []
    # sample_counts = []
    #
    # observed_sample_count = 0
    # for epoch in range(n_epochs):
    #     for batch, _ in iter(train_loader):  # _ is for labels
    #         optim.zero_grad()
    #
    #         batch = batch.to(device)
    #         mean, std, x = vae1(batch)
    #         observed_sample_count += batch.shape[0]
    #
    #         # We want to minimize negative ELBO, i.e. maximize ELBO.
    #         loss = -1*ELBO(mean, std, x, batch)
    #         loss.backward()
    #         optim.step()
    #
    #         if observed_sample_count % log_interval >= log_interval - batch_size:
    #             # Append total observed sample count
    #             sample_counts.append(observed_sample_count)
    #             # Append the last train loss
    #             train_ELBOs.append(-loss.item())
    #             # Sample test images and append test loss
    #
    #             with torch.no_grad():
    #                 batch = next(iter(test_loader))[0]
    #                 batch = batch.to(device)
    #                 mean, std, x = vae1(batch)
    #                 loss = -1*ELBO(mean, std, x, batch)
    #                 test_ELBOs.append(-loss.item())
    #
    #             print('Epoch:', epoch, 'Sample count:', sample_counts[-1], 'Train and test ELBO:', train_ELBOs[-1], test_ELBOs[-1])
    #
    #
    # smoothed_train_curve = smooth_list(train_ELBOs, 2)
    # smoothed_test_curve = smooth_list(test_ELBOs, 2)
    # smoothed_sample_count = smooth_list(sample_counts, 2)
    #
    #
    # fig = plt.figure()
    # plt.plot(sample_counts, train_ELBOs, label='Train loss')
    # plt.plot(sample_counts, test_ELBOs, label='Test loss')
    # plt.legend()
    # plt.grid()
    # plt.xlabel('Training Sample Count')
    # plt.xscale('log')
    # plt.ylabel('ELBO')
    # plt.show()
    #
    # # SAVE
    # torch.save(vae1.state_dict(), '../vae_statedz' + str(dz))

    # LOAD

    vae2 = VAE(device, dz)
    vae2.load_state_dict(torch.load('../vae_statedz5'))
    vae2.eval()

    span = np.linspace(-4, 4, 20)

    image_list = []

    for z1 in span:
        for z2 in span:

            z = torch.ones((1, 1, 2))
            z[0, 0, 0] = z1
            z[0, 0, 1] = z2

            z = torch.randn((1, 1, dz))*1.2

            img = vae2.only_decoder(z)

            img = img.view(1, 28, 28)

            image_list.append(img)

    show(make_grid(image_list, padding=5, nrow=20))



