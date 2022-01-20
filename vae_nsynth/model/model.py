import torch; torch.manual_seed(-1)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import os
import numpy as np

import audiofile
from librosa.feature import melspectrogram


class Decoder(nn.Module):
    def __init__(self, latent_dims, n_chan, n_freqb, n_ts, filter_params=None):
        super(Decoder, self).__init__()
        if filter_params==None :
            self.filter_params={
            "filters": (32, 64, 128, 256, 512),
            "kernels": (3, 3, 3, 3, 3),
            "strides": ((2,1), 2, 2, 2, 2),
            "output_padding": (0, (0,1), (0,1), (0,1), (1,1)),
            "last_shape": (32,3,4)
            }

        self.n_chan = n_chan
        self.n_ts = n_ts
        self.n_freqb = n_freqb

        self.convsT = []


        for i in range(len(self.filter_params["filters"])):
            if i == len(self.filter_params["filters"]) - 1 :
                self.convsT.append(nn.ConvTranspose2d(self.filter_params["filters"][i],
                                        1,
                                        self.filter_params["kernels"][i],
                                        self.filter_params["strides"][i],
                                        output_padding=
                                                  self.filter_params["output_padding"][i]))
                self.convsT.append(nn.ReLU())
                self.convsT.append(nn.BatchNorm2d(1))

            else :
                self.convsT.append(nn.ConvTranspose2d(self.filter_params["filters"][i],
                                        self.filter_params["filters"][i+1],
                                        self.filter_params["kernels"][i],
                                        self.filter_params["strides"][i],
                                        output_padding=
                                                  self.filter_params["output_padding"][i]))
                self.convsT.append(nn.ReLU())
                self.convsT.append(nn.BatchNorm2d(self.filter_params["filters"][i+1]))
        self.convsT = nn.Sequential(*self.convsT)
        h_dim = np.prod(self.filter_params["last_shape"])
        self.lin = nn.Linear(latent_dims, h_dim)

    def forward(self, z):
        z = self.lin(z)
        z = z.reshape(-1, 32, 3, 4)
        z = self.convsT(z)
        return z.reshape(-1, self.n_chan, self.n_ts, self.n_freqb)

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, n_chan, n_freqb, n_ts, filter_params=None):
        """ Args:
        latent_dims (int) : Latent space dimension.

        n_chan  (int) : Number of channels (1 = mono, 2 = stereo, ...).

        n_freqb (int) : Number of frequency band in the mel-spectrogram.

        n_ts (int) : Number of time steps.

        filter_param (list of tuple of int/tuple of int) : Each line are the parameters for 
        one Conv2d and its mirrored ConvTranspose2d.
        """
        super(VariationalEncoder, self).__init__()
        if filter_params==None :
            self.filter_params={
            "filters": (512, 256, 128, 64, 32),
            "kernels": (3, 3, 3, 3, 3),
            "strides": (2, 2, 2, 2, (2,1)),
            "h_dim": 384
            }
        self.convs = []
        for i in range(len(self.filter_params["filters"])):
            if i == 0 :
                k = n_chan
            else :
                k = self.filter_params["filters"][i-1]
            self.convs.append(nn.Conv2d(k, 
                                        self.filter_params["filters"][i],
                                        self.filter_params["kernels"][i],
                                        self.filter_params["strides"][i]))
            self.convs.append(nn.ReLU())
            self.convs.append(nn.BatchNorm2d(self.filter_params["filters"][i]))
        self.convs.append(nn.Flatten())

        self.convs = nn.Sequential(*self.convs)

        self.linear_mu = nn.Linear(self.filter_params["h_dim"], latent_dims)
        self.linear_sigma = nn.Linear(self.filter_params["h_dim"], latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = self.convs(x)
        mu =  self.linear_mu(x)
        sigma = torch.exp(self.linear_sigma(x))

        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()

        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims, n_chan, n_freqb, n_ts):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, n_chan, n_freqb, n_ts)
        self.decoder = Decoder(latent_dims, n_chan, n_freqb, n_ts)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


