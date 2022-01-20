import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
from torch.utils.tensorboard import SummaryWriter 

import os
import numpy as np

from model2 import *
from dataset.nsynth import * 

"""
Resume training from a previous .pth (pickle file) containing : 
-model parameters
-optimizer parameters
-number of epochs
"""


device = 'cuda' if torch.cuda.is_available() else 'cpu'  #Définit le système effectuant les calculs selon la puissance disponible
print(device)


nsynth_ds = NsynthDataset('/fast-1/lemerle/data/nsynth-valid/audio',
                          transform=torchvision.transforms.ToTensor())
data = torch.utils.data.DataLoader(
        nsynth_ds,
        batch_size=128,
        shuffle=True)




def train(autoencoder, opt, data, epochs=20, epoch0=0, warmup=True, beta_0=0, Nt=5, beta_Nt=4):
    #opt = torch.optim.Adam(autoencoder.parameters())
    writer = SummaryWriter()
    print("warmup=", warmup)
    #warmup : beta goes from beta_0 to beta_Nt in Nt epochs
    #No warmup : beta=1
    beta=1
    i = 0
    for epoch in range(epochs):
        print(epoch+1+epoch0,"/",epochs+epoch0)
        if (warmup==True) and (epoch<=Nt):
            beta=epoch*(beta_Nt - beta_0)/Nt + beta_0
            print("beta=",beta)

        for x in data:
            i += 1
            if i%100 == 0:
              print(f"epoch : {epoch} ; samples : {i}")
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + beta*autoencoder.encoder.kl
            loss.backward()
            writer.add_scalar('Loss', loss, i)
            print(loss)
            opt.step()
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': autoencoder.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'i': i
                    
                    },f'trains/retrain18/model-ep{epoch + epoch0}.pth')

    return autoencoder

LATENT_DIMS = 256
EPOCHS = 100
from torchsummary import summary

vae = VariationalAutoencoder(LATENT_DIMS, 1, nsynth_ds.n_freqb, nsynth_ds.n_ts).to(device) # GPU

opt = torch.optim.Adam(vae.parameters())

checkpoint = torch.load("trains/train18-22:54:27/model-ep99.pth")
vae.load_state_dict(checkpoint["model_state_dict"])
opt.load_state_dict(checkpoint["optimizer_state_dict"])

#vae = train(vae, opt, data, EPOCHS, checkpoint["epoch"], warmup=False, beta_0=0.1, beta_Nt=0.5, Nt=10)

summary(vae, (1, 128, 126))

