import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import datetime
from torch.utils.tensorboard import SummaryWriter 

import os
import numpy as np

from model.model2 import *
from dataset.nsynth import * 
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
print(device)


"""
For each architecture,
Grid search over :
			 *latent dimension size, 
			 *learning rate, 
			 *batch size, 
			 *warm-up epochs.
"""

"""
for l_d in [128, 256, 512] : 
	for lr in [0.001, 0.0005, 0.0001] :
		for bs in [64, 128, 256] :
			for nt in [1, 10, 50] :
				LATENT_DIMS = l_d
				LR = lr
				BATCH_SIZE = bs
				NT = nt
"""



LATENT_DIMS = 256
LR = 0.0005
BATCH_SIZE = 128
NT = 4
EPOCHS = 200

subset = ""
with open("subs.txt", "r") as f :
    subset = f.read().split("\n")

nsynth_ds = NsynthDataset('/fast-1/lemerle/data/nsynth-valid/audio',
                          transform=torchvision.transforms.ToTensor(),
                          subset=subset)
                          
data = torch.utils.data.DataLoader(
        nsynth_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True)

def train(autoencoder, data, lr, epochs=20,  warmup=True, beta_0=0, Nt=5, beta_Nt=4):

    now = datetime.datetime.now().strftime("%d-%X")
    os.mkdir(f"trains/train{now}")
    opt = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    writer = SummaryWriter()
    print("warmup=", warmup)
    #warmup : beta goes from beta_0 to beta_Nt in Nt epochs
    #No warmup : beta=1
    beta=1
    i = 0
    for epoch in range(epochs):
        print(epoch+1,"/",epochs)
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
                    
                    },f'trains/train{now}/model-ep{epoch}.pth')

    return autoencoder



vae = VariationalAutoencoder(LATENT_DIMS, 1, nsynth_ds.n_freqb, nsynth_ds.n_ts).to(device) # GPU
vae = train(vae, data, LR, EPOCHS, warmup=True, beta_0=0.1, beta_Nt=1, Nt=NT)
