The dataset folder contains the data processor, which construct a distorted dataset from the NSYNTH dataset.
Model defines the VAE as well as the architecture of the model (number of layers, layer types, kernels...).
The training and setting of hyperparameters (number of epochs, batch size...). Finally, the folder util_scripts
contains a python script for training on a small subset of the whole dataset, thus allowing for quick training 
and tuning, and test_inference, which is used for sampling from the latent, comparing between the mel-spectrogram
of a sample from the dataset and the reconstructed mel-spectrogram, and the inverse transformation from 
mel-spectro to wav. Finally, resume_train allows us to re-run the training from a given epoch.