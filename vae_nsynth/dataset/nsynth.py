import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200
import torch; torch.manual_seed(-1)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import os
import numpy as np

import audiofile
from librosa.feature import melspectrogram

import os
import numpy as np

class NsynthDataset(Dataset):
  def __init__(self, path, n_fft=2048, hop_length=512, transform=None, subset=None):
    """
    Args :
      path (string) : Path to the wav files. WARNING : All wav files
      MUST have same length and same sampling rate.

      n_fft (int) : n_fft option for librosa melspectrogram.

      hop_length (int) : hop_length option for librosa melspectrogram.
      
      transform (callable) : apply a transform to each batch.
      
      subset (list of string) : use only a subset of the dataset 
    """
    self.MAX = 17341.379
    self.MIN = 0
    
    self.path = path
    self.wav_list = os.listdir(path)
    wav_path = os.path.join(self.path,
                            self.wav_list[0])
                            
    if subset != None :
        self.wav_list = []
        for l in subset :
            self.wav_list.append(path + "/" + l)
    wav, sr = audiofile.read(wav_path)

    self.sr = sr
    self.wav_len = len(wav)
    self.n_fft = n_fft
    self.hop_length = hop_length

    melspec = melspectrogram(wav, sr, n_fft=self.n_fft,
                             hop_length=self.hop_length)
    self.n_ts = melspec.shape[0]
    self.n_freqb = melspec.shape[1]

    self.transform = transform

  def __len__(self):
    return len(self.wav_list)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    wav_path = os.path.join(self.path,
                            self.wav_list[idx])
    wav, sr = audiofile.read(wav_path)
    melspec = melspectrogram(wav, sr, n_fft=self.n_fft,
                             hop_length=self.hop_length)
    if self.transform != None :
      melspec = self.transform(melspec)

    return (melspec-self.MIN)/(self.MAX - self.MIN)

