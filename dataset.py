# Copyright (c) 2022 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

import os
import numpy as np

from scipy.io.wavfile import read as wavread
import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

import torchaudio


class CleanNoisyPairDataset(Dataset):
    """
    Create a Dataset of clean and noisy audio pairs. 
    Each element is a tuple of the form (clean waveform, noisy waveform, file_id)
    """
    
    def __init__(self, root='./', subset='training', crop_length_sec=0):
        super(CleanNoisyPairDataset).__init__()

        assert subset is None or subset in ["predicting"]
        self.crop_length_sec = crop_length_sec
        self.subset = subset
        
        
        sortkey = lambda name: '_'.join(name.split('_')[-2:])  # specific for dns due to test sample names
        
        noisy_files = os.listdir(os.path.join("raw_wav"))
        
        noisy_files.sort(key=sortkey)

        self.files = []
        for _n in  noisy_files:
            
            self.files.append((np.array([1]), 
                                os.path.join("raw_wav", _n)))
        self.crop_length_sec = 0

    def __getitem__(self, n):
        fileid = self.files[n]
        noisy_audio, sample_rate = torchaudio.load(fileid[1])
        noisy_audio = noisy_audio.squeeze(0)
        
        noisy_audio =  noisy_audio.unsqueeze(0)
        return (np.array([1]), noisy_audio, fileid)

    def __len__(self):
        return len(self.files)


def load_CleanNoisyPairDataset(root, subset, crop_length_sec, batch_size, sample_rate, num_gpus=1):
    """
    Get dataloader with distributed sampling
    """
    dataset = CleanNoisyPairDataset(root=root, subset=subset, crop_length_sec=crop_length_sec)                                                 
    kwargs = {"batch_size": batch_size, "num_workers": 4, "pin_memory": False, "drop_last": False}

    if num_gpus > 1:
        train_sampler = DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, **kwargs)
    else:
        dataloader = torch.utils.data.DataLoader(dataset, sampler=None, shuffle=True, **kwargs)
        
    return dataloader
    