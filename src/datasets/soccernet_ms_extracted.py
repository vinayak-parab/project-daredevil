# Clip dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
import json
import torchvision
import datetime
import torchaudio
import librosa
from subprocess import Popen, PIPE
import re
import time



class soccernet_ms_npy_audio_only(Dataset):
    """Soccernet Dataset"""
    
    
    def __init__(self,npy_file,
                 root_dir,
                 transform=None,
                 train=True):
    
        self.samples = np.load(root_dir+npy_file,allow_pickle=True)        # GENERALIZE
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
       
    def __len__(self):
        return len(self.samples)
    

    def __getitem__(self,idx):
        """Returns a sample containing video path, clip and label"""
        
        if torch.is_tensor(idx):
            idx.tolist()
        
        if self.train:
            if idx in [2209,2210,2212,2213,2215,2217,2222]: # ultradirty hack - fix later
                idx = 0
        
        path = str(self.samples[idx]['audiopath'][:-11]+str(idx)+"_ms.npy")
        ms = np.load(path)
        ms = ms-np.min(ms) / (np.max(ms)-np.min(ms))
        label = self.samples[idx]['label']
        info = self.samples[idx]['annotation']
        print(info)
        idx_old = self.samples[idx]['idx']
        
        sample = {'path': path,
                  'ms':ms,'idx': idx_old,
                  'label':label, 'info':info}
        
        return sample