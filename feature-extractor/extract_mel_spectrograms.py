# Extracts mel-spectrograms

import argparse
import os
import yaml

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



class soccernet_ms_npy_DS(Dataset):
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
        idx_old = self.samples[idx]['idx']
        
        sample = {'path': path,
                  'ms':ms,'idx': idx_old,
                  'label':label}
        
        return sample



# Consider moving these to src/utils for helper methods
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise NotADirectoryError(string)


# TODO: Add loading of yaml config
def load_yaml(config):
    with open(config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

class 

# TODO: add loading method librosa

# TODO: add pytorch code for melspectogram

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--path',type=dir_path)
    parser.add_argument('--config',type=file_path)
    args = parser.parse_args()
    print(args.path)
    print(args.config)
    config = load_yaml(args.config)
    # Example usage.. TODO: when extracting mel spectogram, send mel_spectogram config to method
    print(config['MEL_SPECTOGRAM']['SAMPLE_RATE'])

    print("hello world")