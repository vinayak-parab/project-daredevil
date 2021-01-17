

# dataset for sound
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
import json
import torchvision
import datetime
from subprocess import Popen, PIPE
import re
import torchaudio
from tqdm import tqdm
# TODO : add background data
class soccernet_dataset_generic(Dataset):
    """Soccernet Dataset"""
    
    def __init__(self,npy_file,
                 root_dir,lang,lang_dict,window_size=4,transform=None, background=True,window_position='center'):
        
        ## Todo : add option for resnet
        self.npy_file = np.load(npy_file)
        self.samples = list()
        self.lang = lang
        self.lang_sub_samples = self._samples_by_language(lang_dict,self.npy_file,lang)
        self.root_dir = root_dir
        self.waves = dict()
        self.ms = dict()
        self.mfcc = dict()
        self.resnet = dict()
        self.window_size=window_size
        self.transform =  transforms.Compose([transforms.ToTensor()])


        for e in self.lang_sub_samples:

            path, annotations = self.get_annotations(e)
            language = self.get_lang_label(e)
            duration1 = self._getVideoLength(self.root_dir + e + "/1.mkv")
            duration2 = self._getVideoLength(self.root_dir + e + "/2.mkv")

            self.samples.append([path,language,duration1,1])
            self.samples.append([path,language,duration2,2])    

    def __len__(self):
        return len(self.samples)
    
    def set_window_position(self,window_position):
        if not window_position in ['back','center','forward']:
            print("Invalid window position! please use one of the following: ['back','center','forward']")
            return
        else:
            self.window_position = window_position

    def get_window_position(self):
        return self.window_position
    
    def _getVideoLength(self,video_file):
        res = Popen(['ffmpeg', '-i', video_file, '-hide_banner'],stdout=PIPE,stderr=PIPE)
        none,meta = res.communicate()
        meta_out = meta.decode()
        #---| Take out info
        duration = re.search(r'Duration:.*', meta_out)
        return duration.group()[:21]
    
    def _samples_by_language(self,path_to_language_dict, sample_list,lang='all'):
        samples = list()
        with open(path_to_language_dict) as file:
            language_annotations = json.load(file)

        for s in sample_list:
            if lang == 'all':
                samples.append(s)
            elif lang == 'eng':
                if lang == language_annotations[s]:
                    samples.append(s)
            elif lang == 'other':
                if not 'eng' == language_annotations[s]:
                    samples.append(s)

        #else:
        #    if lang == language_annotations[s]:
        #        samples.append(s)

        return samples

    def set_window_size(self,window_size):
        self.window_size = window_size

    def __getitem__(self,idx):
        """Returns a sample containing video path, clip and label"""
        if torch.is_tensor(idx):
            idx.tolist()
        
        half = self.samples[idx][3]

        duration_raw = self.samples[idx][2]

        duration_min = int(duration_raw[-8:-6])
        duration_sec = int(duration_raw[-5:-3])
        duration = (duration_min * 60) + duration_sec

            
        vidpath = str(self.samples[idx][0])
        # Add bools to check if loaded -- if bool(self.waves..)
        wave = self.waves.get(vidpath)[half - 1] # Convert to time for waves
        ms = torch.from_numpy(self.ms.get(vidpath)[half - 1]) # Convert to time for waves
        resnet_features = self.resnet.get(vidpath)[half - 1]
        ms[ms.isneginf()] = 0
        # normalize
        #ms_spot = ( ms_spot - torch.min(ms_spot) ) / ( torch.max(ms_spot) - torch.min(ms_spot) )

        sample = {'vidpath': vidpath,
                  'duration:': duration,
                  'half':half,
                  'wave':wave,
                  'ms':ms,
                  'resnet_features':resnet_features,
                  'lang':self.samples[idx][3],
                  'idx':idx}
                    #'mfcc':-1, 'mfcc_idx':mfcc_idx,
    
        return sample
            
    def get_annotations(self,path):
        """ Reads json files and returns """
        with open(self.root_dir+path+"/Labels.json") as jsonfile:
            json_label = json.load(jsonfile)
        
        labels = [e for e in json_label['annotations']]
        
        return path,labels
    

    def get_lang_label(self,path):
        """ Reads json files and returns """
        with open(self.root_dir+path+"/lang_label_test.json") as jsonfile:
            json_label = json.load(jsonfile)
        
        return json_label
    
    def load_waves(self):
        if self.lang_sub_samples:
            for path in tqdm(self.lang_sub_samples):
                half1 = np.load(self.root_dir+path+"/1_wave.npy")
                half2 = np.load(self.root_dir+path+"/2_wave.npy")
                self.waves[path] = (half1,half2)
        else:
            print("Already loaded!")
            
    
    def load_resnet_features(self):
        if self.lang_sub_samples:
            for path in tqdm(self.lang_sub_samples):
                half1 = np.load(self.root_dir+path+"/1_ResNET_PCA512.npy")
                half2 = np.load(self.root_dir+path+"/2_ResNET_PCA512.npy")
                self.resnet[path] = (half1,half2)
        else:
            print("Already loaded!")
            
    def generate_mel_spectrograms(self, sr=16000, win_length=4,n_fft=512,step_size=0.025,save_features=False,load_features=False):
        if load_features:
            # if presaved features
            for key in tqdm(self.waves):
                ms1 = np.load(self.root_dir+key+"/temp_ms_1.npy")
                ms2 = np.load(self.root_dir+key+"/temp_ms_2.npy")
                self.ms[key] = (ms1,ms2)
        else:

            if not bool(self.ms):

                SAMPLE_RATE = sr # Should be 16 kHz
                WINDOW_LENGTH = int(0.025 * sr) # Should be 25 ms
                N_FFT = n_fft # Should be 512
                STEP_SIZE = WINDOW_LENGTH # Should be 25 ms
                
                for key in tqdm(self.waves):
                    half1 = self.waves[key][0]
                    half2 = self.waves[key][1]
                    
                    ms1 = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
                                                            win_length=WINDOW_LENGTH,
                                                            hop_length=STEP_SIZE,
                                                            n_fft=N_FFT)(torch.from_numpy(half1)).log10()
                    ms2 = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
                                            win_length=WINDOW_LENGTH,
                                            hop_length=STEP_SIZE,
                                            n_fft=N_FFT)(torch.from_numpy(half2)).log10()

                    if save_features:
                        np.save(self.root_dir+key+"/temp_ms_1.npy",ms1)
                        np.save(self.root_dir+key+"/temp_ms_2.npy",ms2)
                    self.ms[key] = (ms1,ms2)
            else:
                print("Already loaded")


    def describe(self):
        card = 0
        subs = 0
        goal = 0
        background = 0

        for sample in self.samples:
            annotation = sample[1]
        # Get label
            if ("card" in annotation["label"]): card += 1
            elif ("subs" in annotation["label"]): subs +=1
            elif ("soccer" in annotation["label"]): goal += 1
            elif ("background" in annotation["label"]): background += 1

        print("Description of dataset\n\n")
        print("\n ********* Classes *********")
        print("\n card = 0\n subs = 1\n goals = 2\n background = 3")

        print("\n ********* Distribution and count *********")
        print(f"\n N card: {card} \n N subs: {subs} \n N goal: {goal} \n N background: {background} \n \n Total : {card+subs+goal+background}")
        
        print("\n\n ********* Configuration *********")

        print(f"\n npy_file: {self.npy_file}\
                \n language: {self.lang}\
                \n root_dir: {self.root_dir}\
                \n transform: {self.transform}\
                ")
        print("\n\n ********* End of description *********")