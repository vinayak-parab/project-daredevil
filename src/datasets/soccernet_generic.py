

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
# TODO : add background data
class soccernet_dataset_generic(Dataset):
    """Soccernet Dataset"""
    
    def __init__(self,npy_file,
                 root_dir,lang,lang_dict,window_size=4,transform=None):
        
        ## Todo : add option for resnet
        self.npy_file = np.load(npy_file)
        self.samples = list()
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
            
            for annotation in annotations:
                # Check that annotations hold correct labels
                        if ("card" in annotation["label"]) or ("subs" in annotation["label"]) or ("soccer" in annotation["label"]):
                            self.samples.append([path,annotation,"soccernet",language,(duration1,duration2)])


    def __len__(self):
        return len(self.samples)
    
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
            else:
                if lang == language_annotations[s]:
                    samples.append(s)

        return samples

    def set_window_size(self,window_size):
        self.window_size = window_size

    def __getitem__(self,idx):
        """Returns a sample containing video path, clip and label"""
        if torch.is_tensor(idx):
            idx.tolist()
        
        # get annotations
        time_half = int(self.samples[idx][1]["gameTime"][0])
        time_minute = int(self.samples[idx][1]["gameTime"][-5:-3])
        time_second = int(self.samples[idx][1]["gameTime"][-2:])
        annotation = self.samples[idx][1]
        
        if time_half == 1:
            duration_raw = self.samples[idx][4][0]
        elif time_half == 2:
            duration_raw = self.samples[idx][4][1]
        else:
            return "Error in time_half"

        duration_min = int(duration_raw[-8:-6])
        duration_sec = int(duration_raw[-5:-3])
        duration = (duration_min * 60) + duration_sec
        # Get label
        if ("card" in annotation["label"]): label = 0
        elif ("subs" in annotation["label"]): label = 1
        elif ("soccer" in annotation["label"]): label = 2
        elif ("background" in annotation["label"]): label = 3
        else: 
            print("Warning, label not compatible with set")
            return
        
        seconds_spot = (time_minute*60) + time_second
            
        vidpath = str(self.samples[idx][0])
        # Add bools to check if loaded -- if bool(self.waves..)
        wave = self.waves.get(vidpath)[time_half - 1] # Convert to time for waves
        ms = self.ms.get(vidpath)[time_half - 1] # Convert to time for waves
        resnet_features = self.resnet.get(vidpath)[time_half - 1]
        
        # ,seconds_spot,window_size_seconds,duration_seconds,sr=2,window_position=None):
        wave_idx = self._get_wave_idx_for_sample(seconds_spot = seconds_spot,duration_seconds=duration,window_size_seconds=self.window_size) # Consider addition of window options
        ms_idx = self._get_ms_idx_for_sample(seconds_spot = seconds_spot,duration_seconds=duration,window_size_seconds=self.window_size)
        resnet_idx = self._get_resnet_idx_for_sample(seconds_spot = seconds_spot,duration_seconds=duration,window_size_seconds=self.window_size)
        # todo
        mfcc_idx = None

        #wave_spot = wave[wave_idx[0]:wave_idx[1]]
        ms_spot = ms[1:,ms_idx[0]:ms_idx[1]]
        resnet_spot = resnet_features[resnet_idx[0]:resnet_idx[1],:]

        # normalize
        ms_spot = ( ms_spot - torch.min(ms_spot) ) / ( torch.max(ms_spot) - torch.min(ms_spot) )
        # Todo:
        # pointer to generated mel_sepctrogram and mfcc's
        # generate mel_spectrogram time conversion, and mfcc.
        # Test for classification
        # Add window capability
        # Add window forward or backwards
        # use for state of the art
        #print(wave_spot.size())

        sample = {'vidpath': vidpath,
                  'annotation':annotation,
                  'duration:': duration,
                  'label':label,
                  'lang':self.samples[idx][3],
                  'idx':idx,
                  'ms_idx':ms_idx, 'ms_spot':ms_spot,
                  'resnet_idx': resnet_idx,'resnet_spot':resnet_spot}
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
            for path in self.lang_sub_samples:
                half1 = np.load(self.root_dir+path+"/1_wave.npy")
                half2 = np.load(self.root_dir+path+"/2_wave.npy")
                self.waves[path] = (half1,half2)
        else:
            print("Already loaded!")
    
    def load_resnet_features(self):
        if self.lang_sub_samples:
            for path in self.lang_sub_samples:
                half1 = np.load(self.root_dir+path+"/1_ResNET_PCA512.npy")
                half2 = np.load(self.root_dir+path+"/2_ResNET_PCA512.npy")
                self.resnet[path] = (half1,half2)
        else:
            print("Already loaded!")
            
    def generate_mel_spectrograms(self, sr=16000, win_length=4,n_fft=512,step_size=0.025):
        if not bool(self.ms):

            SAMPLE_RATE = sr # Should be 16 kHz
            WINDOW_LENGTH = int(0.025 * sr) # Should be 25 ms
            N_FFT = n_fft # Should be 512
            STEP_SIZE = WINDOW_LENGTH # Should be 10ms
            
            for key in self.waves:
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
                self.ms[key] = (ms1,ms2)
        else:
            print("Already loaded")
        
    def _get_wave_idx_for_sample(self,seconds_spot,window_size_seconds,duration_seconds,sr=16000,window_position=None):
        
        anchor = seconds_spot * sr
        
        shift = (window_size_seconds * sr) // 2

        # If anchor is at start 
        if shift > anchor:
            start = 0
            end = start + (shift * 2)
        
            # if anchor is at the end
        elif (anchor + shift) >= ((duration_seconds * sr) - 1):
            end = ((duration_seconds * sr) - 1)
            start = end - (shift * 2)
        
        else:
            start = anchor - shift
            end = anchor + shift
        
        return (start,end)
        
        
    def _get_ms_idx_for_sample(self,seconds_spot,window_size_seconds,duration_seconds,sr=40,window_position=None):
        
        anchor = seconds_spot * sr
        
        shift = (window_size_seconds * sr) // 2
        
        # If anchor is at start 
        if shift > anchor:
            start = 0
            end = start + (shift * 2)
        
            # if anchor is at the end
        elif (anchor + shift) >= ((duration_seconds * sr) - 1):
            end = ((duration_seconds * sr) - 1)
            start = end - (shift * 2)
        
        else:
            start = anchor - shift
            end = anchor + shift
        
        return (start,end)
        
    def _get_resnet_idx_for_sample(self,seconds_spot,window_size_seconds,duration_seconds,sr=2,window_position=None):
        
        anchor = seconds_spot * sr
        
        shift = (window_size_seconds * sr) // 2
        
        # If anchor is at start 
        if shift > anchor:
            start = 0
            end = start + (shift * 2)
        
            # if anchor is at the end
        elif (anchor + shift) >= ((duration_seconds * sr) - 1):
            end = ((duration_seconds * sr) - 1)
            start = end - (shift * 2)
        
        else:
            start = anchor - shift
            end = anchor + shift
        
        return (start,end)
        

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
        print(f"\n npy_file: {self.npy_file} \n tshift: {self.tshift} \n root_dir: {self.root_dir} \n transform: {self.transform} \n frame_center: {self.frame_center} \n nframes: {self.nframes} \n stride_frames: {self.stride_frames} \n background: {self.background}")
        print("\n\n ********* End of description *********")