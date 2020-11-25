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



class SoccerNetDataset(Dataset):
    """Soccernet Dataset"""
    
    SAMPLE_RATE = 8000 # Should be 16 kHz
    WINDOW_LENGTH = int(0.025 * SAMPLE_RATE) # Should be 25 ms
    N_FFT = 512 # Should be 512
    STEP_SIZE = int(0.01 * SAMPLE_RATE) # Should be 10ms
    
    def __init__(self,npy_file,
                 root_dir,
                 transform=None,
                 background=False,
                 wsize=4):
    
        self.wsize = wsize
        self.npy_file = np.load(npy_file)
        self.samples = list() # maybe change structure later depending on efficiency        
        self.root_dir = root_dir
        self.transform = transform

        for e in self.npy_file:
            path, annotations = self.get_annotations(e)

            duration1 = self.getVideoLength(self.root_dir + e + "/1.mkv")
            duration2 = self.getVideoLength(self.root_dir + e + "/2.mkv")
            #print(f"duration1 : {duration1}, duration2: {duration2}")
            for annotation in annotations:
                # Check that annotations hold correct labels
                        if ("card" in annotation["label"]) or ("subs" in annotation["label"]) or ("soccer" in annotation["label"]):
                            annotation["duration1"] = duration1
                            annotation["duration2"] = duration2
                            self.samples.append([path,annotation])

       
    def __len__(self):
        return len(self.samples)
    
    def getVideoLength(self,video_file):
        res = Popen(['ffmpeg', '-i', video_file, '-hide_banner'],stdout=PIPE,stderr=PIPE)
        none,meta = res.communicate()
        meta_out = meta.decode()
        #---| Take out info
        duration = re.search(r'Duration:.*', meta_out)
        return duration.group()[:21]

    def __getitem__(self,idx):
        """Returns a sample containing video path, clip and label"""
        if torch.is_tensor(idx):
            idx.tolist()
        
        # get annotations
        time_half = int(self.samples[idx][1]["gameTime"][0])
        time_minute = int(self.samples[idx][1]["gameTime"][-5:-3])
        time_second = int(self.samples[idx][1]["gameTime"][-2:])
        annotation = self.samples[idx][1]

        # Get label
        if ("card" in annotation["label"]): label = 0
        elif ("subs" in annotation["label"]): label = 1
        elif ("soccer" in annotation["label"]): label = 2
        elif ("background" in annotation["label"]): label = 3
        else: 
            print("Warning, label not compatible with set")
            return

        # Get audiopath
        audiopath = os.path.join(self.root_dir,
                              str(self.samples[idx][0]),
                                str(time_half)+"_audio.wav")
        

        
        one_hot_label = np.zeros(4)
        one_hot_label[label] = 1
        # Get video frames 
        
        # get start in second, use labeled time as center TODO: fix centerframe as keyframe and stride
        fps = 25.0 # assume fps = 25 for now, should be so
        start_sec = time_minute*60 + time_second
        end_sec = start_sec
        
        if start_sec == 0:
            end_sec += (1/fps) # possibly unstable solution
            
        end_sec = end_sec + self.wsize # might need to subtract 1/fps
        # Shift backwards to center around time but check that time > 0
        diff = (end_sec - start_sec) / 2 # TODO : Might result in bad precision
        temp_start_sec = start_sec - diff
        temp_end_sec = end_sec - diff

        # Only change as long as the shift operation doesnt shift out of bounds 
        if temp_start_sec >= 0:
            start_sec = temp_start_sec
            end_sec = temp_end_sec
        
        # Buffer to endsec incase of bad load
        end_sec = end_sec + 0.9 # loads more frames than needed, then reduced later
        """
        y, sr = librosa.load(audiopath,sr=SAMPLE_RATE,offset=start_sec,duration=self.wsize+1)
        
        print(len(y))
        y = y[:SAMPLE_RATE*4]
        
        ms = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE,
                                                n_fft=N_FFT,
                                                win_length=WINDOW_LENGTH,
                                                hop_length=STEP_SIZE)(torch.Tensor(y)).log10().unsqueeze(0)
        print(f"ms size = {ms.size()}")
        
        
        ms = ms[:,:,:401]
        if ms.size() != (1,128,401):
            print(f"ms size :{ms.size()}, using zeros instead ...")
            ms = torch.zeros((1,128,401))
        """
        
        sample = {'audiopath': audiopath, 'annotation':annotation,'idx':idx, 'one_hot_label':one_hot_label,'label':label}
        
        return sample
            
    def get_annotations(self,path):
        """ Reads json files and returns """
        with open(self.root_dir+path+"/Labels.json") as jsonfile:
            json_label = json.load(jsonfile)
        
        labels = [e for e in json_label['annotations']]
        
        return path,labels
    def get_keyframe(self,idx):
        if self.frame_center == 'back': return self.__getitem__(idx)['clip'][0,:,:,:]
        elif self.frame_center == 'center': return self.__getitem__(idx)['clip'][self.nframes//2,:,:,:]
        elif self.frame_center == 'front': return self.__getitem__(idx)['clip'][self.nframes-1,:,:,:]
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