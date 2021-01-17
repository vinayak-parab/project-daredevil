#test

import numpy as np
import torchvision.transforms._transforms_video as video_transform
import torch

class ReSize(object):
    def __init__(self,output_size,interpolation='bilinear'):
        self.output_size = output_size
        self.interpolation = interpolation
    
    def __call__(self,clip):
        c,t,h,w = clip.size()
        return video_transform.F.resize(clip,self.output_size,interpolation_mode=self.interpolation)