# coding: utf-8
import os
import numpy as np
import pandas as pd
from torch.utils import data
import random
import soundfile as sf
import librosa

def gt_label_founder():
    labels = pd.read_csv('../../groundtruth.csv' )
    labels.loc[ : , ['PHQ_Binary']]
    y = []
    for i in range( labels.shape[0] ):
        name =  labels.loc[ i , ['Participant_ID']].values[0]

        
        degree =  labels.loc[ i , ['PHQ_Binary']].values[0]
        
        if degree == 0:
            y.append( [ str(name)+'_P' , 'normal' ] )
           
        else: # is_hearning_loss == 1
            y.append( [ str(name)+'_P' , 'depression'])
        
        #degree =  labels.loc[ i , ['PHQ_Score']].values[0]
        
        #if degree < 6:
        #    y.append( [ str(name)+'_P' , 'normal' ] )
            
        #else: # degree < 24
        #    y.append( [ str(name)+'_P' , 'depression'])
        
        #degree =  labels.loc[ i , ['PHQ_Score']].values[0]
        
        #if degree < 3:
        #    y.append( [ str(name)+'_P' , 'normal' ] )
            
        #elif degree < 9:
        #    y.append( [ str(name)+'_P' , 'mild_dep'])
            
        #else: # degree < 24
        #    y.append( [ str(name)+'_P' , 'severe_dep'])
            
    y = pd.DataFrame( y , columns = ['name' , 'degree' ] )
    
    return y


class EDAIC(data.Dataset):
    def __init__(self, root, split, input_length=None):
        split = split.lower()
        self.mappeing = {
            "normal": 0,
            "depression": 1,
            #"mild_dep": 1,
            #"severe_dep": 2,
        }
        self.files = [
            f
            for f in open(f"{root}/{split}_filtered.txt", "r").readlines()
            #if "D:/data set/DAICWOZ/download_EDAIC/300_P/300_AUDIO.wav" not in f
        ]
        ### Need to change
        self.class_num = 2
        self.split = split
        self.seg_length = input_length
        self.root = root

    def __len__(self):
        if self.split == "train":
            return len(self.files)
        else:
            return len(self.files)

    def __getitem__(self, idx):
        #for idx in range( 0, len(self.files) ):
        if self.split == "train":
            idx = random.randint(0, len(self.files) - 1)
        file = self.files[idx].strip()
        frame = sf.info(file).frames
        gt = gt_label_founder()
        label = np.zeros(self.class_num)
        #n_class = gt[gt['name'] == file.split("/")[-2]].is_depression.values[0]
        #n_class = gt[gt['name'] == file.split("/")[-2]].degree.values[0]
        name = str(file.split("/")[-1][:3])+'_P'
        n_class = gt[gt['name'] == name].degree.values[0]
        #print(n_class)
        label[self.mappeing[n_class]] = 1
        #print(label)
        if self.split == "train":
            audio, sr = librosa.load(file, sr=16000)
            start = random.randint(0, len(audio) - self.seg_length*10 - 16000)
            #start = random.randint(0,10)
            #start = 0
            #audio = audio[start : start + self.seg_length]
            audio = audio.astype("float32")
            new_audio = audio[start : start + int(self.seg_length*1.2)]
            for i in range(1, 10):
                new_audio = np.vstack((new_audio, audio[start + self.seg_length*i : start + int(self.seg_length*(i+1.2))]))
            
            audio = new_audio.astype("float32")
            
            return audio, label.astype("float32")
        else:
            audio, sr = librosa.load(file, sr=16000)
            #start = random.randint(0, len(audio) - self.seg_length*10 - 16000)
            start = 0
            audio = audio[start : start + self.seg_length]
            audio = audio.astype("float32")
            #new_audio = audio[start : start + self.seg_length]

            #for i in range(1, 10):
            #    new_audio = np.vstack((new_audio, audio[start + self.seg_length*i : start + self.seg_length*(i+1)]))
            
            #audio = new_audio.astype("float32")
            n_chunk = len(audio) // self.seg_length
            audio_chunks = np.split(audio[: int(n_chunk * self.seg_length)], n_chunk)
            audio_chunks.append(audio[-int(self.seg_length) :])
            audio = np.array(audio_chunks)

            return audio, label.astype("float32")


def get_audio_loader(
    root,
    batch_size,
    split="TRAIN",
    num_workers=0,
    input_length=None
):
    data_loader = data.DataLoader(
        dataset=EDAIC(root, split=split, input_length=input_length),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    return data_loader
