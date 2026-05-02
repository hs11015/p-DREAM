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

        
        #degree =  labels.loc[ i , ['PHQ_Binary']].values[0]
        
        #if degree == 0:
        #    y.append( [ str(name)+'_P' , 'normal' ] )
           
        #else: # is_hearning_loss == 1
        #    y.append( [ str(name)+'_P' , 'depression'])
        
        
        degree =  labels.loc[ i , ['PHQ_Score']].values[0]
        
        if degree < 10:
            y.append( [ str(name)+'_P' , 'normal' ] )
            
        else: # degree < 24
            y.append( [ str(name)+'_P' , 'depression'])
        
        
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
            for f in open(f"{root}/{split}_daicwoz_augmentation_123.txt").readlines()
            #if "D:/data set/DAICWOZ/download_EDAIC/300_P/300_AUDIO.wav" not in f
        ]
        ### Need to change
        #self.new_audio_left = np.zeros(0)
        #self.new_audio_left = np.random.seed(160)
        self.new_audio_left = np.random.rand(40000)
        #self.new_audio_left[:] -= 1.0
        #self.new_audio_right = np.zeros(0)
        #self.new_audio_right = np.random.seed(1600)
        self.new_audio_right = np.random.rand(40000)
        
        '''
        f = open('./random_both_noise.txt', 'w')
        print("LEFT", file=f)
        #print(self.new_audio_left, file=f)
        print("[", end=' ', file=f)
        for i in range(40000):
            print(f"{self.new_audio_left[i]}", end=' ', file=f)
        print("]", file=f)
        
        print("RIGHT", file=f)
        print("[", end=' ', file=f)
        for i in range(40000):
            print(f"{self.new_audio_right[i]}", end=' ', file=f)
        print("]", file=f)
        #print(self.new_audio_right, file=f)
        f.close()
        '''
        
        #self.new_audio_right = np.hstack((self.new_audio_left, self.new_audio_right))
        #self.new_audio_left = np.zeros(0)
        #self.new_audio_right[:] -= 1.0
        ### whole noise
        self.new_audio = np.zeros(160000)
        #self.new_audio = np.zeros(240000)
        
        #self.new_audio_l = np.random.rand(40000)    
        #self.new_audio_m = np.ones(80000)
        #self.new_audio_m[:] -= 1.0
        #self.new_audio_m = np.zeros(80000)
        #self.new_audio_r = np.random.rand(40000)
        #self.new_audio = np.hstack((self.new_audio_l, self.new_audio_m))
        #self.new_audio = np.hstack((self.new_audio, self.new_audio_r))
        #self.new_audio[:] -= 2.0
        
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
        #print("left noise : ", self.new_audio_left)
        #print("right noise : ", self.new_audio_right)
        #print("over noise : ", self.new_audio)
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
        #print()
        if self.split == "train":
            #print(name)
            audio, sr = librosa.load(file, sr=16000)
            #start = (idx+4)%5*10
            start = (idx%10)*10
            
            #start = random.randint(0, len(audio) - self.seg_length*5 - 16000)
            #start = random.randint(0,5)
            #start = 0
            #audio = audio[start : start + self.seg_length]
            audio = audio.astype("float32")
            
            #print("original: ",audio.min(), audio.max())
            ### original
            _audio = audio[start : start + int(self.seg_length)]
            if len(_audio) != 160000:
                temp_length = 160000 - len(_audio)
                temp_audio = np.hstack((np.zeros(temp_length//2), _audio, np.zeros(temp_length-(temp_length//2))))
                _audio = temp_audio

            ### linear noise 추가
            #self.new_audio_left = np.repeat(audio[start], 40000)
            #self.new_audio_right = np.repeat(audio[start+int(self.seg_length)-1], 40000)
            
            
            ### 추가
            new_audio = np.hstack((self.new_audio_left, _audio + self.new_audio))
            new_audio = np.hstack((new_audio, self.new_audio_right))
            new_audio = np.expand_dims(new_audio, axis=0)
            #print(f"\nfile: {file}, index:{idx}, start: {start}, new audio: {new_audio.shape}")
            
            '''
            for i in range(1, 5):
                
                _temp = np.hstack((self.new_audio_left, (audio[start + int(self.seg_length*i) : start + int(self.seg_length*(i+1))] + self.new_audio)))
                _temp = np.hstack((_temp, self.new_audio_right))
                #print("temp", _temp.shape)
                ### original
                #new_audio = np.vstack((new_audio, audio[start + self.seg_length*i : start + int(self.seg_length*(i+1))]))
                ### 추가
                new_audio = np.vstack((new_audio, _temp))
                #print("new audio", new_audio.shape)
            '''  
            audio = new_audio.astype("float32")
            #print("noise: ",audio.min(), audio.max())
            
            
            '''
            new_audio = np.random.rand(8000)
            new_audio = np.hstack((new_audio, (audio[start : start + int(self.seg_length*1.2)])))
            new_audio = np.hstack((new_audio, np.random.rand(8000)))
            for i in range(1, 5):
                _temp = np.random.rand(8000)
                _temp = np.hstack((_temp, audio[start + self.seg_length*i : start + int(self.seg_length*(i+1.2))]))
                _temp = np.hstack((_temp, np.random.rand(8000)))
                new_audio = np.vstack((new_audio, _temp))
            
            audio = new_audio.astype("float32")
            '''
            
            return audio, label.astype("float32")
        else:
            #print(name)
            audio, sr = librosa.load(file, sr=16000)
            start = random.randint(0, len(audio) - self.seg_length*5 - 16000)
            #start = 0
            #audio = audio[start : start + self.seg_length]
            #audio = audio.astype("float32")
            
            audio = audio.astype("float32")
            
            #print("original: ",audio.min(), audio.max())
            ### original
            #new_audio = audio[start : start + int(self.seg_length)]
            
    
            ### linear noise 추가
            #self.new_audio_left = np.repeat(audio[start], 40000)
            #self.new_audio_right = np.repeat(audio[start+int(self.seg_length)-1], 40000)
            
            ### 추가
            new_audio = np.hstack((self.new_audio_left, (audio[start : start + int(self.seg_length)] + self.new_audio)))
            new_audio = np.hstack((new_audio, self.new_audio_right))
            
            for i in range(1, 5):
                    
                _temp = np.hstack((self.new_audio_left, (audio[start + self.seg_length*i : start + int(self.seg_length*(i+1))] + self.new_audio)))
                _temp = np.hstack((_temp, self.new_audio_right))
                ### original
                #new_audio = np.vstack((new_audio, audio[start + self.seg_length*i : start + int(self.seg_length*(i+1))]))
                ### 추가
                new_audio = np.vstack((new_audio, _temp))
            
            audio = new_audio.astype("float32")
            #print("noise: ",audio.min(), audio.max())
            
            '''
            new_audio = np.random.rand(8000)
            new_audio = np.hstack((new_audio, (audio[start : start + int(self.seg_length*1.2)])))
            new_audio = np.hstack((new_audio, np.random.rand(8000)))
            for i in range(1, 5):
                _temp = np.random.rand(8000)
                _temp = np.hstack((_temp, audio[start + self.seg_length*i : start + int(self.seg_length*(i+1.2))]))
                _temp = np.hstack((_temp, np.random.rand(8000)))
                new_audio = np.vstack((new_audio, _temp))
            
            audio = new_audio.astype("float32")
            '''
            
            '''
            n_chunk = len(audio) // self.seg_length
            audio_chunks = np.split(audio[: int(n_chunk * self.seg_length)], n_chunk)
            audio_chunks.append(audio[-int(self.seg_length) :])
            audio = np.array(audio_chunks)
            '''
            
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
