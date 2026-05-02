# coding: utf-8
import os
import numpy as np
import pandas as pd
from torch.utils import data
import random
import soundfile as sf
import librosa
import openpyxl

def gt_label_founder():
    labels = pd.read_excel('D:/hyunseo/hearingloss/최종 데이터 정리 파일 (data balancing1).xlsx' , sheet_name = 0)
    labels.loc[ : , ['주진단']]
    y = []
    for i in range( labels.shape[0] ):
        name =  labels.loc[ i , ['폴더 이름']].values[0]
        is_hearing_loss =  labels.loc[ i , ['주진단']].values[0]
        degree =  labels.loc[ i , ['주진단정도']].values[0]
        #나이 넣고싶으면 다시 VGGish 코드 참고하기
        if is_hearing_loss == '정상':
            y.append([name, 'normal', 'normal'])
        
        elif is_hearing_loss != '혼합난청':
            if degree == '경도':
                y.append( [name , 'hearingloss', 'mild'])
            elif degree == '중등도':
                y.append( [name , 'hearingloss', 'moderate'] )
            elif degree =='고도':
                y.append( [name , 'hearingloss', 'moderate'])
        
    y = pd.DataFrame( y , columns = ['name' , 'is_hearing_loss', 'degree' ] )
    return y


class HearingLoss(data.Dataset):
    def __init__(self, root, split, input_length=None):
        split = split.lower()
        self.mappeing = {
            "normal": 0,
            #"hearingloss": 1,
            "mild": 1,
            "moderate": 2,
        }
        self.files = [
            f
            for f in open(f"{root}/{split}_filtered_hearingloss.txt", "r", encoding='UTF-8').readlines()
        ]
        ### Need to change
        self.class_num = 3
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
        if os.path.exists(file)==False: print(file)
        frame = sf.info(file).frames
        #print()
        #print(str(file.split("/")[-2])+str(file.split("/")[-1]))
        gt = gt_label_founder()
        label = np.zeros(self.class_num)
        name = str(file.split("/")[-2])
        #n_class = gt[gt['name'] == name].is_hearing_loss.values[0]
        n_class = gt[gt['name'] == name].degree.values[0]
        label[self.mappeing[n_class]] = 1
        #print(n_class)
        if self.split == "train":
            audio, sr = librosa.load(file, sr=16000)
            if len(audio) < 200000: print(file)
            start = random.randint(0, len(audio) - self.seg_length*11 - 16000)
            #start = random.randint(0,11)
            #start = 0
            #audio = audio[start : start + self.seg_length]
            audio = audio.astype("float32")
            new_audio = audio[start : start + self.seg_length]
            for i in range(1, 11):
                new_audio = np.vstack((new_audio, audio[start + self.seg_length*i : start + self.seg_length*(i+1)]))
            
            audio = new_audio.astype("float32")
            
            return audio, label.astype("float32")
        else:
            audio, sr = librosa.load(file, sr=16000)
            #start = random.randint(0, len(audio) - self.seg_length*6 - 16000)
            start = 0
            #audio = audio[start : start + self.seg_length]
            audio = audio.astype("float32")
            new_audio = audio[start : start + self.seg_length]
            for i in range(1, 11):
                new_audio = np.vstack((new_audio, audio[start + self.seg_length*i : start + self.seg_length*(i+1)]))
            
            audio = new_audio.astype("float32")
            #n_chunk = len(audio) // self.seg_length
            #audio_chunks = np.split(audio[: int(n_chunk * self.seg_length)], n_chunk)
            #audio_chunks.append(audio[-int(self.seg_length) :])
            #audio = np.array(audio_chunks)

            return audio, label.astype("float32")


def get_audio_loader(
    root,
    batch_size,
    split="TRAIN",
    num_workers=0,
    input_length=None
):
    data_loader = data.DataLoader(
        dataset=HearingLoss(root, split=split, input_length=input_length),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )
    return data_loader
