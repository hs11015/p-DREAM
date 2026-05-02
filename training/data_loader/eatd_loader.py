# coding: utf-8
import os
import numpy as np
import pandas as pd
from torch.utils import data
import random
import soundfile as sf
import librosa
import openpyxl


class EATD(data.Dataset):
    def __init__(self, root, split, input_length=None):
        split = split.lower()
        self.mappeing = {
            "Normal": 0,
            "Depression": 1
        }
        self.files = [
            f
            for f in open(f"{root}/{split}_aug_balanced_eatd.txt", "r", encoding='UTF-8').readlines()
        ]
        ### Need to change
        self.class_num = 2
        self.split = split
        self.seg_length = input_length
        self.root = root
        self.prompt_sec = 5
        self.prompt_len = self.prompt_sec * 16000

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
        gt = pd.read_csv('/data/DAICWOZ/EATD-Corpus/label_table.csv', encoding='utf-8')
        label = np.zeros(self.class_num)
        id = str(file.split("/")[-2])
        n_class = gt[gt['id'] == id].label.values[0]
        label[self.mappeing[n_class]] = 1
        #print(n_class)
        if self.split == "train":
            audio, sr = librosa.load(file, sr=16000)
            # if len(audio) < 1600: print(file)
            # start = random.randint(0, len(audio) - self.seg_length*11 - 160)
            #start = random.randint(0,11)
            #start = 0
            #audio = audio[start : start + self.seg_length]
            prompt = np.random.normal(0, 1, self.prompt_len)
            
            if self.prompt_sec != 0:
                if len(audio) < self.seg_length:
                    repeat_times = int(np.ceil(self.seg_length / len(audio)))
                    audio = np.tile(audio, repeat_times)[:self.seg_length]  # 잘라서 정확히 맞춤
                    audio = np.concatenate((prompt, audio), axis=0)
                elif len(audio) >= self.seg_length:
                    audio = audio[:self.seg_length]
                    audio = np.concatenate((prompt, audio), axis=0)
                # print(audio.shape)
            else:
                if len(audio) < self.seg_length:
                    repeat_times = int(np.ceil(self.seg_length / len(audio)))
                    audio = np.tile(audio, repeat_times)[:self.seg_length]  # 잘라서 정확히 맞춤
                elif len(audio) > self.seg_length:
                    audio = audio[:self.seg_length]

            audio = np.expand_dims(audio, axis=0)  # shape: (1, 160000)
            audio = audio.astype("float32")

            # new_audio = audio[start : start + self.seg_length]
            # for i in range(1, 11):
            #     new_audio = np.vstack((new_audio, audio[start + self.seg_length*i : start + self.seg_length*(i+1)]))
            
            # audio = new_audio.astype("float32")
            # print(audio.shape)
            # print(label)
            return audio, label.astype("float32")
        else:
            # print(file, id)
            audio, sr = librosa.load(file, sr=16000)
            #start = random.randint(0, len(audio) - self.seg_length*6 - 16000)
            # start = 0
            #audio = audio[start : start + self.seg_length]
            prompt = np.random.rand(self.prompt_len)
            
            if self.prompt_sec != 0:
                if len(audio) < self.seg_length:
                    repeat_times = int(np.ceil(self.seg_length / len(audio)))
                    audio = np.tile(audio, repeat_times)[:self.seg_length]  # 잘라서 정확히 맞춤
                    audio = np.concatenate((prompt, audio), axis=0)
                elif len(audio) >= self.seg_length:
                    audio = audio[:self.seg_length]
                    audio = np.concatenate((prompt, audio), axis=0)
                # print(audio.shape)
            else:
                if len(audio) < self.seg_length:
                    repeat_times = int(np.ceil(self.seg_length / len(audio)))
                    audio = np.tile(audio, repeat_times)[:self.seg_length]  # 잘라서 정확히 맞춤
                elif len(audio) > self.seg_length:
                    audio = audio[:self.seg_length]

            audio = np.expand_dims(audio, axis=0)  # shape: (1, 160000)
            audio = audio.astype("float32")
            # new_audio = audio[start : start + self.seg_length]
            # for i in range(1, 11):
            #     new_audio = np.vstack((new_audio, audio[start + self.seg_length*i : start + self.seg_length*(i+1)]))
            
            # audio = new_audio.astype("float32")
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
        dataset=EATD(root, split=split, input_length=input_length),
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
    )
    return data_loader
