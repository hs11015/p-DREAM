# coding: utf-8
import os
import numpy as np
import librosa
import scipy.io
from glob import glob
from torch.utils import data

class AVEC2014Dataset(data.Dataset):
    def __init__(self, root="/data/DAICWOZ/AVEC2014", split=None, input_length=None, audio_ext="wav"):
        self.root = root
        self.split = split.lower()
        self.input_length = input_length
        self.seg_length = input_length
        self.audio_ext = audio_ext
        self.class_num = 1  # Regression이므로 label shape은 (1,)
        self.prompt_sec = 0
        self.prompt_len = self.prompt_sec * 16000
        
        # Step 1: 파일 리스트 수집 (Freeform only)
        audio_dir = os.path.join(root, self.split, "Freeform")
        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")])

        # Step 2: 라벨 로딩 및 participant ID 매핑
        label_path = os.path.join(root, "label", f"{self.split}_label.mat")
        self.label_dict = self._load_labels(label_path, self.audio_files)

    def _load_labels(self, mat_path, audio_files):
        mat = scipy.io.loadmat(mat_path)
        bdi_scores = mat["data"].flatten()  # shape: (N,)
        
        # Freeform 폴더 내 파일 이름 기준으로 PID 추출
        pid_list = [os.path.basename(path)[0:5] for path in audio_files]
        
        # print(bdi_scores)
        # print(len(bdi_scores))
        # print(pid_list)
        # print(len(pid_list))
        
        assert len(pid_list) == len(bdi_scores), "Mismatch between number of files and labels"
        
        labels = {pid: float(score) for pid, score in zip(pid_list, bdi_scores)}
        return labels

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        file = self.audio_files[idx]
        filename = os.path.basename(file)
        pid = filename[0:5]  # '203_1_Freeform.wav' → '203_1'

        label_val = self.label_dict[pid]
        label = np.array([label_val], dtype=np.float32)  # Regression label

        audio, sr = librosa.load(os.path.join(self.root, self.split, "Freeform", file), sr=16000)
        prompt = np.random.normal(0, 1, self.prompt_len)
        
        if self.prompt_sec != 0:
            if len(audio) < self.seg_length:
                repeat_times = int(np.ceil(self.seg_length / len(audio)))
                audio = np.tile(audio, repeat_times)[:self.seg_length]  # 잘라서 정확히 맞춤
                audio = np.concatenate((prompt, audio), axis=0)
                audio = np.concatenate((audio, prompt), axis=0)
            elif len(audio) >= self.seg_length:
                audio = audio[:self.seg_length]
                audio = np.concatenate((prompt, audio), axis=0)
                audio = np.concatenate((audio, prompt), axis=0)
            # print(audio.shape)
        else:
            if len(audio) < self.seg_length:
                repeat_times = int(np.ceil(self.seg_length / len(audio)))
                audio = np.tile(audio, repeat_times)[:self.seg_length]  # 잘라서 정확히 맞춤
            elif len(audio) > self.seg_length:
                audio = audio[:self.seg_length]

        audio = np.expand_dims(audio, axis=0)  # shape: (1, 160000)
        audio = audio.astype("float32")
        return audio, label[0]

def get_audio_loader(
    root="/data/DAICWOZ/AVEC2014",
    batch_size=16,
    split="train",
    num_workers=0,
    input_length=160000,
    audio_ext="wav"
):
    return data.DataLoader(
        dataset=AVEC2014Dataset(
            root=root,
            split=split,
            input_length=input_length,
            audio_ext=audio_ext
        ),
        batch_size=batch_size,
        shuffle=True if split == "train" else False,
        drop_last=True,
        num_workers=num_workers,
    )
