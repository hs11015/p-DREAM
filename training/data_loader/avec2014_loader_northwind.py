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
        # self.prompt_sec = 1
        # self.prompt_sec = 2.5
        # self.prompt_sec = 5
        # self.prompt_sec = 10
        
        self.prompt_len = int(self.prompt_sec * 1600)
        
        # Step 1: 파일 리스트 수집 (Northwind only)
        audio_dir = os.path.join(root, self.split, "Northwind")
        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")])

        # Step 2: 라벨 로딩 및 participant ID 매핑
        label_path = os.path.join(root, "label", f"{self.split}_label.mat")
        self.label_dict = self._load_labels(label_path, self.audio_files)

    def _load_labels(self, mat_path, audio_files):
        mat = scipy.io.loadmat(mat_path)
        bdi_scores = mat["data"].flatten()  # shape: (N,)
        
        # Northwind 폴더 내 파일 이름 기준으로 PID 추출
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
        pid = filename[0:5]  # e.g., '203_1'

        label_val = self.label_dict[pid]
        label = np.array([label_val], dtype=np.float32)

        audio, sr = librosa.load(os.path.join(self.root, self.split, "Northwind", file), sr=16000)

        chunks = []
        if self.prompt_sec != 0:
            prompt = np.random.normal(0, 1, self.prompt_len)

            # 오디오 길이 = seg_len - prompt*2*5
            audio_len = self.seg_length
            chunk_len = audio_len // 5

            if len(audio) < audio_len:
                repeat = int(np.ceil(audio_len / len(audio)))
                audio = np.tile(audio, repeat)[:audio_len]

            for i in range(5):
                start = i * chunk_len
                end = (i + 1) * chunk_len
                core = audio[start:end]
                chunk = np.concatenate((prompt, core, prompt), axis=0)  # total = chunk_len + 2*prompt_len
                chunks.append(chunk)

            # Now truncate/pad each to same length if needed (safety)
            chunks = [c[:chunk_len + 2 * self.prompt_len] for c in chunks]
            audio = np.stack(chunks)  # shape = (5, chunk_len + 2*prompt_len)

        else:
            chunk_len = self.seg_length // 5
            if len(audio) < self.seg_length:
                repeat = int(np.ceil(self.seg_length / len(audio)))
                audio = np.tile(audio, repeat)[:self.seg_length]

            for i in range(5):
                start = i * chunk_len
                end = (i + 1) * chunk_len
                chunks.append(audio[start:end])

            audio = np.stack(chunks)  # shape = (5, chunk_len)

        return audio.astype("float32"), label[0]



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
