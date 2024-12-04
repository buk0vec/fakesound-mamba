import torch
import torchaudio
import numpy as np
import glob
import os
import re
import warnings
import torchaudio.transforms as T
import torch.nn.functional as F
from torch import nn
# import librosa
# from torch.utils.tensorboard import SummaryWriter
import json
import torchvision
import lightning as L
import librosa
import s3fs
import argparse
from lightning.pytorch.loggers import TensorBoardLogger
from PIL import Image
import pandas as pd
from transforms import transform

fs = s3fs.S3FileSystem()

BATCH_SIZE = 1

warnings.simplefilter(action='ignore', category=FutureWarning)
# from Vim.vim.models_mamba import VisionMamba

class ReconstructedFakeSoundDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, split="train", **transform_args):
        self.split = split
        self.data_dir = data_dir
        with fs.open(f"{data_dir}/meta_data/{split}_reconstructed.json") as m:
            d = json.load(m)
            self.metadata = d["audios"]
        self.transform = transform
        self.precision = 0.01
        
        self.transform_args = transform_args

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        m = self.metadata[idx]
        filepath = m["filepath"]
        label = m["label"]
        filepath = f"s3://bukovec-ml-data/FakeAudio/{filepath[25:]}"
        audio = None
        sample_rate = None
        # with fs.open(filepath) as d:
        #     audio, sample_rate = torchaudio.load(d)
        #     if sample_rate != 32000:
        #         # print(f"WARNING - audio {m['audio_id']} resampled from {sample_rate} to 32khz")
        #         audio = torchaudio.functional.resample(audio, sample_rate, 32000)
        #         sample_rate = 32000
        with fs.open(filepath) as d:
            audio, sample_rate = librosa.load(d, sr=32000)
        onset, offset = (float(x) for x in m["onset_offset"].split("_"))
        segment = torch.zeros(int(10 / self.precision))
        segment[int(onset / self.precision): int(offset/self.precision)] = 1.0
        if self.transform:
            audio = self.transform(torch.tensor(audio).unsqueeze(0), sample_rate, **self.transform_args)
        # print(f"{audio_name} {audio_path} {label}")
        return (audio, torch.tensor(label, dtype=torch.float32), segment, {
            "onset": onset,
            "offset": offset,
            "audio_id": m["audio_id"]
        })
    
easy_set = ReconstructedFakeSoundDataset('s3://bukovec-ml-data/FakeAudio', split='train', transform=transform)
# hard_set = ReconstructedFakeSoundDataset('s3://bukovec-ml-data/FakeAudio', split='hard', transform=transform)
# val_set = torch.utils.data.ConcatDataset([easy_set, hard_set])
num_pos = 0
num_neg = 0

for d in easy_set:
    _, _, s, _, = d
    num_pos += torch.sum(s == 1).item()
    num_neg += torch.sum(s == 0).item()

print(f"num_pos: {num_pos}")
print(f"num_neg: {num_neg}")
print(f"ratio: {num_pos / num_neg}")
