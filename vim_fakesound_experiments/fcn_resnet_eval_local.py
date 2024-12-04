import torch
from torch import nn
import torch
import torchaudio
import torchvision
import pandas as pd
import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt
import torchaudio.transforms as T
import json
import torch.nn.functional as F
from torch import nn
import librosa
import s3fs

fs = s3fs.S3FileSystem()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FinetunedMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel_up = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
        self.backbone = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights='DEFAULT')
        self.backbone.classifier[4] = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
        
        self.classifier = nn.Linear(300 * 1000, 1)
        self.segmentation = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(300 * 1000, 1000), nn.ReLU(), nn.Dropout(p=0.3), nn.Linear(1000, 1000))

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.channel_up(x)
        x = self.backbone(x)['out']
        x = x.view(B, 1 * H * W)
        c = self.classifier(x)
        c = torch.sigmoid(c)
        s = self.segmentation(x)
        s = torch.sigmoid(s)
        return c, s

train_mean = 2.5853811548103334
train_std = 48.60284136954955
normalize_t = torchvision.transforms.Normalize(mean=train_mean, std=train_std)


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
    
mel_spectrogram_32khz = T.MelSpectrogram(
        sample_rate=32000,
        n_fft=2048,
        hop_length=320,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        n_mels=300,
        mel_scale="htk",
        f_min=0,
        f_max = 32000/2
    )

def transform(audio, sample_rate, n_fft=2048, hop_length=320, n_mels=280, win_length=None):
    if sample_rate == 32000:
        m = mel_spectrogram_32khz
    else:
        m = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        win_length=None,
        hop_length=320,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        n_mels=300,
        mel_scale="htk",
        f_min=0,
        f_max = sample_rate/2
    )
    # s = spectrogram(audio)
    s = m(audio)
    s = normalize_t(s)
    # this truncates the last 0.005 seconds, but idrc
    s = F.pad(s, (0, 1000 - s.shape[2]))
    # if s.shape != (1, 300, 2000):
    #     print("ruh roh")
    return s

model = FinetunedMobileNet()
model.load_state_dict(torch.load('mobilenet-baseline.pth', weights_only=True))
model.to(device)

model.eval()

easy_set = ReconstructedFakeSoundDataset('s3://bukovec-ml-data/FakeAudio', split='easy', transform=transform)
hard_set = ReconstructedFakeSoundDataset('s3://bukovec-ml-data/FakeAudio', split='hard', transform=transform)
zeroshot_set = ReconstructedFakeSoundDataset('s3://bukovec-ml-data/FakeAudio', split='zeroshot', transform=transform)

num_workers = 0
batch_size = 16

easy_loader = torch.utils.data.DataLoader(easy_set, num_workers=num_workers, batch_size=batch_size, shuffle=False)
hard_loader = torch.utils.data.DataLoader(hard_set, num_workers=num_workers, batch_size=batch_size, shuffle=False)
zeroshot_loader = torch.utils.data.DataLoader(zeroshot_set, num_workers=num_workers, batch_size=batch_size, shuffle=False)

loaders = [easy_loader, hard_loader, zeroshot_loader]
names = ['easy', 'hard', 'zeroshot']


for i in range(len(loaders)):
    loader = loaders[i]
    name = names[i]

    class_batches = np.zeros((len(loader.dataset), 1))
    pred_class_batches = np.zeros((len(loader.dataset), 1))
    seg_batches = np.zeros((len(loader.dataset), 1000))
    pred_seg_batches = np.zeros((len(loader.dataset), 1000))
    counter = 0
    for idx, batch in enumerate(loader):
        with torch.no_grad():
            audio, labels, segs, _ = batch
            B = labels.shape[0]
            audio = audio.to(device)
            labels_, segs_ = model(audio)
            class_batches[counter:counter+B] = labels.numpy()[:, np.newaxis]
            seg_batches[counter:counter+B] = segs.numpy()
            pred_class_batches[counter:counter+B] = labels_.cpu().numpy()
            pred_seg_batches[counter:counter+B] = segs_.cpu().numpy()
            counter += B

    np.savetxt(f"stats/baseline_{names[i]}_class_pred.csv", pred_class_batches, delimiter=",", fmt="%.8f")
    np.savetxt(f"stats/baseline_{names[i]}_class.csv", class_batches, delimiter=",", fmt="%.8f")
    np.savetxt(f"stats/baseline_{names[i]}_seg_pred.csv", pred_seg_batches, delimiter=",", fmt="%.8f")
    np.savetxt(f"stats/baseline_{names[i]}_seg.csv", seg_batches, delimiter=",", fmt="%.8f")
        
        