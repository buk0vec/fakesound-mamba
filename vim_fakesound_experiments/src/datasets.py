import torch
from PIL import Image
import os
import pandas as pd
import json
import librosa
import torchaudio
try:
    import s3fs
    fs = s3fs.S3FileSystem()
except ImportError:
    pass

# torch.multiprocessing.set_sharing_strategy("file_system")

class ImagenetteDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, split="train", transform=None):
        self.data = pd.read_csv(f"{root_dir}/noisy_imagenette.csv")
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        self.data = self.data[self.data['path'].str.startswith(f"{split}/")].reset_index(drop=True)

        self.label_map = {l: i for i, l in zip(range(10), self.data['noisy_labels_0'].unique())}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0]) 
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        label = self.data.iloc[idx, 1]
        one_hot_label = torch.zeros(10)
        one_hot_label[self.label_map[label]] = 1.0
        
        return image, one_hot_label

class ReconstructedFakeSoundDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, split="train", augment=None):
        self.split = split
        self.data_dir = root_dir
        # with fs.open(f"{data_dir}/meta_data/{split}_reconstructed.json") as m:
        with open(f"{self.data_dir}/meta_data/{split}_reconstructed.json") as m:
            d = json.load(m)
            self.metadata = d["audios"]
        self.transform = transform
        self.precision = 0.01
        
        self.augment = augment

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        m = self.metadata[idx]
        filepath = m["filepath"]
        label = m["label"]
        filepath = f"{self.data_dir}/{filepath[25:]}"
        # audio, sample_rate = librosa.load(filepath, sr=32000)
        audio, sample_rate = torchaudio.load(filepath)
        if not torch.isfinite(audio).all():
            raise ValueError(f"Audio {filepath} contains inf")
        if sample_rate != 32000:
            # print(f"WARNING - audio {m['audio_id']} resampled from {sample_rate} to 32khz")
            audio = torchaudio.functional.resample(audio, sample_rate, 32000)
            sample_rate = 32000
        onset, offset = (float(x) for x in m["onset_offset"].split("_"))
        segment = torch.zeros(int(10 / self.precision))
        segment[int(onset // self.precision) : int(offset//self.precision)] = 1.0
        if self.transform:
            # audio = self.transform(torch.tensor(audio.unsqueeze(0)), sample_rate, **self.transform_args)
            audio = self.transform(audio, sample_rate)

        if self.augment:
            audio = self.augment(audio)
            
        # print(f"{audio_name} {audio_path} {label}")
        # print(f"Audio shape: {audio.shape}")
        return (audio, torch.tensor(label, dtype=torch.float32), segment)


class ReconstructedFakeSoundDatasetS3(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, split="train", augment=None):
        self.split = split
        self.data_dir = root_dir
        with fs.open(f"{self.data_dir}/meta_data/{split}_reconstructed.json") as m:
        # with open(f"{self.data_dir}/meta_data/{split}_reconstructed.json") as m:
            d = json.load(m)
            self.metadata = d["audios"]
        self.transform = transform
        self.precision = 0.01
        
        self.augment = augment

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        m = self.metadata[idx]
        filepath = m["filepath"]
        label = m["label"]
        filepath = f"{self.data_dir}/{filepath[25:]}"
        # audio, sample_rate = librosa.load(filepath, sr=32000)
        with fs.open(filepath) as d:
            audio, sample_rate = torchaudio.load(d)
        if not torch.isfinite(audio).all():
            raise ValueError(f"Audio {filepath} contains inf")
        if sample_rate != 32000:
            # print(f"WARNING - audio {m['audio_id']} resampled from {sample_rate} to 32khz")
            audio = torchaudio.functional.resample(audio, sample_rate, 32000)
            sample_rate = 32000
        onset, offset = (float(x) for x in m["onset_offset"].split("_"))
        segment = torch.zeros(int(10 / self.precision))
        segment[int(onset // self.precision) : int(offset//self.precision)] = 1.0
        if self.transform:
            # audio = self.transform(torch.tensor(audio.unsqueeze(0)), sample_rate, **self.transform_args)
            audio = self.transform(audio, sample_rate)

        if self.augment:
            audio = self.augment(audio)
            
        # print(f"{audio_name} {audio_path} {label}")
        # print(f"Audio shape: {audio.shape}")
        return (audio, torch.tensor(label, dtype=torch.float32), segment)
