import torch
from PIL import Image
import os
import pandas as pd

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
