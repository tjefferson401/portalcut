# kitti_dataset.py
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2

class KITTIDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = self.img_labels.iloc[idx, 2:6].astype('float').values.reshape(-1, 4)
        labels = torch.tensor(self.img_labels.iloc[idx, 1]).unsqueeze(0)

        target = {'boxes': torch.tensor(boxes, dtype=torch.float32), 'labels': labels}

        if self.transform:
            image = self.transform(image=image)['image']
        
        return image, target
