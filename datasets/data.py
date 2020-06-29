from torch.utils.data import Dataset
from utils.preprocessing import json_to_dataframe
import cv2
import torch
import numpy as np
from torchvision import transforms


class DatasetRetriever(Dataset):

    def __init__(self, data_root, df, transform=None, preprocessing=None):
        self.df = df
        self.data_root = data_root
        self.transforms = transform
        self.preprocessing = preprocessing
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.df.shape[0]

    def onehot(self, label):
        encoded = torch.zeros(6)
        encoded[label] = 1.
        return encoded

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        image = cv2.imread(f"{self.data_root}/train/{sample['name']}.png")
        mask = cv2.imread(f"{self.data_root}/masks/{sample['name']}.png")
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask // 255
        mask = mask[:, :, np.newaxis]
        label = self.onehot(sample['label_encoded'])

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = torch.tensor(augmented['mask']).float().permute(2, 0, 1)
            # if self.preprocessing:
            #     preprocessed = self.preprocessing(image=image, mask=mask)
            #     image, mask = preprocessed['image'], preprocessed['mask']
        image = self.norm(image)
        return image, (mask, label)
