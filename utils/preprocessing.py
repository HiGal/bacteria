import json
from glob import glob
import pandas as pd
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import StratifiedKFold


def json_to_dataframe(data_root):
    data_pd = []
    for path in glob(f"{data_root}/train/*.json"):
        with open(path, "r") as f:
            data = json.load(f)
        name = (path.split("/")[-1]).split(".")[0]
        label = data["shapes"][0]['label']
        data_pd.append({"name": name, "label": label})

    df = pd.DataFrame(data_pd)
    df['label_encoded'] = df['label'].factorize()[0]
    skf = StratifiedKFold()
    for fold_number, (train_index, val_index) in enumerate(skf.split(X=df.index, y=df['label'])):
        df.loc[df.iloc[val_index].index, 'fold'] = fold_number
    return df


def polygon_to_mask(data_root):
    if not os.path.exists(f"{data_root}/masks"):
        os.mkdir(f"{data_root}/masks")
    for path in glob(f"{data_root}/train/*.json"):
        with open(path, "r") as f:
            layout = json.load(f)
        h, w = layout['imageHeight'], layout['imageWidth']
        true_mask = np.zeros((h, w), np.uint8)
        name = (path.split("/")[-1]).split(".")[0]
        for shape in layout['shapes']:
            polygon = np.array([point for point in shape['points']])
            cv2.fillPoly(true_mask, [polygon], 255)
        cv2.imwrite(f"{data_root}/masks/{name}.png", true_mask)


def visualize(img, mask):
    if isinstance(img, torch.Tensor):
        img = img.permute(1, 2, 0).detach().cpu().numpy()

    if isinstance(mask, torch.Tensor):
        mask = mask.permute(1, 2, 0).squeeze(-1).detach().cpu().numpy()

    plt.figure(figsize=(15,5))
    plt.subplot('121')
    plt.imshow(img)
    plt.subplot('122')
    plt.imshow(mask)
    plt.show()
