import torchvision
from utils.preprocessing import json_to_dataframe
from augmentations.transforms import get_test_transforms
from datasets.data import TestDataset
from torch.utils.data import DataLoader
import pandas as pd
import torch
import base64
from io import BytesIO

if __name__ == '__main__':
    label2id = {0: "staphylococcus_epidermidis", 1: "ent_cloacae", 2: "staphylococcus_aureus", 3: "c_kefir",
                4: "moraxella_catarrhalis", 5: "klebsiella_pneumoniae"}

    model = torch.load("best_model.pth")
    model = model.cuda()

    transforms = get_test_transforms()
    df = json_to_dataframe("data", train=False)

    test_dataset = TestDataset("data", df)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    data_pd = []

    for idx, batch in enumerate(test_dataloader):
        pred_mask, label = model(batch.cuda())
        label = label2id[torch.argmax(label).item()]
        mask = torchvision.transforms.ToPILImage()(pred_mask[0].cpu())
        buff = BytesIO()
        mask.save(buff, format='PNG')
        new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
        name = df.iloc[idx]['name']
        data_pd.append({"id": name, "class": label, "base64 encoded PNG (mask)": new_image_string})

    submission = pd.DataFrame(data_pd)
    submission.to_csv("bacteria.csv",index=False)

