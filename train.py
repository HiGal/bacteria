from models.trainer import Fitter
from models.factory import get_net
from utils.config import TrainGlobalConfig
from datasets.data import DatasetRetriever
from utils.preprocessing import json_to_dataframe
from augmentations.transforms import get_train_transforms, get_test_transforms
import torch


def run_training():
    net, preprocess_params = get_net(TrainGlobalConfig.backbone)
    device = 'cuda:0'  # define by yourself
    net.to(device).half()

    df = json_to_dataframe(TrainGlobalConfig.data_root)

    train_dataset = DatasetRetriever(TrainGlobalConfig.data_root, df[df['fold'] != 0], get_train_transforms(preprocess_params))
    val_dataset = DatasetRetriever(TrainGlobalConfig.data_root, df[df['fold'] == 0], get_test_transforms(preprocess_params))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TrainGlobalConfig.train_batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=TrainGlobalConfig.valid_batch_size,
        shuffle=False,
        num_workers=4
    )

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
    fitter.fit(train_loader, val_loader)


if __name__ == '__main__':
    run_training()
