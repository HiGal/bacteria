from models.trainer import Fitter
from models.factory import get_net
from utils.config import TrainGlobalConfig
from datasets.data import DatasetRetriever
import torch


def run_training():
    net = get_net()
    device = None  # define by yourself
    net.to(device)

    train_dataset = DatasetRetriever()
    val_dataset = DatasetRetriever()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        # and so on
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        # and so on
    )

    fitter = Fitter(model=net, device=device, config=TrainGlobalConfig)
    fitter.fit()


if __name__ == '__main__':
    run_training()
