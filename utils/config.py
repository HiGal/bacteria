import torch

class TrainGlobalConfig:
    lr = 1e-3
    epochs = 40
    backbone = "timm-efficientnet-b6"
    data_root = "data"
    scheduler_step = True
    train_batch_size = 6
    valid_batch_size = 4
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True,
        threshold=0.0001,
        threshold_mode='abs',
        cooldown=0,
        min_lr=1e-8,
        eps=1e-08
    )
    pass


class TestGlobalConfig:
    pass
