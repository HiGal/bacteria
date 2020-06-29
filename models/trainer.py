import os
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils.train import Epoch
from segmentation_models_pytorch.utils.meter import AverageValueMeter
import sys
from tqdm import tqdm


class TrainClassEpoch(Epoch):

    def __init__(self, model, seg_loss, cls_loss, metrics, optimizer, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=seg_loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.cls_loss = cls_loss
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction_mask, y_pred = self.model.forward(x)
        loss = 0.5*(self.loss(prediction_mask, y[0]) + self.cls_loss(y_pred, y[1]))
        loss.backward()
        self.optimizer.step()
        return loss, prediction_mask, y_pred

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), (y[0].to(self.device),y[1].to(self.device))
                loss, mask_pred, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(mask_pred, y[0]).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class ValidClassEpoch(Epoch):

    def __init__(self, model, seg_loss, cls_loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=seg_loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )
        self.cls_loss = cls_loss

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction_mask, y_pred = self.model.forward(x)
            loss = 0.5*(self.loss(prediction_mask, y[0]) + self.cls_loss(y_pred, y[1]))
        return loss, prediction_mask, y_pred

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), (y[0].to(self.device),y[1].to(self.device))
                loss, mask_pred, y_pred = self.batch_update(x, y)

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(mask_pred, y[0]).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs

class Fitter:

    def __init__(self, model, device, config):
        self.config = config
        self.epoch = 0
        self.model = model
        self.device = device

    def fit(self, train_loader, val_loader):
        metrics = [
            smp.utils.metrics.IoU(threshold=0.5)
        ]
        optimizer = torch.optim.Adam([
            dict(params=self.model.parameters(), lr=self.config.lr),
        ])

        loss = smp.utils.losses.DiceLoss()
        cls_loss = torch.nn.BCELoss()
        train_epoch = TrainClassEpoch(
            self.model,
            seg_loss=loss,
            cls_loss=cls_loss,
            metrics=metrics,
            optimizer=optimizer,
            device=self.device,
            verbose=True,
        )

        valid_epoch = ValidClassEpoch(
            self.model,
            seg_loss=loss,
            cls_loss=cls_loss,
            metrics=metrics,
            device=self.device,
            verbose=True,
        )

        max_score = 0

        for i in range(0, self.config.epochs):

            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(val_loader)

            # do something (save model, change lr, etc.)
            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(self.model, './best_model.pth')
                print('Model saved!')

            if i == 25:
                optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')
