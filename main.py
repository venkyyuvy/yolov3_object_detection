import config
import torch
import torch.optim as optim
from torch_lr_finder import LRFinder
from pytorch_lightning import LightningModule, Trainer, Callback
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


from model import YOLOv3
from callbacks import (
    CheckClassAccuracyCallback,
    MAPCallback,
    PlotTestExamplesCallback,
)
from utils import get_loaders
from loss import YoloLoss


class YoloV3(LightningModule):
    def __init__(self, ):
        super().__init__()

        self.model = YOLOv3(num_classes=config.NUM_CLASSES)
        # todo
        #self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)


    def lr_finder(self, optimizer, criterion, 
        num_iter=50, 
    ):
        lr_finder = LRFinder(self, optimizer, criterion,
            device=self.device)
        lr_finder.range_test(
            self.train_dataloader(), end_lr=1,
            num_iter=num_iter, step_mode='exp',
            )
        ax, suggested_lr = lr_finder.plot(suggest_lr=True)
        lr_finder.reset() 
        return suggested_lr

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        lr = opt.param_groups[0]['lr']
        self.log('learning_rate', lr)
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)
        self.log(
            "train_loss", loss, prog_bar=True,
            logger=True, on_step=True, on_epoch=True
        )
        return loss

    def evaluate(self, batch, batch_idx, stage=None):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)
        self.log('val_loss', loss.item())

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, batch_idx, "val", )

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, batch_idx, "test", )

    @property
    def get_scaled_anchors(self):
        return (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1)\
                .unsqueeze(1).repeat(1, 3, 2)
        ).to(self.device)

    def criterion(self, out, y):
        loss = YoloLoss()
        scaled_anchors = self.get_scaled_anchors
        y0, y1, y2 = (y[0], y[1], y[2])
        return (
            loss(out[0], y0, scaled_anchors[0])
            + loss(out[1], y1, scaled_anchors[1])
            + loss(out[2], y2, scaled_anchors[2])
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # todo
        #suggested_lr = self.lr_finder(optimizer, self.criterion)
        steps_per_epoch = len(self.train_dataloader())
        scheduler = OneCycleLR(
                optimizer, max_lr=1e-3,
                steps_per_epoch=steps_per_epoch,
                epochs=config.NUM_EPOCHS,
                pct_start=5 / config.NUM_EPOCHS,
                three_phase=False,
                div_factor=100,
                final_div_factor=100,
                anneal_strategy='linear',)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        return [optimizer], [
                {"scheduler": scheduler, "interval": "step", "frequency": 1}
            ]
    def on_train_epoch_end(self) -> None:
        print(
            f"EPOCH: {self.current_epoch}, "
            +f"Loss: {self.trainer.callback_metrics['train_loss_epoch']}"
        )

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_loader, self.test_loader, _ = get_loaders(
                train_csv_path=config.DATASET + "/train.csv", 
                test_csv_path=config.DATASET + "/test.csv"
            )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.test_loader


if __name__ == '__main__':
    trainer = Trainer(
        callbacks=[
            ModelCheckpoint(
                dirpath=config.CHECKPOINT_PATH,
                save_on_train_epoch_end=True,
                verbose=True,
            ),
            PlotTestExamplesCallback(every_n_epochs=10),
            CheckClassAccuracyCallback(
                train_every_n_epochs=2, 
                test_every_n_epochs=5),
            MAPCallback(every_n_epochs=40),
            LearningRateMonitor(logging_interval="step",
                                log_momentum=True),
        ],
        accelerator=config.DEVICE, devices=2,
        max_epochs = 40,
        enable_progress_bar = True,
        #overfit_batches = 10,
        log_every_n_steps = 10,
        precision='16',
        # limit_train_batches=0.01,
        limit_val_batches=0.25,
        # limit_test_batches=0.01,
        #num_sanity_val_steps = 3
        # detect_anomaly=True
    )

    # Train the model
    ckpt_fname = config.CHECKPOINT_PATH + '/epoch=0-step=1035.ckpt'
    yolo_v3 = YoloV3()
    yolo_v3.load_from_checkpoint(ckpt_fname)
    trainer.fit(yolo_v3)

