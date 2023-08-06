import config
import torch
import torch.optim as optim

from model import YOLOv3

from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    get_loaders,
    plot_couple_examples
)
from loss import YoloLoss
import warnings
warnings.filterwarnings("ignore")
from torch_lr_finder import LRFinder
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy


class YoloV3(LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
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
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)
        self.log("train_loss", loss.item())
        if (batch_idx + 1)  % 100 == 0:
            check_class_accuracy(self.model, self.val_dataloader(),
                                 threshold=config.CONF_THRESHOLD)

            pred_boxes, true_boxes = get_evaluation_bboxes(
                self.test_dataloader,
                self.model,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )

            mapval = mean_average_precision(
                pred_boxes,
                true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            )
            print(f"MAP: {mapval.item()}")
        return loss

    @property
    def get_scaled_anchors(self):
        return (
            torch.tensor(config.ANCHORS)
            * torch.tensor(config.S).unsqueeze(1)\
                .unsqueeze(1).repeat(1, 3, 2)
        ).to(self.device)

    def criterion(self, out, y):
        loss_fn = YoloLoss()
        scaled_anchors = self.get_scaled_anchors
        y0, y1, y2 = (
                y[0].to(self.device),
                y[1].to(self.device),
                y[2].to(self.device),
            )
        loss = (
                    loss_fn(out[0], y0, scaled_anchors[0])
                    + loss_fn(out[1], y1, scaled_anchors[1])
                    + loss_fn(out[2], y2, scaled_anchors[2])
                )
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        # todo
        #suggested_lr = self.lr_finder(optimizer, self.criterion)
        steps_per_epoch = len(self.train_dataloader())
        scheduler_dict = {
            "scheduler":  OneCycleLR(
        optimizer, max_lr=0.05,
        steps_per_epoch=steps_per_epoch,
        epochs=self.config.NUM_EPOCHS,
        pct_start=5/self.config.NUM_EPOCHS,
        three_phase=False,
        div_factor=100,
        final_div_factor=100,
        anneal_strategy='linear',
    ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_loader, self.test_loader, _ = get_loaders(
                train_csv_path=self.config.DATASET + "/train.csv", 
                test_csv_path=self.config.DATASET + "/test.csv"
            )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.test_loader


# initialize the trainer
if __name__ == '__main__':
    trainer = Trainer(
        accelerator="mps", devices=1,
        max_epochs = 20,
        enable_progress_bar = True,
        #overfit_batches = 10,
        log_every_n_steps = 20,
        # limit_train_batches=0.01,
        # limit_test_batches=0.01,
        #num_sanity_val_steps = 3
    )

    # Train the model
    yolo_v3 = YoloV3(config)
    trainer.fit(yolo_v3)
