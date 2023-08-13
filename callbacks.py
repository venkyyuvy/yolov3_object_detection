
import pytorch_lightning as pl

import config
from utils import (
    get_evaluation_bboxes,
    mean_average_precision,
)

class PlotTestExamplesCallback(pl.Callback):
    def __init__(self, every_n_epochs: int = 10):
        super().__init__()
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_start(
        self, 
        trainer: pl.Trainer,
        pl_module: pl.LightningModule):
        if (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            pl_module.plot_images = True
        else:
            pl_module.plot_images = False

class CheckClassAccuracyCallback(pl.Callback):
    def __init__(
        self, train_every_n_epochs: int = 1,
        test_every_n_epochs: int = 3
    ) -> None:
        super().__init__()
        self.train_every_n_epochs = train_every_n_epochs
        self.test_every_n_epochs = test_every_n_epochs

    def on_train_epoch_start(
        self, 
        trainer: pl.Trainer,
            pl_module: pl.LightningModule):
        if (trainer.current_epoch + 1) % self.train_every_n_epochs == 0:
            pl_module.train_accuracy = True
        else:
            pl_module.train_accuracy = False

    def on_test_epoch_start(
        self, 
        trainer: pl.Trainer,
            pl_module: pl.LightningModule):
        if (trainer.current_epoch + 1) % self.test_every_n_epochs == 0:
            pl_module.test_accuracy = True
        else:
            pl_module.test_accuracy = False

class MAPCallback(pl.Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:

        if trainer.current_epoch == trainer.max_epochs:
            pl_module.eval()
            pred_boxes, true_boxes = get_evaluation_bboxes(
                loader=trainer.datamodule.test_dataloader(),
                model=pl_module,
                iou_threshold=config.NMS_IOU_THRESH,
                anchors=config.ANCHORS,
                threshold=config.CONF_THRESHOLD,
            )

            map_val = mean_average_precision(
                pred_boxes=pred_boxes,
                true_boxes=true_boxes,
                iou_threshold=config.MAP_IOU_THRESH,
                box_format="midpoint",
                num_classes=config.NUM_CLASSES,
            ).item()
            pl_module.log('mAP', map_val, logger=True)
            print("+++ MAP: ", map_val)
            pl_module.train()
