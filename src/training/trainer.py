import os
import time
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils.stats_aggregator import StatsAggregator


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        device,
        config,
    ):
        self.model = model
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = config.epochs
        self.optimizer = optimizer
        self.patience = config.patience
        self.model_ckpt_path = config.model_ckpt_path
        self.loss_threshold = config.loss_threshold
        self.train_loss_epoch_stats = StatsAggregator()
        self.val_loss_epoch_stats = StatsAggregator()
        self.train_loss_history = []
        self.val_loss_history = []
        self.timestr = time.strftime("%m%d-%H%M%S")

    def run(self):
        self.model.to(self.device)
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1} of {self.epochs}")
            self.train_loss_epoch_stats.reset()
            self.val_loss_epoch_stats.reset()

            self.train_epoch()
            self.validate_epoch()

            self.train_loss_epoch_stats.print_training_loss(epoch)
            self.val_loss_epoch_stats.print_validation_loss(epoch)

            self.train_loss_history.append(self.train_loss_epoch_stats.value)
            self.val_loss_history.append(self.val_loss_epoch_stats.value)

            self.val_loss_epoch_stats.check_lowest_val_loss(epoch)
            self._check_patience(epoch)

        self._plot_loss()
        self.val_loss_epoch_stats.print_lowest_val_loss()

    def _check_patience(self, epoch):
        if len(self.val_loss_history) > self.patience:
            for val_loss in self.val_loss_history[-self.patience :]:
                if self._check_val_loss(val_loss):
                    self._save_model(epoch)

    def _check_val_loss(self, val_loss):
        return val_loss < (
            self.val_loss_history[-self.patience - 1] - self.loss_threshold
        )

    def _save_model(self, epoch):
        torch.save(
            self.model.state_dict(),
            os.path.join(
                self.model_ckpt_path,
                f"{self.timestr}_epoch_{epoch+1}.pth",
            ),
        )

    def _plot_loss(self):
        plt.plot(self.train_loss_history, label="Training Loss")
        plt.plot(self.val_loss_history, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title(f"Train & Val Loss: {self.timestr}")
        plt.savefig(os.path.join("visuals", "loss_plots", f"loss_{self.timestr}.png"))

    def train_epoch(self):
        self.model.train()

        loop = tqdm(self.train_dataloader, total=len(self.train_dataloader))
        for images, targets in loop:
            self.optimizer.zero_grad()
            images = [image.to(self.device) for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            self.train_loss_epoch_stats.send(loss_value)

            losses.backward()
            self.optimizer.step()

            loop.set_postfix(running_train_loss=f"{loss_value:.4f}")

    def validate_epoch(self):
        loop = tqdm(self.val_dataloader, total=len(self.val_dataloader))
        for images, targets in loop:
            images = [image.to(self.device) for image in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            with torch.no_grad():
                loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            self.val_loss_epoch_stats.send(loss_value)

            loop.set_postfix(running_val_loss=f"{loss_value:.4f}")
