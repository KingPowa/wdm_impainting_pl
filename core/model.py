import pytorch_lightning as pl
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict
from omegaconf import DictConfig


class BaseLightningModule(pl.LightningModule, ABC):
    def __init__(self, config: DictConfig):
        """
        Initializes the model with a configuration dictionary.
        Args:
            config (DictConfig): Configuration for the model, optimizers, and additional components.
        """
        super(BaseLightningModule, self).__init__()
        self.config = config
        self.save_hyperparameters(config)  # Logs the config for reproducibility
        self.model = self._initialize_model()  # Model architecture
        self.loss_fn = self._initialize_loss_function()  # Loss function
        self.metric_fn = self._initialize_metrics_function()  # Metrics (optional)

    @abstractmethod
    def _initialize_model(self):
        """Initialize model architecture from config. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _initialize_loss_function(self):
        """Initialize the loss function. Must be implemented by subclasses."""
        pass

    def _initialize_metrics_function(self):
        """Optionally initialize additional metrics. Can be overridden by subclasses."""
        return None
    
    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        return self.__step(batch, batch_idx, stage="train")

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        return self.__step(batch, batch_idx, stage="val")
    
    def __step(self, batch: Any, batch_idx: int, stage: str):
        inputs, age_cond = batch
        output = self.forward(inputs, age_cond)
        loss = self.calc_loss(batch, output)
        for loss_n, loss_val in loss.items():
            self.log(
                f"{stage}_{loss_n}", loss_val, on_epoch=True, prog_bar=True, logger=True
            )
        return loss["loss"]

    def configure_optimizers(self):
        # You can define the optimizer here
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.train_config.learning_rate)
        return optimizer
