# Final training fileimport os
from sklearn.model_selection import KFold
from pytorch_lightning import Trainer, LightningModule
from logging import Logger

from ml.core.dataloaders.dataloaders import MRIHoldoutDataLoader, MRIKFoldDataLoader
from ml.core.configuration import *
from ml.core.logging import AdvancedWandLogger, AdvancedModelCheckpoint, Session

# Cross-validation support
def cross_validation_split(dataset, n_splits=5):
    kfold = KFold(n_splits=n_splits, shuffle=True)
    splits = list(kfold.split(dataset))
    return splits

def train(model: LightningModule, dataloader: MRIHoldoutDataLoader, session: Session, logger: Logger):
    logger.info(f"Starting training of {model._get_name()} - HOLDOUT")
    
    logger.info(f"################## Dataset Description ##################")
    for key, element in dataloader.info():
        logger.info(f"\t{key} : {element}")
    logger.info(f"#########################################################")

    logger.info(f"Setting up Wandbboard")
    wand_logger = AdvancedWandLogger(model, session)
    checkpoint_callback = AdvancedModelCheckpoint(session=session,
                                            filename_suffix='holdout',
                                            monitor='val_loss',
                                            mode='min')
    logger.info(f"Setting up Trainer")
    trainer = Trainer(
        max_epochs=session.config.train_config.epochs,
        devices=session.devices,
        accelerator=session.accelerator,
        logger=wand_logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=(not session.is_slurm())
    )

    logger.info(f"Starting trainer")
    trainer.fit(model, datamodule=dataloader)

    performances = trainer.callback_metrics
    logger.info(f"Finished. Total performance: {performances}")