from torch.utils.data import DataLoader
from torch import Generator

from .mri import MRIDataloader
from ml.core.datasets.file_based import MedicalDataset
from ml.core.datasets.mri import MRIMaskedDataset
from ml.core.configuration import MaskConfig

class MRIInpaintDataLoader(MRIDataloader):
    
    def __init__(self,
                 dataset: MedicalDataset,
                 max_age: float,
                 min_age: float = 0,
                 seed: int = 11111,
                 num_workers: int = 15,
                 batch_size: int = 16,
                 mask_config: MaskConfig = MaskConfig()):
        # Now explicitly call the parent constructor with all parameters
        super(MRIInpaintDataLoader, self).__init__(dataset=dataset, 
                         max_age=max_age, 
                         min_age=min_age, 
                         seed=seed,
                         batch_size=batch_size)
        self.save_hyperparameters(ignore="dataset", logger=False)
        

    def setup(self, stage=None):
        if not self.train_set:
            self.train_set = MRIMaskedDataset(self.dataset, self.hparams.max_age, self.hparams.min_age, mask_config=self.hparams.mask_config)

    def train_dataloader(self):
        return DataLoader(self.train_set, num_workers=self.hparams.num_workers, batch_size=self.hparams.batch_size,
                          shuffle=True, generator=Generator().manual_seed(self.hparams.seed))

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError
    