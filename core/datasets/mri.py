import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as tr

from typing import Optional, Collection

from .file_based import MedicalDataset
from ml.core.utils.masking import generate_perlin_mask_with_contour_smoothing, intersect_noise_with_brain
from ml.core.configuration import MaskConfig
    
class MRIDataset(Dataset):

    def __init__(self, dataset: MedicalDataset, 
                 max_age: float, 
                 min_age: float = 0,
                 transforms: Optional[Collection[torch.nn.Module]] | Optional[torch.nn.Module] = []):
        self.min_age = min_age
        self.max_age = max_age
        self.dataset = dataset
        self.length = len(dataset)
        self.mtransforms = tr.Compose(transforms) if transforms else torch.nn.Identity()

    def __len__(self):
        return self.length
    
    def get_sample(self, index: int):
        slice, age, sex = self.dataset[index]
        # Slice is Modalities x Channels x Width X Length. We add 1 x W x L to Channels
        # Normalize the integer value
        normalized_age = (age - self.min_age) / (self.max_age - self.min_age)
        # Convert gender to binary (0 for M, 1 for F)
        sex_binary = 0 if sex == 'M' else 1
        return slice, torch.tensor([normalized_age, sex_binary])

    def __getitem__(self, index: int):
        image, cond = self.get_sample(index)
        image = torch.from_numpy(image).float()
        image = self.mtransforms(image)

        return image.unsqueeze(0), cond # Add channel information
    
class MRIMaskedDataset(MRIDataset):

    def __init__(self, dataset: MedicalDataset, 
                 max_age: float, 
                 min_age: float = 0, 
                 mask_config: MaskConfig = MaskConfig(),
                 transforms: Optional[Collection[torch.nn.Module]] | Optional[torch.nn.Module] = []):
        super().__init__(dataset=dataset, max_age=max_age, min_age=min_age, transforms=transforms)
        self.mask_config = mask_config
        self.length = self.length * mask_config.per_sample

    def __getitem__(self, index: int):
        slice, cond = self.get_sample(index // self.mask_config.per_sample)
        slice_shape = slice.shape # (H, W)
        # mask = np.zeros((1))
        # Consider deterministic salt in the configuration to account for 0 masks, until it is non 0
        # salt = 0
        # Note: seed = index since this is the EXTENDED index, counting both the num of samples that the same sample
        # while mask.all(arr=0):
        mask = generate_perlin_mask_with_contour_smoothing(image_size=slice_shape, **self.mask_config.to_dict(), seed=index) # salt=salt), seed=index) # (n, H, W)
        mask = intersect_noise_with_brain(image=slice, mask=mask)
        #    salt += 0.01
        # Add channel
        mask = np.expand_dims(mask, axis=0)

        image = torch.from_numpy(slice).float()
        image = self.mtransforms(image)
        return image.unsqueeze(0), mask, cond

        
        
