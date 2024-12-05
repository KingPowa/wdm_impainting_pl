import os
import lmdb
import pickle
import torch
from abc import ABC, abstractmethod

from torch import nn
from torchvision import transforms
from collections.abc import Collection
from typing import Optional
from torch.utils.data import Dataset

class LMDBDatasetConfig:

    def __init__(self, lmdb_folder : str, 
                 lmdb_file: str = None, 
                 modality : str = "T1", 
                 type_img: str = "slices", 
                 name: str = "MRIDataset", 
                 mtransforms: Optional[Collection[nn.Module]] | Optional[nn.Module] = []):
        """_summary_

        Args:
            lmdb_folder (str): location of the LMDB folder.
            lmdb_file (str, optional): Plain location of the lmbd file. Defaults to None.
            modality (str, optional): modality of the LMDB. It is used to concat to the folder. Defaults to "T1".
            type_img (str, optional): slices or volume. Defaults to "slices".
            name (str, optional): name of the dataset
            mtransforms (Collection, optional): set of transformations to apply. Defaults to [].

        Raises:
            FileNotFoundError: Not found the lmdb file
            ValueError: Incorrect parameters provided
        """
        if lmdb_file is not None:
            self.lmdb_file = lmdb_file
            if not os.path.exists(lmdb_file):
                raise FileNotFoundError(f"File {lmdb_file} does not exist!")
        elif lmdb_folder is not None and modality is not None and type_img is not None:
            self.lmdb_file = os.path.join(lmdb_folder, f"{type_img}_{modality}")
            if not os.path.exists(os.path.join(lmdb_folder, f"{type_img}_{modality}")):
                raise FileNotFoundError(f"File {self.lmdb_file} does not exist!")
        else:
            raise ValueError(f"Incorrect parameter for LMDB provided.")
        
        if mtransforms is None or len(mtransforms) < 0:
            mtransforms = nn.Identity
        else:
            mtransforms = transforms.Compose(mtransforms)
        
        self.mtransforms = mtransforms
        self.name = name
        self.modality = modality

    def get_conf(self):
        return {
            "lmdb_file": self.lmdb_file,
            "mtransforms": self.mtransforms
        }
    
    def get_name(self):
        return self.name
    
class MedicalDataset(Dataset):

    @abstractmethod
    def get_sample(self, index: int):
        pass

# Custom Dataset for loading data from a generic LMDB file
class LMDBDataset(MedicalDataset):
    def __init__(self, config: LMDBDatasetConfig):
        """_summary_

        Args:
            lmdb_folder (str): location of the LMDB folder.
            lmdb_file (str, optional): Plain location of the lmbd file. Defaults to None.
            modality (str, optional): modality of the LMDB. It is used to concat to the folder. Defaults to "T1".
            type_img (str, optional): slices or volume. Defaults to "slices".
            mtransforms (Collection, optional): set of transformations to apply. Defaults to [].

        Raises:
            FileNotFoundError: Not found the lmdb file
            ValueError: Incorrect parameters provided
        """
        self.lmdb_file = config.lmdb_file
        self.mtransforms = config.mtransforms

        self.env = lmdb.open(self.lmdb_file, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']

    def __len__(self):
        return self.length

    def __open_lmdb(self):
        """Open an lmdb file specified in the constructor
        """
        self.env = lmdb.open(
            self.lmdb_file,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.txn = self.env.begin(write=False)

    def __getitem__(self, index: int):
        tensor, age, sex = self.get_sample(index)

        tensor: torch.Tensor = torch.from_numpy(tensor).float()
        if tensor.dim() <= 2:
            tensor = tensor[None]
        tensor = self.mtransforms(tensor)
        return tensor.unsqueeze(1), age, sex
    
    def get_sample(self, index: int):
        if not hasattr(self, "txn"):
            self.__open_lmdb()
        byteflow = self.txn.get(f"{index:08}".encode("ascii"))
        unpacked = pickle.loads(byteflow)

        return unpacked["image"], int(unpacked["age"]), unpacked["sex"]
    
class MRIDataset(LMDBDataset):

    def __init__(self, config: LMDBDatasetConfig, max_age: float, min_age: float = 0):
        super().__init__(config=config)
        self.min_age = min_age
        self.max_age = max_age

    def __getitem__(self, index: int):
        slice, age, sex = super(MRIDataset, self).__getitem__(index)

        # Slice is Modalities x Channels x Width X Length. We add 1 x W x L to Channels
        # Normalize the integer value
        normalized_age = (age - self.min_age) / (self.max_age - self.min_age)

        # Convert gender to binary (0 for M, 1 for F)
        sex_binary = 0 if sex == 'M' else 1

        return slice, torch.tensor([normalized_age, sex_binary])
    
class MRIMaskedDataset(MRIDataset):

    def __init__(self, config: LMDBDatasetConfig, max_age: float, min_age: float = 0, mask_config: MaskConfig = MaskConfig()):
        super().__init__(config=config, max_age=max_age, min_age=min_age)

    def __getitem__(self, index: int):
        slice, age, sex = super(MRIMaskedDataset, self).__getitem__(index)

        slice_shape = ()

        
