import os
from abc import abstractmethod
from torchvision import transforms
from collections.abc import Collection, Callable
from typing import Optional

from ml.core.utils.transforms import Nop, Compose
from ml.core.utils.filetypes import SourceFile

class MedicalFileConfig:

    def __init__(self, is_volume: bool = True):
        self.modality = None
        self.mtransforms = Nop()
        self.is_volume = is_volume

    @abstractmethod
    def get_name(self):
        raise NotImplementedError

class NIFTIDatasetConfig(MedicalFileConfig):

    def __init__(self,
                 sourcefile: SourceFile,
                 modality : str = "T1",
                 return_slices: bool = False,
                 name: str = "MRIDataset", 
                 is_volume: bool = True,
                 mtransforms: Optional[Collection[Callable]] | Optional[Callable] = []):
        """Provide interface for masterfiles and nifti files

        Args:
            nifti_file (str): A file specifying the location of each nifti file of a precise modality, with covariates.
            modality (str, optional): modality of the LMDB. It is used to concat to the folder. Defaults inferred from last name before ext.
            name (str, optional): name of the dataset
            is_volume (bool, optional): if the dataset outputs volumes
            mtransforms (Collection, optional): set of transformations to apply. Defaults to [].
            return_slices (bool, optional): if set, it returns images as slices. Defaults to False.
        Raises:
            FileNotFoundError: Not found the lmdb file
            ValueError: Incorrect parameters provided
        """
        super(NIFTIDatasetConfig, self).__init__(is_volume=is_volume)

        if sourcefile is not None and not sourcefile.exists():
            raise FileNotFoundError(f"File {sourcefile} ({type(sourcefile).__name__}) does not exist!")
        elif sourcefile is not None:
            self.sourcefile = sourcefile

        self.sanity_check()

        modality = self.sourcefile.modality if not modality else modality
        
        if mtransforms is None or len(mtransforms) < 0:
            mtransforms = Nop()
        else:
            mtransforms = Compose(mtransforms)
        
        self.mtransforms = mtransforms
        self.name = name
        self.modality = modality
        self.return_slices = return_slices

    def get_conf(self):
        return {
            "lmdb_file": self.lmdb_file,
            "mtransforms": self.mtransforms
        }
    
    def get_name(self):
        return self.name
    
    def sanity_check(self):
        non_ext = []
        if self.sourcefile:
            paths = self.sourcefile.get_paths()
            non_ext = [path[0] for path in paths if not os.path.exists(path[0])]

        if len(non_ext) > 0:
            with open("missing_niis.txt", 'w') as f:
                f.writelines([l+'\n' for l in non_ext])
            raise FileNotFoundError(f"Found {len(non_ext)} non-existent nifti files. Dumped to missing_niis.txt")

    def get_files_and_cov(self):
        return self.sourcefile.get_paths()           

class LMDBDatasetConfig(MedicalFileConfig):

    def __init__(self, lmdb_folder : str, 
                 lmdb_file: str = None, 
                 modality : str = "T1", 
                 type_img: str = "slices", 
                 name: str = "MRIDataset", 
                 is_volume: bool = True,
                 mtransforms: Optional[Collection[Callable]] | Optional[Callable] = []):
        """_summary_

        Args:
            lmdb_folder (str): location of the LMDB folder.
            lmdb_file (str, optional): Plain location of the lmbd file. Defaults to None.
            modality (str, optional): modality of the LMDB. It is used to concat to the folder. Defaults to "T1".
            type_img (str, optional): slices or volume. Defaults to "slices".
            name (str, optional): name of the dataset
            is_volume (bool, optional): if the dataset outputs volumes
            mtransforms (Collection, optional): set of transformations to apply. Defaults to [].

        Raises:
            FileNotFoundError: Not found the lmdb file
            ValueError: Incorrect parameters provided
        """
        super(LMDBDatasetConfig, self).__init__(is_volume=is_volume)

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
            mtransforms = Nop()
        else:
            mtransforms = Compose(mtransforms)
        
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

class MaskConfig():

    def __init__(self, 
                 scale=100, 
                 octaves=6, 
                 persistence=0.5, 
                 lacunarity=2.0,
                 threshold=0.5, 
                 sigma=1.0, 
                 per_sample=10):
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.threshold = threshold
        self.sigma = sigma
        self.per_sample = per_sample

    def __iter__(self):
        # Return the instance attributes as a dictionary for unpacking
        return iter(self.__dict__.items())

    def to_dict(self, salt=0):
        # Return a dictionary of initialized attributes
        return {key: value for key, value in self.__dict__.items() if key in self.__class__.__init__.__code__.co_varnames and key != "per_sample"}


class ModalityConfig():

    def __init__(self, modality: str, shape: Collection[int], flip: bool, rotate: int = None):
        self.modality = modality
        self.shape = shape
        self.transforms = self.__get_transforms(flip, rotate)

    def __get_transforms(self, flip: bool, rotate: int = None):
        transformations = [transforms.CenterCrop(self.shape[1:])]
        if flip: transformations.append(transforms.RandomHorizontalFlip())
        if rotate: transformations.append(transforms.RandomRotation(rotate))
        return transforms.Compose(transformations)